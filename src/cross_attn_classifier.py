import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch


class CrossAttentionLayer(nn.Module):
    """
    Multi-head cross-attention between two sets of atom embeddings.
    Query from molecule A, Key/Value from molecule B (and vice versa).
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # Projections for A→B attention
        self.q_a = nn.Linear(embed_dim, embed_dim)
        self.k_b = nn.Linear(embed_dim, embed_dim)
        self.v_b = nn.Linear(embed_dim, embed_dim)
        self.out_a = nn.Linear(embed_dim, embed_dim)

        # Projections for B→A attention
        self.q_b = nn.Linear(embed_dim, embed_dim)
        self.k_a = nn.Linear(embed_dim, embed_dim)
        self.v_a = nn.Linear(embed_dim, embed_dim)
        self.out_b = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_a  = nn.LayerNorm(embed_dim)
        self.norm_b  = nn.LayerNorm(embed_dim)

    def _attention(self, q, k, v):
        """Scaled dot-product attention. q: [B, N, D], k/v: [B, M, D]"""
        B, N, D = q.shape
        M       = k.shape[1]
        H       = self.num_heads
        d       = self.head_dim

        q = q.view(B, N, H, d).transpose(1, 2)   # [B, H, N, d]
        k = k.view(B, M, H, d).transpose(1, 2)   # [B, H, M, d]
        v = v.view(B, M, H, d).transpose(1, 2)   # [B, H, M, d]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, M]
        attn   = self.dropout(torch.softmax(scores, dim=-1))

        out = torch.matmul(attn, v)               # [B, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return out, attn

    def forward(self, atoms_a, atoms_b):
        """
        atoms_a: [B, N_a, D]  — padded atom embeddings for drug A
        atoms_b: [B, N_b, D]  — padded atom embeddings for drug B
        Returns context-aware embeddings for A and B.
        """
        # A attends to B
        q_a  = self.q_a(atoms_a)
        k_b_ = self.k_b(atoms_b)
        v_b_ = self.v_b(atoms_b)
        ctx_a, attn_ab = self._attention(q_a, k_b_, v_b_)
        ctx_a = self.norm_a(atoms_a + self.out_a(ctx_a))

        # B attends to A
        q_b  = self.q_b(atoms_b)
        k_a_ = self.k_a(atoms_a)
        v_a_ = self.v_a(atoms_a)
        ctx_b, attn_ba = self._attention(q_b, k_a_, v_a_)
        ctx_b = self.norm_b(atoms_b + self.out_b(ctx_b))

        return ctx_a, ctx_b, attn_ab


def pad_atom_embeddings(embeddings_list, embed_dim):
    """
    Pad variable-length atom embedding lists to same length for batched attention.
    Returns padded tensor [B, max_N, D] and mask [B, max_N].
    """
    lengths = [e.shape[0] for e in embeddings_list]
    max_len = max(lengths)
    B       = len(embeddings_list)

    padded = torch.zeros(B, max_len, embed_dim, device=embeddings_list[0].device)
    mask   = torch.zeros(B, max_len, dtype=torch.bool, device=embeddings_list[0].device)

    for i, (emb, length) in enumerate(zip(embeddings_list, lengths)):
        padded[i, :length] = emb
        mask[i, :length]   = True

    return padded, mask


class CrossAttnDDIClassifier(nn.Module):
    """
    Cross-attention DDI classifier.

    Takes atom-level embeddings from two drugs (as PyG batch outputs),
    applies multi-head cross-attention so each drug's atoms can attend
    to the other drug's atoms, pools the context-aware embeddings,
    then predicts interaction probability and severity.
    """

    def __init__(self,
                 embed_dim=256,
                 extra_features=6,
                 num_attn_heads=8,
                 num_attn_layers=2,
                 dropout=0.3):
        super().__init__()

        self.embed_dim = embed_dim

        # Stack of cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_attn_heads, dropout)
            for _ in range(num_attn_layers)
        ])

        # Feed-forward after attention (per molecule)
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # MLP classifier trunk
        # Input: pooled_a + pooled_b + extra = 256 + 256 + 6 = 518
        input_dim = embed_dim * 2 + extra_features

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Heads
        self.prob_head     = nn.Linear(256, 1)
        self.severity_head = nn.Linear(256, 3)   # Minor / Moderate / Major

    def _masked_mean_pool(self, embeddings, mask):
        """
        Mean pool over valid (non-padded) atom positions.
        embeddings: [B, N, D], mask: [B, N] bool
        Returns: [B, D]
        """
        mask_f = mask.unsqueeze(-1).float()           # [B, N, 1]
        summed = (embeddings * mask_f).sum(dim=1)     # [B, D]
        counts = mask_f.sum(dim=1).clamp(min=1.0)    # [B, 1]
        return summed / counts

    def forward(self, atom_embeds_a, batch_a, atom_embeds_b, batch_b, extra):
        """
        atom_embeds_a: [N_total_a, D] — all atoms from batch of drug A graphs
        batch_a:       [N_total_a]    — batch assignment vector from PyG
        atom_embeds_b: [N_total_b, D]
        batch_b:       [N_total_b]
        extra:         [B, extra_features]
        """
        # Split into per-molecule lists
        atoms_a_list = unbatch(atom_embeds_a, batch_a)  # list of [N_i, D]
        atoms_b_list = unbatch(atom_embeds_b, batch_b)

        # Pad for batched cross-attention
        atoms_a_pad, mask_a = pad_atom_embeddings(atoms_a_list, self.embed_dim)
        atoms_b_pad, mask_b = pad_atom_embeddings(atoms_b_list, self.embed_dim)

        # Apply cross-attention layers
        ctx_a, ctx_b = atoms_a_pad, atoms_b_pad
        for layer in self.cross_attn_layers:
            ctx_a, ctx_b, _ = layer(ctx_a, ctx_b)

        # Feed-forward refinement
        ctx_a = self.ffn_a(ctx_a)
        ctx_b = self.ffn_b(ctx_b)

        # Masked mean pooling → molecule-level vectors
        pool_a = self._masked_mean_pool(ctx_a, mask_a)   # [B, D]
        pool_b = self._masked_mean_pool(ctx_b, mask_b)   # [B, D]

        # Fuse with extra features and classify
        x = torch.cat([pool_a, pool_b, extra], dim=-1)
        x = self.trunk(x)

        prob     = self.prob_head(x)
        severity = self.severity_head(x)

        return prob, severity
