from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from mol_graph      import smiles_to_graph
from gnn_encoder    import GNNEncoder
from ddi_classifier import DDIClassifier
import os

os.makedirs('models', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class DDIDataset(Dataset):
    def __init__(self, df, graph_cache):
        self.df          = df.reset_index(drop=True)
        self.graph_cache = graph_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        graph_a = self.graph_cache.get(row['drug_1_id'])
        graph_b = self.graph_cache.get(row['drug_2_id'])

        if graph_a is None or graph_b is None:
            return None

        # Normalised extra features
        extra = torch.tensor([
            min(float(row.get('shared_enzyme_count', 0) or 0), 21.0) / 21.0,
            min(float(row.get('shared_target_count', 0) or 0), 36.0) / 36.0,
            min(float(row.get('max_PRR', 0.0) or 0.0), 50.0) / 50.0,
            float(row.get('twosides_found', 0) or 0),
        ], dtype=torch.float)

        label = torch.tensor(row['label'], dtype=torch.long)
        return graph_a, graph_b, extra, label


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    graphs_a, graphs_b, extras, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs_a),
        Batch.from_data_list(graphs_b),
        torch.stack(extras),
        torch.stack(labels)
    )


# ── Load & Filter Data ─────────────────────────────────────────────────────────
print("Loading data...")

def is_valid_smiles(smi):
    mol = Chem.MolFromSmiles(str(smi).strip())
    return mol is not None

smiles_df = pd.read_csv('data/processed/drugbank_smiles_filtered.csv')
smiles_df = smiles_df[smiles_df['smiles'].apply(is_valid_smiles)]
smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))
print(f"Valid drugs: {len(smiles_dict)}")

print("Pre-computing molecular graphs...")
graph_cache = {}
for drug_id, smi in smiles_dict.items():
    g = smiles_to_graph(smi)
    if g is not None:
        graph_cache[drug_id] = g
print(f"Cached {len(graph_cache)} graphs")

# ── Enzyme/target lookups ──────────────────────────────────────────────────────
print("Building enzyme/target lookups...")
enzymes_df   = pd.read_csv('data/processed/drugbank_enzymes.csv')
targets_df   = pd.read_csv('data/processed/drugbank_targets.csv')
drug_enzymes = enzymes_df.groupby('drugbank_id')['enzyme_id'].apply(set).to_dict()
drug_targets = targets_df.groupby('drugbank_id')['target_id'].apply(set).to_dict()

interactions = pd.read_csv('data/processed/drugbank_interactions_enriched.csv')
interactions = interactions[
    interactions['drug_1_id'].isin(smiles_dict) &
    interactions['drug_2_id'].isin(smiles_dict)
]
#remove for full run
# interactions = interactions.sample(n=50000, random_state=42)   
print(f"Interactions loaded: {len(interactions)}")
interactions['label'] = 1


# ── 1. Drug-level split ────────────────────────────────────────────────────────
all_drugs = list(set(interactions['drug_1_id']) | set(interactions['drug_2_id']))
train_drugs, val_drugs = train_test_split(all_drugs, test_size=0.2, random_state=42)
train_drugs, val_drugs = set(train_drugs), set(val_drugs)

train_pos = interactions[
    interactions['drug_1_id'].isin(train_drugs) &
    interactions['drug_2_id'].isin(train_drugs)
].copy()

#both drugs unseen in val set
val_pos = interactions[        
    interactions['drug_1_id'].isin(val_drugs) &   
    interactions['drug_2_id'].isin(val_drugs)
].copy()

# ── One unseen drug ────────────────────────────────────────
# val_pos = interactions[
#     (interactions['drug_1_id'].isin(val_drugs) |
#      interactions['drug_2_id'].isin(val_drugs)) &
#     ~(interactions['drug_1_id'].isin(val_drugs) &
#       interactions['drug_2_id'].isin(val_drugs))
# ].copy()

print(f"Train positives: {len(train_pos)} | Val positives: {len(val_pos)}")


# ── 2. Negative sampling with real enzyme/target features ─────────────────────
def sample_negatives(drug_pool, pos_pairs_global, n, seed=42):
    np.random.seed(seed)
    drug_pool   = list(drug_pool)
    neg_samples = []
    candidates  = np.random.choice(drug_pool, size=(n * 5, 2))
    for a, b in candidates:
        if (a != b
                and (a, b) not in pos_pairs_global
                and (b, a) not in pos_pairs_global):
            neg_samples.append({
                'drug_1_id': a,
                'drug_2_id': b,
                'label': 0,
                'shared_enzyme_count': len(drug_enzymes.get(a, set()) & drug_enzymes.get(b, set())),
                'shared_target_count': len(drug_targets.get(a, set()) & drug_targets.get(b, set())),
                'max_PRR': 0.0,
                'twosides_found': 0
            })
        if len(neg_samples) >= n:
            break
    return pd.DataFrame(neg_samples)

pos_pairs_global = set(zip(interactions['drug_1_id'], interactions['drug_2_id']))

train_neg = sample_negatives(train_drugs, pos_pairs_global, n=len(train_pos), seed=42)
val_neg   = sample_negatives(val_drugs,   pos_pairs_global, n=len(val_pos),   seed=43)

print(f"Train negatives: {len(train_neg)} | Val negatives: {len(val_neg)}")


# ── 3. Assemble & shuffle ──────────────────────────────────────────────────────
train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)
val_df   = pd.concat([val_pos,   val_neg],   ignore_index=True).sample(frac=1, random_state=42)

for col in ['shared_enzyme_count', 'shared_target_count', 'max_PRR', 'twosides_found']:
    for d in [train_df, val_df]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = d[col].fillna(0.0)


# ── 4. Sanity check ───────────────────────────────────────────────────────────
train_ids = set(train_df['drug_1_id']) | set(train_df['drug_2_id'])
val_ids   = set(val_df['drug_1_id'])   | set(val_df['drug_2_id'])
print(f"Drug overlap: {len(train_ids & val_ids)}")  # must be 0
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

train_ds = DDIDataset(train_df, graph_cache)
val_ds   = DDIDataset(val_df,   graph_cache)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                          collate_fn=collate_fn, num_workers=0, pin_memory=True)


# ── Models ─────────────────────────────────────────────────────────────────────
gnn        = GNNEncoder().to(DEVICE)
classifier = DDIClassifier(extra_features=4).to(DEVICE)

optimizer = torch.optim.Adam(
    list(gnn.parameters()) + list(classifier.parameters()),
    lr=1e-4, weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

criterion = nn.BCEWithLogitsLoss()

# Embedding sanity check
gnn.eval()
with torch.no_grad():
    sample_batch = next(iter(train_loader))
    if sample_batch:
        g_a, g_b, extra, labels = sample_batch
        g_a = g_a.to(DEVICE)
        emb = gnn(g_a)
        print(f"Embedding mean: {emb.mean().item():.4f}")
        print(f"Embedding std:  {emb.std().item():.4f}")
        print(f"Any NaN:        {torch.isnan(emb).any()}")
        print(f"Extra sample:   {extra[0]}")


# ── Training loop ──────────────────────────────────────────────────────────────
def train_epoch(loader):
    gnn.train(); classifier.train()
    total_loss = 0

    for batch in loader:
        if batch is None:
            continue
        g_a, g_b, extra, labels = batch
        g_a    = g_a.to(DEVICE)
        g_b    = g_b.to(DEVICE)
        extra  = extra.to(DEVICE)
        labels = labels.float().to(DEVICE)

        embed_a = gnn(g_a)
        embed_b = gnn(g_b)

        prob, _ = classifier(embed_a, embed_b, extra)
        prob = prob.view(-1)
        loss = criterion(prob, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gnn.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(loader):
    gnn.eval(); classifier.eval()
    correct = total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            g_a, g_b, extra, labels = batch
            g_a    = g_a.to(DEVICE)
            g_b    = g_b.to(DEVICE)
            extra  = extra.to(DEVICE)
            labels = labels.to(DEVICE)

            embed_a = gnn(g_a)
            embed_b = gnn(g_b)
            prob, _ = classifier(embed_a, embed_b, extra)
            prob  = torch.sigmoid(prob).view(-1)
            preds = (prob > 0.5).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for pred, label in zip(preds, labels):
                if   label == 1 and pred == 1: tp += 1
                elif label == 0 and pred == 0: tn += 1
                elif label == 1 and pred == 0: fn += 1
                elif label == 0 and pred == 1: fp += 1

    print(f"  TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    return correct / total if total > 0 else 0


# ── Run training ───────────────────────────────────────────────────────────────
EPOCHS   = 15
best_acc = 0

for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(train_loader)
    acc  = eval_epoch(val_loader)
    scheduler.step(acc)

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'gnn':        gnn.state_dict(),
            'classifier': classifier.state_dict()
        }, 'models/ddi_model.pt')
        print(f"  Saved best model (acc={best_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")