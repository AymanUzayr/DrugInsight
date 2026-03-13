import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mol_graph    import smiles_to_graph
from gnn_encoder  import GNNEncoder
from ddi_classifier import DDIClassifier
import os

os.makedirs('models', exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────
class DDIDataset(Dataset):
    def __init__(self, df, smiles_dict):
        self.df          = df.reset_index(drop=True)
        self.smiles_dict = smiles_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        smiles_a = self.smiles_dict.get(row['drug_1_id'])
        smiles_b = self.smiles_dict.get(row['drug_2_id'])

        graph_a  = smiles_to_graph(smiles_a) if smiles_a else None
        graph_b  = smiles_to_graph(smiles_b) if smiles_b else None

        if graph_a is None or graph_b is None:
            return None

        # Extra features: has_mechanism, num_shared_enzymes,
        #                 num_shared_targets, twosides_prr, twosides_found
        extra = torch.tensor([
            float(pd.notna(row.get('mechanism', None))),
            float(row.get('shared_enzyme_count', 0) or 0),   # or 0 handles NaN
            float(row.get('shared_target_count', 0) or 0),
            float(row.get('max_PRR', 0.0) or 0.0),
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


# ── Load Data ──────────────────────────────────────────────
print("Loading data...")

smiles_df   = pd.read_csv('data/processed/drugbank_smiles_filtered.csv')
smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))

interactions = pd.read_csv('data/processed/drugbank_interactions_filtered.csv')
interactions = interactions[
    interactions['drug_1_id'].isin(smiles_dict) &
    interactions['drug_2_id'].isin(smiles_dict)
]
print(f"Interactions with full SMILES coverage: {len(interactions)}")
interactions['label'] = 1

# ── 1. DRUG-LEVEL SPLIT (fix for memorization) ────────────────────────────────
all_drugs = list(set(interactions['drug_1_id']) | set(interactions['drug_2_id']))
train_drugs, val_drugs = train_test_split(all_drugs, test_size=0.2, random_state=42)
train_drugs, val_drugs = set(train_drugs), set(val_drugs)

train_pos = interactions[
    interactions['drug_1_id'].isin(train_drugs) &
    interactions['drug_2_id'].isin(train_drugs)
].copy()

val_pos = interactions[
    interactions['drug_1_id'].isin(val_drugs) &
    interactions['drug_2_id'].isin(val_drugs)
].copy()

print(f"Train positives: {len(train_pos)} | Val positives: {len(val_pos)}")

# ── 2. NEGATIVE SAMPLING WITHIN EACH SPLIT (no cross-contamination) ───────────
def sample_negatives(pos_df, drug_pool, pos_pairs_global, n, seed=42):
    np.random.seed(seed)
    drug_pool = list(drug_pool)
    neg_samples = []
    candidates  = np.random.choice(drug_pool, size=(n * 5, 2))
    for a, b in candidates:
        if (a != b
                and (a, b) not in pos_pairs_global
                and (b, a) not in pos_pairs_global):
            neg_samples.append({
                'drug_1_id': a, 'drug_2_id': b,
                'mechanism': None, 'label': 0,
                'shared_enzyme_count': 0, 'shared_target_count': 0,
                'max_PRR': 0.0, 'twosides_found': 0
            })
        if len(neg_samples) >= n:
            break
    return pd.DataFrame(neg_samples)

pos_pairs_global = set(zip(interactions['drug_1_id'], interactions['drug_2_id']))

train_neg = sample_negatives(train_pos, train_drugs, pos_pairs_global, n=len(train_pos), seed=42)
val_neg   = sample_negatives(val_pos,   val_drugs,   pos_pairs_global, n=len(val_pos),   seed=43)

print(f"Train negatives: {len(train_neg)} | Val negatives: {len(val_neg)}")

# ── 3. ASSEMBLE & SHUFFLE ─────────────────────────────────────────────────────
train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42)
val_df   = pd.concat([val_pos,   val_neg],   ignore_index=True).sample(frac=1, random_state=42)

for col in ['shared_enzyme_count', 'shared_target_count', 'max_PRR', 'twosides_found']:
    for d in [train_df, val_df]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = d[col].fillna(0.0)

# ── 4. SANITY CHECK ───────────────────────────────────────────────────────────
train_ids = set(train_df['drug_1_id']) | set(train_df['drug_2_id'])
val_ids   = set(val_df['drug_1_id'])   | set(val_df['drug_2_id'])
print(f"Drug overlap after fix: {len(train_ids & val_ids)}")  # must be 0

train_ds = DDIDataset(train_df, smiles_dict)
val_ds   = DDIDataset(val_df,   smiles_dict)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── Models ─────────────────────────────────────────────────
gnn        = GNNEncoder().to(DEVICE)
classifier = DDIClassifier().to(DEVICE)

optimizer  = torch.optim.Adam(
    list(gnn.parameters()) + list(classifier.parameters()),
    lr=1e-4, weight_decay=1e-5
)

criterion = nn.BCEWithLogitsLoss()
# Add this before the training loop
print("Checking for NaN in data...")
print(f"Interactions NaN cols: {interactions.isnull().sum()}")
print(f"Sample extra features from first batch:")

for batch in train_loader:
    if batch is None:
        continue
    g_a, g_b, extra, labels = batch
    print(f"Extra features: {extra[:3]}")
    print(f"Labels: {labels[:3]}")
    print(f"Any NaN in extra: {torch.isnan(extra).any()}")
    print(f"Any NaN in labels: {torch.isnan(labels.float()).any()}")
    break




# ── Training Loop ──────────────────────────────────────────
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
        prob = prob.squeeze()   
        loss    = criterion(prob.squeeze(), labels)

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
            prob = torch.sigmoid(prob)
            print(f"  prob range: {prob.min().item():.4f} – {prob.max().item():.4f}")
            preds = (prob.squeeze() > 0.5).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0


# ── Run Training ───────────────────────────────────────────
EPOCHS = 5
best_acc = 0
# Quick check — paste output of this
train_drug_ids = set(train_df['drug_1_id']) | set(train_df['drug_2_id'])
val_drug_ids   = set(val_df['drug_1_id'])   | set(val_df['drug_2_id'])
overlap = train_drug_ids & val_drug_ids
print(f"Train drugs: {len(train_drug_ids)}")
print(f"Val drugs:   {len(val_drug_ids)}")
print(f"Overlap:     {len(overlap)}")
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(train_loader)
    acc  = eval_epoch(val_loader)

    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'gnn': gnn.state_dict(),
            'classifier': classifier.state_dict()
        }, 'models/ddi_model.pt')
        print(f"  Saved best model (acc={best_acc:.4f})")

print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
