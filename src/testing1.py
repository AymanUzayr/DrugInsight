import pandas as pd
import numpy as np

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading files...")
interactions = pd.read_csv('data/processed/drugbank_interactions_filtered.csv')
enzymes      = pd.read_csv('data/processed/drugbank_enzymes.csv')
targets      = pd.read_csv('data/processed/drugbank_targets.csv')
twosides     = pd.read_csv('data/processed/twosides_features_filtered.csv')
rxnorm       = pd.read_csv('data/processed/rxnorm_bridge.csv')

# ── Shared enzyme count ────────────────────────────────────────────────────────
print("Building enzyme sets...")
drug_enzymes = enzymes.groupby('drugbank_id')['enzyme_id'].apply(set).to_dict()

# ── Shared target count ────────────────────────────────────────────────────────
print("Building target sets...")
drug_targets = targets.groupby('drugbank_id')['target_id'].apply(set).to_dict()

# ── Vectorized shared counts ───────────────────────────────────────────────────
print("Computing shared enzyme/target counts...")

def compute_shared(df, lookup):
    counts = []
    for _, row in df.iterrows():
        a = lookup.get(row['drug_1_id'], set())
        b = lookup.get(row['drug_2_id'], set())
        counts.append(len(a & b))
    return counts

interactions['shared_enzyme_count'] = compute_shared(interactions, drug_enzymes)
interactions['shared_target_count'] = compute_shared(interactions, drug_targets)

print(f"  Enzyme count > 0: {(interactions['shared_enzyme_count'] > 0).sum()}")
print(f"  Target count > 0: {(interactions['shared_target_count'] > 0).sum()}")

# ── Twosides join via RxNorm bridge ───────────────────────────────────────────
print("Joining twosides PRR...")

# Map drugbank_id → rxnorm_id
db_to_rx = dict(zip(rxnorm['drugbank_id'], rxnorm['rxnorm_id'].astype(str)))

interactions['rx_1'] = interactions['drug_1_id'].map(db_to_rx)
interactions['rx_2'] = interactions['drug_2_id'].map(db_to_rx)

# Get max PRR per drug pair from twosides
twosides['drug_1_rxnorn_id'] = twosides['drug_1_rxnorn_id'].astype(str)
twosides['drug_2_rxnorm_id'] = twosides['drug_2_rxnorm_id'].astype(str)

ts_max = (twosides
    .groupby(['drug_1_rxnorn_id', 'drug_2_rxnorm_id'])['PRR']
    .max()
    .reset_index()
    .rename(columns={
        'drug_1_rxnorn_id': 'rx_1',
        'drug_2_rxnorm_id': 'rx_2',
        'PRR': 'max_PRR'
    })
)

# Merge both directions (A→B and B→A)
ts_rev = ts_max.rename(columns={'rx_1': 'rx_2', 'rx_2': 'rx_1'})
ts_both = pd.concat([ts_max, ts_rev], ignore_index=True).drop_duplicates()

interactions = interactions.merge(ts_both, on=['rx_1', 'rx_2'], how='left')
interactions['max_PRR']       = interactions['max_PRR'].fillna(0.0)
interactions['twosides_found'] = (interactions['max_PRR'] > 0).astype(int)

print(f"  Twosides found > 0: {(interactions['twosides_found'] > 0).sum()}")

# ── Cleanup & save ─────────────────────────────────────────────────────────────
interactions = interactions.drop(columns=['rx_1', 'rx_2'])

print("\nFinal feature stats:")
print(interactions[['shared_enzyme_count', 'shared_target_count',
                     'max_PRR', 'twosides_found']].describe())

out_path = 'data/processed/drugbank_interactions_enriched.csv'
interactions.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print(f"Shape: {interactions.shape}")