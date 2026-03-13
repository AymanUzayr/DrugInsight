import pandas as pd

bridge = pd.read_csv('data/processed/rxnorm_bridge.csv')
bridge['rxnorm_id'] = bridge['rxnorm_id'].astype(str)

# ── Filter DrugBank interactions to only bridged drugs ──
interactions = pd.read_csv('data/processed/drugbank_interactions.csv')
bridged_ids = set(bridge['drugbank_id'])

interactions_filtered = interactions[
    (interactions['drug_1_id'].isin(bridged_ids)) &
    (interactions['drug_2_id'].isin(bridged_ids))
]

print(f"Original interactions: {len(interactions)}")
print(f"Filtered interactions: {len(interactions_filtered)}")

# ── Filter TWOSIDES to only bridged drugs ──
twosides = pd.read_csv('data/processed/twosides_processed.csv')
twosides['drug_1_rxnorn_id'] = twosides['drug_1_rxnorn_id'].astype(str)
twosides['drug_2_rxnorm_id'] = twosides['drug_2_rxnorm_id'].astype(str)

bridged_rxnorm = set(bridge['rxnorm_id'])

twosides_filtered = twosides[
    (twosides['drug_1_rxnorn_id'].isin(bridged_rxnorm)) &
    (twosides['drug_2_rxnorm_id'].isin(bridged_rxnorm))
]

print(f"Original TWOSIDES pairs: {len(twosides)}")
print(f"Filtered TWOSIDES pairs: {len(twosides_filtered)}")

# ── Filter SMILES to only bridged drugs ──
smiles = pd.read_csv('data/processed/drugbank_smiles.csv')
smiles_filtered = smiles[smiles['drugbank_id'].isin(bridged_ids)]

print(f"Original SMILES: {len(smiles)}")
print(f"Filtered SMILES: {len(smiles_filtered)}")

# ── Save all filtered versions ──
interactions_filtered.to_csv('data/processed/drugbank_interactions_filtered.csv', index=False)
twosides_filtered.to_csv('data/processed/twosides_processed_filtered.csv', index=False)
smiles_filtered.to_csv('data/processed/drugbank_smiles_filtered.csv', index=False)

print("\nAll filtered files saved.")