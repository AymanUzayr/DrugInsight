import pandas as pd

df = pd.read_csv('data/processed/drugbank_interactions_enriched.csv')

mask = (
    (df['drug_1_id'] == 'DB00331') & (df['drug_2_id'] == 'DB00722')
) | (
    (df['drug_1_id'] == 'DB00722') & (df['drug_2_id'] == 'DB00331')
)
print(df[mask][['drug_1_id', 'drug_2_id', 'mechanism', 'shared_enzyme_count', 'shared_target_count']])