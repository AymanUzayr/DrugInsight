import pandas as pd

external_ids = pd.read_csv('data/processed/drugbank_external_ids.csv')

# Filter only RxCUI rows
rxnorm_map = external_ids[external_ids['resource'] == 'RxCUI'][
    ['drugbank_id', 'drug_name', 'identifier']
].rename(columns={'identifier': 'rxnorm_id'})

# Make sure types match for joining (both should be strings or both int)
rxnorm_map['rxnorm_id'] = rxnorm_map['rxnorm_id'].astype(str)

print(f"Drugs with RxCUI mapping: {len(rxnorm_map)}")
print(rxnorm_map.head())

rxnorm_map.to_csv('data/processed/rxnorm_bridge.csv', index=False)