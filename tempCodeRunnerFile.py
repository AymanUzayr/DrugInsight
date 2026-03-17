from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import pandas as pd

smiles_df = pd.read_csv('data/processed/drugbank_smiles_filtered.csv')

valid = 0
invalid = 0
invalid_examples = []

for _, row in smiles_df.iterrows():
    smi = str(row['smiles']).strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid += 1
        if len(invalid_examples) < 5:
            invalid_examples.append(smi)
    else:
        valid += 1

print(f"Valid:   {valid}")
print(f"Invalid: {invalid}")
print(f"Total:   {valid + invalid}")
print(f"\nInvalid examples:")
for s in invalid_examples:
    print(f"  {s[:80]}")