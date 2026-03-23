import os
import pandas as pd
import numpy as np

def analyze_csvs():
    data_dir = os.path.join("data", "processed")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    drug_ids_with_features = set()
    feature_files = [
        "drugbank_enzymes.csv",
        "drugbank_targets.csv",
        "drugbank_transporters.csv",
        "drugbank_carriers.csv"
    ]
    
    for f in csv_files:
        path = os.path.join(data_dir, f)
        print(f"\n" + "="*50)
        print(f"Analyzing {f}")
        print("="*50)
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"  Shape: {df.shape}")
            
            null_sum = df.isnull().sum()
            print(f"  Null counts:")
            if (null_sum > 0).any():
                for col, count in null_sum[null_sum > 0].items():
                    print(f"    {col}: {count} nulls")
            else:
                print("    No null columns.")
                
            print(f"  Unique values (all cols):")
            unique_counts = df.nunique()
            for col, count in unique_counts.items():
                print(f"    {col}: {count}")

            if f == "drugbank_interactions_enriched.csv":
                print("\n  --- Feature distributions ---")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                     print(df[numeric_cols].describe().to_string())
                if "label" in df.columns:
                     print("\n  --- Class balance (label) ---")
                     print(df["label"].value_counts().to_string())
                elif "positive_label" in df.columns or "is_interaction" in df.columns:
                     print("\n  --- Class balance (other known labels) ---")
                     for c in ["positive_label", "is_interaction", "interaction"]:
                         if c in df.columns:
                             print(f"    {c}: \n{df[c].value_counts().to_string()}")
            
            if f == "drugbank_smiles_filtered.csv" or f == "drugbank_smiles.csv":
                smiles_col = [c for c in df.columns if 'smiles' in c.lower()]
                if smiles_col:
                     scol = smiles_col[0]
                     valid_smiles = df[scol].notna().sum()
                     total = len(df)
                     print(f"\n  --- SMILES validity rate ---")
                     print(f"  {valid_smiles}/{total} ({valid_smiles/total:.2%} valid)")
            
            if f in feature_files:
                id_col = [c for c in df.columns if 'drugbank_id' in c.lower() or 'drug_id' in c.lower() or 'parent_key' in c.lower() or 'id' in c.lower() and not 'uniprot' in c.lower() and not 'pubchem' in c.lower()]
                if id_col:
                    unique_drugs = df[id_col[0]].dropna().unique()
                    drug_ids_with_features.update(unique_drugs)

            # Check duplicate drug pairs
            if 'interactions' in f and 'drug_a' in df.columns and 'drug_b' in df.columns:
                print("\n  --- Duplicates Check ---")
                pairs = df.apply(lambda row: tuple(sorted([str(row['drug_a']), str(row['drug_b'])])), axis=1)
                num_duplicates = len(pairs) - len(set(pairs))
                print(f"  Duplicate drug pairs: {num_duplicates}")

        except Exception as e:
            print(f"  Error reading {f}: {e}")

    print("\n" + "="*50)
    print("Aggregate Statistics")
    print("="*50)
    print(f"Total unique drugs across enzyme/target/transporter/carrier data: {len(drug_ids_with_features)}")

if __name__ == "__main__":
    analyze_csvs()

# ======================================================================
# DATA QUALITY ISSUES IDENTIFIED (Stage 0):
# 1. Duplicate drug pairs in interactions: Since there are ~1.87 million rows but only ~3100 unique drugs, some unique pairs have duplicate interaction records (especially since TWOSIDES might have introduced duplicates or swapped order).
# 2. Drugs with missing features: Out of 19,830 drugs, only 10,280 have any target/enzyme/carrier/transporter data. This means ~9,550 drugs have NO feature data at all.
# 3. Suspiciously high shared feature counts: Max shared_target_count is 36, and max shared_enzyme_count is 21. This indicates possible data errors or broad-acting small molecules where feature overlap is not meaningful.
# 4. PRR Distribution: max_PRR has an extreme outlier at 960.0, while the 75th percentile is 0.0. This indicates a heavily skewed distribution that needs capping.
# ======================================================================
