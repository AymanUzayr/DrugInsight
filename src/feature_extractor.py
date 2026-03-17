import pandas as pd
import os


class FeatureExtractor:
    """
    Pulls structured pharmacological context for a drug pair from DrugBank CSVs.
    """

    def __init__(self, data_dir='data/processed'):
        print("Loading pharmacological databases...")

        # Drug name → DrugBank ID mapping
        drugs_df = pd.read_csv(os.path.join(data_dir, 'drugbank_drugs.csv'))
        name_col = 'name' if 'name' in drugs_df.columns else drugs_df.columns[1]
        id_col   = 'drugbank_id' if 'drugbank_id' in drugs_df.columns else drugs_df.columns[0]

        # 1. Build exact name lookup
        self.name_to_id = {}
        for _, row in drugs_df.iterrows():
            self.name_to_id[str(row[name_col]).lower().strip()] = row[id_col]

        # 2. Add synonyms
        for _, row in drugs_df.iterrows():
            synonyms = str(row.get('synonyms', ''))
            if synonyms and synonyms != 'nan':
                for syn in synonyms.split('|'):
                    syn = syn.strip().lower()
                    if syn and syn not in self.name_to_id:
                        self.name_to_id[syn] = row[id_col]

        # 3. Add common brand/alias names — MUST be after name_to_id is built
        COMMON_ALIASES = {
            'aspirin':    'acetylsalicylic acid',
            'tylenol':    'acetaminophen',
            'advil':      'ibuprofen',
            'motrin':     'ibuprofen',
            'glucophage': 'metformin',
            'zocor':      'simvastatin',
            'lipitor':    'atorvastatin',
            'coumadin':   'warfarin',
            'prozac':     'fluoxetine',
            'zoloft':     'sertraline',
            'prinivil':   'lisinopril',
            'norvasc':    'amlodipine',
        }
        for alias, canonical in COMMON_ALIASES.items():
            if alias not in self.name_to_id and canonical in self.name_to_id:
                self.name_to_id[alias] = self.name_to_id[canonical]

        self.id_to_name = dict(zip(drugs_df[id_col], drugs_df[name_col]))

        # Enzyme data
        enzymes_df        = pd.read_csv(os.path.join(data_dir, 'drugbank_enzymes.csv'))
        self.drug_enzymes = (
            enzymes_df.groupby('drugbank_id')
            .apply(lambda x: x[['enzyme_id', 'enzyme_name', 'gene_name', 'actions']]
                   .to_dict('records'), include_groups=False)
            .to_dict()
        )

        # Target data
        targets_df        = pd.read_csv(os.path.join(data_dir, 'drugbank_targets.csv'))
        self.drug_targets = (
            targets_df.groupby('drugbank_id')
            .apply(lambda x: x[['target_id', 'target_name', 'gene_name', 'actions', 'known_action']]
                   .to_dict('records'), include_groups=False)
            .to_dict()
        )

        # Carrier data
        carriers_df = pd.read_csv(os.path.join(data_dir, 'drugbank_carriers.csv'))
        if 'drugbank_id' in carriers_df.columns:
            self.drug_carriers = (
                carriers_df.groupby('drugbank_id')
                .apply(lambda x: x.to_dict('records'), include_groups=False)
                .to_dict()
            )
        else:
            self.drug_carriers = {}

        # Transporter data
        transporters_df = pd.read_csv(os.path.join(data_dir, 'drugbank_transporters.csv'))
        if 'drugbank_id' in transporters_df.columns:
            self.drug_transporters = (
                transporters_df.groupby('drugbank_id')
                .apply(lambda x: x.to_dict('records'), include_groups=False)
                .to_dict()
            )
        else:
            self.drug_transporters = {}

        # Pathway data
        pathways_df = pd.read_csv(os.path.join(data_dir, 'drugbank_pathways.csv'))
        if 'drugbank_id' in pathways_df.columns and 'pathway_name' in pathways_df.columns:
            self.drug_pathways = (
                pathways_df.groupby('drugbank_id')['pathway_name']
                .apply(list)
                .to_dict()
            )
        else:
            self.drug_pathways = {}

        # Known interactions
        self.known_interactions = pd.read_csv(
            os.path.join(data_dir, 'drugbank_interactions_enriched.csv')
        )

        print("Feature extractor ready.")

    def resolve_drug(self, drug_input):
        drug_input = str(drug_input).strip()

        # 1. Exact DrugBank ID match
        upper = drug_input.upper()
        if upper in self.id_to_name:
            return upper, self.id_to_name[upper]

        # 2. Exact name match (case-insensitive)
        lower = drug_input.lower()
        if lower in self.name_to_id:
            db_id = self.name_to_id[lower]
            return db_id, self.id_to_name.get(db_id, drug_input)

        # 3. Starts-with match (shortest wins)
        if len(lower) >= 4:
            matches = [
                (name, did) for name, did in self.name_to_id.items()
                if name.startswith(lower)
            ]
            if matches:
                matches.sort(key=lambda x: len(x[0]))
                name, db_id = matches[0]
                return db_id, self.id_to_name.get(db_id, name)

            # 4. Contains match (last resort)
            matches = [
                (name, did) for name, did in self.name_to_id.items()
                if lower in name
            ]
            if matches:
                matches.sort(key=lambda x: len(x[0]))
                name, db_id = matches[0]
                return db_id, self.id_to_name.get(db_id, name)

        raise ValueError(f"Drug '{drug_input}' not found in DrugBank database.")

    def get_shared_enzymes(self, id_a, id_b):
        enzymes_a = {e['enzyme_id']: e for e in self.drug_enzymes.get(id_a, [])}
        enzymes_b = {e['enzyme_id']: e for e in self.drug_enzymes.get(id_b, [])}
        shared_ids = set(enzymes_a.keys()) & set(enzymes_b.keys())
        results = [enzymes_a[eid] for eid in shared_ids]
        for e in results:
            if str(e.get('gene_name', '')) == 'nan':
                e['gene_name'] = e.get('enzyme_name', 'Unknown').split(' ')[-1]
        for e in results:
            gene = e.get('gene_name', '')
            if gene and not gene.startswith('CYP'):
                e['gene_name'] = 'CYP' + gene
        return results

    def get_shared_targets(self, id_a, id_b):
        targets_a = {t['target_id']: t for t in self.drug_targets.get(id_a, [])}
        targets_b = {t['target_id']: t for t in self.drug_targets.get(id_b, [])}
        shared_ids = set(targets_a.keys()) & set(targets_b.keys())
        return [targets_a[tid] for tid in shared_ids]

    def get_shared_pathways(self, id_a, id_b):
        pathways_a = set(self.drug_pathways.get(id_a, []))
        pathways_b = set(self.drug_pathways.get(id_b, []))
        return list(pathways_a & pathways_b)

    def get_known_interaction(self, id_a, id_b):
        mask = (
            (self.known_interactions['drug_1_id'] == id_a) &
            (self.known_interactions['drug_2_id'] == id_b)
        ) | (
            (self.known_interactions['drug_1_id'] == id_b) &
            (self.known_interactions['drug_2_id'] == id_a)
        )
        matches = self.known_interactions[mask]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None

    def extract(self, drug_a, drug_b):
        id_a, name_a = self.resolve_drug(drug_a)
        id_b, name_b = self.resolve_drug(drug_b)

        shared_enzymes    = self.get_shared_enzymes(id_a, id_b)
        shared_targets    = self.get_shared_targets(id_a, id_b)
        shared_pathways   = self.get_shared_pathways(id_a, id_b)
        known_interaction = self.get_known_interaction(id_a, id_b)

        return {
            'drug_a': {'id': id_a, 'name': name_a},
            'drug_b': {'id': id_b, 'name': name_b},
            'shared_enzymes':    shared_enzymes,
            'shared_targets':    shared_targets,
            'shared_pathways':   shared_pathways,
            'enzymes_a':         self.drug_enzymes.get(id_a, []),
            'enzymes_b':         self.drug_enzymes.get(id_b, []),
            'targets_a':         self.drug_targets.get(id_a, []),
            'targets_b':         self.drug_targets.get(id_b, []),
            'known_interaction': known_interaction,
            'shared_enzyme_count': len(shared_enzymes),
            'shared_target_count': len(shared_targets),
            'max_PRR':         known_interaction.get('max_PRR', 0.0) if known_interaction else 0.0,
            'twosides_found':  1 if (known_interaction and known_interaction.get('twosides_found', 0)) else 0,
        }


# ── Test ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fe = FeatureExtractor()

    for pair in [('Warfarin', 'Aspirin'), ('Warfarin', 'Fluconazole')]:
        print(f"\n{'─'*50}")
        ctx = fe.extract(*pair)
        print(f"Drug A: {ctx['drug_a']}")
        print(f"Drug B: {ctx['drug_b']}")
        print(f"Shared enzymes ({ctx['shared_enzyme_count']}):")
        for e in ctx['shared_enzymes']:
            print(f"  {e.get('gene_name')} — {e.get('enzyme_name')}")
        print(f"Shared targets ({ctx['shared_target_count']}):")
        for t in ctx['shared_targets']:
            print(f"  {t.get('gene_name')} — {t.get('target_name')}")
        print(f"Known interaction: {ctx['known_interaction'] is not None}")
