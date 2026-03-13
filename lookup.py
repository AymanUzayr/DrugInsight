import pandas as pd

# ── Load everything once at startup ────────────────────────
bridge        = pd.read_csv('data/processed/rxnorm_bridge.csv', dtype={'rxnorm_id': str})
interactions  = pd.read_csv('data/processed/drugbank_interactions_filtered.csv')
twosides      = pd.read_csv('data/processed/twosides_processed_filtered.csv', dtype={'drug_1_rxnorn_id': str, 'drug_2_rxnorm_id': str})
smiles        = pd.read_csv('data/processed/drugbank_smiles_filtered.csv')
enzymes       = pd.read_csv('data/processed/drugbank_enzymes.csv')
targets       = pd.read_csv('data/processed/drugbank_targets.csv')
transporters  = pd.read_csv('data/processed/drugbank_transporters.csv')

# ── Build lookup dictionaries for O(1) access ──────────────
bridge_by_name    = bridge.set_index('drug_name')['drugbank_id'].to_dict()
bridge_by_rxnorm  = bridge.set_index('rxnorm_id')['drugbank_id'].to_dict()
bridge_to_rxnorm  = bridge.set_index('drugbank_id')['rxnorm_id'].to_dict()
smiles_dict       = smiles.set_index('drugbank_id')['smiles'].to_dict()

# ── 1. Name Normalizer ─────────────────────────────────────
def resolve_drug(name):
    """
    Given a free text drug name, return drugbank_id and rxnorm_id.
    Returns None if not found.
    """
    # Try exact match first
    name_clean = name.strip().lower()
    for key, db_id in bridge_by_name.items():
        if key.lower() == name_clean:
            rxnorm_id = bridge_to_rxnorm.get(db_id)
            return {'drugbank_id': db_id, 'rxnorm_id': rxnorm_id, 'name': key}

    # Try fuzzy match if no exact match
    from difflib import get_close_matches
    close = get_close_matches(name_clean, [k.lower() for k in bridge_by_name.keys()], n=1, cutoff=0.85)
    if close:
        matched_name = next(k for k in bridge_by_name.keys() if k.lower() == close[0])
        db_id = bridge_by_name[matched_name]
        rxnorm_id = bridge_to_rxnorm.get(db_id)
        return {'drugbank_id': db_id, 'rxnorm_id': rxnorm_id, 'name': matched_name}

    return None


# ── 2. DrugBank Evidence Lookup ────────────────────────────
def query_drugbank(db_id_1, db_id_2):
    """
    Given two DrugBank IDs, return mechanism, shared enzymes,
    shared targets, shared transporters.
    """
    result = {}

    # Direct interaction record
    match = interactions[
        ((interactions['drug_1_id'] == db_id_1) & (interactions['drug_2_id'] == db_id_2)) |
        ((interactions['drug_1_id'] == db_id_2) & (interactions['drug_2_id'] == db_id_1))
    ]
    result['mechanism']        = match['mechanism'].values[0] if len(match) else None
    result['interaction_found'] = len(match) > 0

    # Shared enzymes
    enz_1 = set(enzymes[enzymes['drugbank_id'] == db_id_1]['enzyme_name'].dropna())
    enz_2 = set(enzymes[enzymes['drugbank_id'] == db_id_2]['enzyme_name'].dropna())
    result['shared_enzymes']   = list(enz_1 & enz_2)

    # Shared targets
    tgt_1 = set(targets[targets['drugbank_id'] == db_id_1]['target_name'].dropna())
    tgt_2 = set(targets[targets['drugbank_id'] == db_id_2]['target_name'].dropna())
    result['shared_targets']   = list(tgt_1 & tgt_2)

    # Shared transporters
    trp_1 = set(transporters[transporters['drugbank_id'] == db_id_1]['transporter_name'].dropna())
    trp_2 = set(transporters[transporters['drugbank_id'] == db_id_2]['transporter_name'].dropna())
    result['shared_transporters'] = list(trp_1 & trp_2)

    return result


# ── 3. TWOSIDES Evidence Lookup ────────────────────────────
def query_twosides(rxnorm_1, rxnorm_2):
    """
    Given two RxNorm IDs, return PRR signals and top condition.
    Checks both orderings.
    """
    rxnorm_1, rxnorm_2 = str(rxnorm_1), str(rxnorm_2)

    match = twosides[
        ((twosides['drug_1_rxnorn_id'] == rxnorm_1) & (twosides['drug_2_rxnorm_id'] == rxnorm_2)) |
        ((twosides['drug_1_rxnorn_id'] == rxnorm_2) & (twosides['drug_2_rxnorm_id'] == rxnorm_1))
    ]

    if len(match) == 0:
        return {'signal_found': False}

    row = match.iloc[0]
    return {
        'signal_found'     : True,
        'max_PRR'          : row['max_PRR'],
        'mean_PRR'         : row['mean_PRR'],
        'num_signals'      : row['num_signals'],
        'total_coreports'  : row['total_coreports'],
        'mean_report_freq' : row['mean_report_freq'],
        'top_condition'    : row['top_condition'],
    }


# ── 4. SMILES Lookup ───────────────────────────────────────
def get_smiles(db_id):
    return smiles_dict.get(db_id, None)


# ── 5. Master Lookup — combines everything ─────────────────
def lookup_drug_pair(drug_name_1, drug_name_2):
    """
    Given two drug names, return all evidence from both sources.
    This is the single entry point for Stage 2.
    """
    drug_1 = resolve_drug(drug_name_1)
    drug_2 = resolve_drug(drug_name_2)

    if not drug_1:
        return {'error': f'Drug not found: {drug_name_1}'}
    if not drug_2:
        return {'error': f'Drug not found: {drug_name_2}'}

    db_evidence = query_drugbank(drug_1['drugbank_id'], drug_2['drugbank_id'])
    tw_evidence = query_twosides(drug_1['rxnorm_id'],   drug_2['rxnorm_id'])
    smiles_1    = get_smiles(drug_1['drugbank_id'])
    smiles_2    = get_smiles(drug_2['drugbank_id'])

    return {
        'drug_1'        : drug_1,
        'drug_2'        : drug_2,
        'smiles_1'      : smiles_1,
        'smiles_2'      : smiles_2,
        'drugbank'      : db_evidence,
        'twosides'      : tw_evidence,
    }


# ── Test it ────────────────────────────────────────────────
if __name__ == '__main__':
    result = lookup_drug_pair('Warfarin','Acetylsalicylic acid')
    import json
    print(json.dumps(result, indent=2, default=str))