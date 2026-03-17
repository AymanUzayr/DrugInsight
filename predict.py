from matplotlib.style import context
import torch
import pandas as pd
import json
import argparse
import os
from rdkit import Chem, RDLogger

from mol_graph        import smiles_to_graph
from gnn_encoder      import GNNEncoder
from ddi_classifier   import DDIClassifier
from feature_extractor import FeatureExtractor
from explainer        import Explainer

RDLogger.DisableLog('rdApp.*')

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models/ddi_model.pt'
DATA_DIR   = 'data/processed'

SEVERITY_THRESHOLD = 0.5   # probability above which interaction is flagged


class DDIPredictor:
    """
    End-to-end DDI prediction pipeline.
    Input:  two drug names (or DrugBank IDs)
    Output: interaction prediction + severity + mechanism explanation
    """

    def __init__(self, model_path=MODEL_PATH, data_dir=DATA_DIR):
        print("Initialising DDI prediction pipeline...")

        # Load SMILES lookup
        smiles_df   = pd.read_csv(os.path.join(data_dir, 'drugbank_smiles_filtered.csv'))
        self.smiles_dict = dict(zip(smiles_df['drugbank_id'], smiles_df['smiles']))

        # Load model
        self.gnn        = GNNEncoder().to(DEVICE)
        self.classifier = DDIClassifier(extra_features=4).to(DEVICE)

        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.gnn.load_state_dict(checkpoint['gnn'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.gnn.eval()
        self.classifier.eval()
        print(f"Model loaded from {model_path}")

        # Load feature extractor and explainer
        self.feature_extractor = FeatureExtractor(data_dir)
        self.explainer         = Explainer()

        print("Pipeline ready.\n")

    def _get_graph(self, drugbank_id):
        """Convert a drug's SMILES to a PyG graph."""
        smiles = self.smiles_dict.get(drugbank_id)
        if smiles is None:
            raise ValueError(f"No SMILES found for {drugbank_id}")
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Could not parse SMILES for {drugbank_id}")
        return graph

    def predict(self, drug_a, drug_b):
        """
        Predict DDI between two drugs.

        Args:
            drug_a: drug name or DrugBank ID (e.g. 'Warfarin' or 'DB00682')
            drug_b: drug name or DrugBank ID

        Returns:
            dict with full prediction and explanation
        """
        # ── 1. Resolve drug names → IDs + context ─────────────────────────────
        context = self.feature_extractor.extract(drug_a, drug_b)
        # After context = self.feature_extractor.extract(drug_a, drug_b)
        id_a = context['drug_a']['id']
        id_b = context['drug_b']['id']

        if id_a not in self.smiles_dict:
            return {'error': f"No molecular structure (SMILES) available for {context['drug_a']['name']} ({id_a})"}
        if id_b not in self.smiles_dict:
            return {'error': f"No molecular structure (SMILES) available for {context['drug_b']['name']} ({id_b})"}
        name_a  = context['drug_a']['name']
        name_b  = context['drug_b']['name']

        # ── 2. Build molecular graphs ──────────────────────────────────────────
        try:
            graph_a = self._get_graph(id_a)
            graph_b = self._get_graph(id_b)
        except ValueError as e:
            return {'error': str(e)}

        # ── 3. Build extra features (normalised, matching training) ────────────
        extra = torch.tensor([[
            min(float(context['shared_enzyme_count']), 21.0) / 21.0,
            min(float(context['shared_target_count']), 36.0) / 36.0,
            min(float(context.get('max_PRR', 0.0)), 50.0) / 50.0,
            float(context.get('twosides_found', 0)),
        ]], dtype=torch.float).to(DEVICE)

        # ── 4. Run GNN + classifier ────────────────────────────────────────────
        from torch_geometric.data import Batch
        batch_a = Batch.from_data_list([graph_a]).to(DEVICE)
        batch_b = Batch.from_data_list([graph_b]).to(DEVICE)

        with torch.no_grad():
            embed_a = self.gnn(batch_a)
            embed_b = self.gnn(batch_b)
            prob_logit, severity_logits = self.classifier(embed_a, embed_b, extra)

            prob         = torch.sigmoid(prob_logit).item()
            severity_idx = torch.argmax(severity_logits, dim=-1).item()

        # ── 5. Build prediction dict ───────────────────────────────────────────
        interaction = prob >= SEVERITY_THRESHOLD
        prediction  = {
            'interaction':   bool(interaction),
            'probability':   round(prob, 4),
            'severity_idx':  severity_idx,
        }

        # ── 6. Generate explanation ────────────────────────────────────────────
        explanation = self.explainer.explain(context, prediction)

        # ── 7. Assemble final output ───────────────────────────────────────────
        return {
            'drug_a':          name_a,
            'drug_b':          name_b,
            'drugbank_id_a':   id_a,
            'drugbank_id_b':   id_b,
            'interaction':     bool(interaction),
            'probability':     round(prob, 4),
            'severity':        explanation['severity'],
            'confidence':      explanation['confidence'],
            'summary':         explanation['summary'],
            'mechanism':       explanation['mechanism'],
            'recommendation':  explanation['recommendation'],
            'supporting_evidence': explanation['supporting_evidence'],
            'full_explanation': explanation['full_text'],
        }


# ── CLI interface ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Predict drug-drug interaction')
    parser.add_argument('drug_a', type=str, help='First drug name or DrugBank ID')
    parser.add_argument('drug_b', type=str, help='Second drug name or DrugBank ID')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    predictor = DDIPredictor()
    result    = predictor.predict(args.drug_a, args.drug_b)

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{'='*60}")
            print(f"  DDI PREDICTION REPORT")
            print(f"{'='*60}")
            print(f"  Drug A:       {result['drug_a']} ({result['drugbank_id_a']})")
            print(f"  Drug B:       {result['drug_b']} ({result['drugbank_id_b']})")
            print(f"{'─'*60}")
            print(f"  Interaction:  {'YES' if result['interaction'] else 'NO'}")
            print(f"  Severity:     {result['severity']}")
            print(f"  Confidence:   {result['confidence']}")
            print(f"{'─'*60}")
            print(f"  Summary:")
            print(f"    {result['summary']}")
            print(f"\n  Mechanism:")
            print(f"    {result['mechanism']}")
            print(f"\n  Recommendation:")
            print(f"    {result['recommendation']}")
            print(f"{'─'*60}")
            ev = result['supporting_evidence']
            print(f"  Supporting Evidence:")
            print(f"    Shared enzymes:  {ev['shared_enzymes'] or 'None'}")
            print(f"    Shared targets:  {ev['shared_targets'] or 'None'}")
            print(f"    TWOSIDES signal: {'Yes' if ev['twosides_signal'] else 'No'}")
            if ev['max_PRR'] > 0:
                print(f"    Max PRR:         {ev['max_PRR']:.2f}")
            print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
