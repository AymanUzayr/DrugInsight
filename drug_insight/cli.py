"""
DrugInsight CLI — terminal interface for drug-drug interaction prediction.

Commands:
    druginsight predict "Warfarin" "Fluconazole"
    druginsight predict "Warfarin" "Fluconazole" --json
    druginsight predict "Warfarin" "Fluconazole" --output report.json
    druginsight info "Warfarin"
    druginsight batch input.csv
    druginsight batch input.csv --output results.csv
"""

import argparse
import json
import sys
import os
import pandas as pd


def cmd_predict(args):
    from drug_insight import DrugInsight
    di     = DrugInsight()
    result = di.predict(args.drug_a, args.drug_b)

    if 'error' in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json or args.output:
        output = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Saved to {args.output}")
        else:
            print(output)
        return

    # Pretty print
    unc = result['uncertainty']
    cs  = result['component_scores']
    ev  = result['evidence']

    print(f"\n{'='*62}")
    print(f"  DRUGINSIGHT -- DDI PREDICTION REPORT")
    print(f"{'='*62}")
    print(f"  Drug A : {result['drug_a']:30s} ({result['drugbank_id_a']})")
    print(f"  Drug B : {result['drug_b']:30s} ({result['drugbank_id_b']})")
    print(f"{'─'*62}")
    print(f"  Interaction : {'YES' if result['interaction'] else 'NO'}")
    print(f"  Severity    : {result['severity']}")
    print(f"  Risk Index  : {result['risk_index']} / 100")
    print(f"  Confidence  : {result['confidence']}")
    print(f"{'─'*62}")
    print(f"  Summary:\n    {result['summary']}")
    print(f"\n  Mechanism:\n    {result['mechanism']}")
    print(f"\n  Recommendation:\n    {result['recommendation']}")
    print(f"{'─'*62}")
    print(f"  Evidence Sources:")
    print(f"    DrugBank [{unc['drugbank_confidence']:10s}]  "
          f"enzymes={ev['drugbank']['shared_enzymes'] or 'none'}  "
          f"targets={ev['drugbank']['shared_targets'] or 'none'}")
    print(f"    TWOSIDES [{unc['twosides_confidence']:10s}]  PRR={ev['twosides']['max_PRR']:.1f}"
          + ("  (confounding possible)" if ev['twosides']['confounding_flag'] else ""))
    print(f"    ML Model [{unc['ml_confidence']:10s}]  raw_prob={cs['ml_score']:.3f}")
    print(f"{'─'*62}")
    print(f"  Component Scores:")
    print(f"    Rule score     : {cs['rule_score']:.3f}  (w={cs['weights']['rule']})")
    print(f"    ML score       : {cs['ml_score']:.3f}  (w={cs['weights']['ml']})")
    print(f"    Twosides score : {cs['twosides_score']:.3f}  (w={cs['weights']['twosides']})")
    print(f"  Overall confidence : {unc['overall_confidence']}")
    print(f"{'='*62}\n")


def cmd_info(args):
    from drug_insight import DrugInsight
    di     = DrugInsight()
    result = di.resolve_drug(args.drug)

    if 'error' in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  {result['name']} ({result['drugbank_id']})")
    print(f"{'='*50}")
    print(f"  Has SMILES: {'Yes' if result['has_smiles'] else 'No'}")
    print(f"  Enzymes ({len(result['enzymes'])}):")
    for e in result['enzymes'][:5]:
        print(f"    {e.get('gene_name', '?')} — {e.get('enzyme_name', '?')}")
    print(f"  Targets ({len(result['targets'])}):")
    for t in result['targets'][:5]:
        print(f"    {t.get('gene_name', '?')} — {t.get('target_name', '?')}")
    print(f"  Pathways ({len(result['pathways'])}):")
    for p in result['pathways'][:5]:
        print(f"    {p}")
    print(f"{'='*50}\n")


def cmd_batch(args):
    from drug_insight import DrugInsight

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if 'drug_a' not in df.columns or 'drug_b' not in df.columns:
        print("Error: CSV must have columns 'drug_a' and 'drug_b'", file=sys.stderr)
        sys.exit(1)

    di      = DrugInsight()
    results = []

    for i, row in df.iterrows():
        print(f"Processing {i+1}/{len(df)}: {row['drug_a']} + {row['drug_b']}")
        result = di.predict(row['drug_a'], row['drug_b'])
        results.append({
            'drug_a':      row['drug_a'],
            'drug_b':      row['drug_b'],
            'interaction': result.get('interaction', None),
            'severity':    result.get('severity', None),
            'risk_index':  result.get('risk_index', None),
            'confidence':  result.get('confidence', None),
            'error':       result.get('error', None),
        })

    out_df = pd.DataFrame(results)

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(results)} predictions to {args.output}")
    else:
        print(out_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        prog='druginsight',
        description='DrugInsight — Explainable Drug-Drug Interaction Prediction'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # predict command
    p_predict = subparsers.add_parser('predict', help='Predict DDI between two drugs')
    p_predict.add_argument('drug_a', type=str, help='First drug name or DrugBank ID')
    p_predict.add_argument('drug_b', type=str, help='Second drug name or DrugBank ID')
    p_predict.add_argument('--json',   action='store_true', help='Output as JSON')
    p_predict.add_argument('--output', type=str, default=None, help='Save output to file')

    # info command
    p_info = subparsers.add_parser('info', help='Show drug profile')
    p_info.add_argument('drug', type=str, help='Drug name or DrugBank ID')

    # batch command
    p_batch = subparsers.add_parser('batch', help='Predict from CSV file')
    p_batch.add_argument('input',    type=str, help='Input CSV with drug_a and drug_b columns')
    p_batch.add_argument('--output', type=str, default=None, help='Save results to CSV')

    args = parser.parse_args()

    if args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'batch':
        cmd_batch(args)


if __name__ == '__main__':
    main()
