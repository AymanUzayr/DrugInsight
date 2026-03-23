"""
DrugInsight — Explainable Drug-Drug Interaction Prediction Framework

Usage:
    from drug_insight import DrugInsight

    di = DrugInsight()
    result = di.predict('Warfarin', 'Fluconazole')
    print(result['severity'])
    print(result['mechanism'])
"""

from drug_insight.predictor import DrugInsight

__version__ = '0.1.0'
__all__     = ['DrugInsight']
