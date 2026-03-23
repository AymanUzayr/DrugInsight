"""
DrugInsight REST API
Run with: uvicorn src.api:app --reload

Endpoints:
    POST /predict
    POST /predict/batch
    GET  /health
    GET  /drugs/{name}
    GET  /drugs/{name}/interactions
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI(
    title="DrugInsight API",
    description="Explainable drug-drug interaction prediction",
    version="0.1.0",
)

# CORS — allow all origins for research use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ─────────────────────────────────────────────────
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    from drug_insight import DrugInsight
    predictor = DrugInsight()
    print("DrugInsight model loaded.")


# ── Request / Response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    drug_a: str
    drug_b: str

class BatchPredictRequest(BaseModel):
    pairs: List[PredictRequest]

class PredictResponse(BaseModel):
    drug_a:          str
    drug_b:          str
    drugbank_id_a:   str
    drugbank_id_b:   str
    interaction:     bool
    probability:     float
    risk_index:      int
    severity:        str
    confidence:      str
    summary:         str
    mechanism:       str
    recommendation:  str
    evidence:        dict
    component_scores: dict
    uncertainty:     dict
    full_explanation: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — confirms model is loaded."""
    return {
        "status":  "ok",
        "model":   "loaded" if predictor else "not loaded",
        "version": "0.1.0",
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict drug-drug interaction between two drugs.

    Body: { "drug_a": "Warfarin", "drug_b": "Fluconazole" }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predictor.predict(req.drug_a, req.drug_b)

    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])

    return result


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    """
    Predict DDI for multiple drug pairs.

    Body: { "pairs": [{"drug_a": "Warfarin", "drug_b": "Aspirin"}, ...] }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(req.pairs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 pairs per batch request")

    results = []
    for pair in req.pairs:
        result = predictor.predict(pair.drug_a, pair.drug_b)
        results.append(result)

    return {"results": results, "count": len(results)}


@app.get("/drugs/{name}")
def get_drug(name: str):
    """
    Resolve drug name or DrugBank ID to full drug profile.

    Returns enzymes, targets, pathways, and whether SMILES is available.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predictor.resolve_drug(name)

    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])

    return result


@app.get("/drugs/{name}/interactions")
def get_drug_interactions(name: str, limit: int = 20):
    """
    List known interactions for a drug from DrugBank.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        fe = predictor._predictor.feature_extractor
        db_id, canonical_name = fe.resolve_drug(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    ki = fe.known_interactions
    mask = (ki['drug_1_id'] == db_id) | (ki['drug_2_id'] == db_id)
    matches = ki[mask].head(limit)

    interactions = []
    for _, row in matches.iterrows():
        other_id = row['drug_2_id'] if row['drug_1_id'] == db_id else row['drug_1_id']
        interactions.append({
            'drug_id':               other_id,
            'drug_name':             fe.id_to_name.get(other_id, other_id),
            'shared_enzyme_count':   row.get('shared_enzyme_count', 0),
            'shared_target_count':   row.get('shared_target_count', 0),
            'max_PRR':               row.get('max_PRR', 0.0),
            'twosides_found':        bool(row.get('twosides_found', 0)),
        })

    return {
        'drug_id':      db_id,
        'drug_name':    canonical_name,
        'interactions': interactions,
        'count':        len(interactions),
    }


@app.get("/drugs")
def list_drugs(search: Optional[str] = None, limit: int = 50):
    """
    List all known drugs, optionally filtered by name search.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    names = predictor.drug_names()

    if search:
        names = [n for n in names if search.lower() in n.lower()]

    return {
        "drugs": names[:limit],
        "count": len(names[:limit]),
        "total": len(names),
    }
