"""Inference over trusted local artifacts; never load user-supplied models."""
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import DMatrix, XGBClassifier
from fraud_core import ARTIFACTS, CONFIG, FEATURES, MAX_BATCH, records, validate_frame
app = FastAPI(title='Explainable Fraud Detection API', version='2.0.0', description='Feature-ready IEEE-CIS inference. Risk scores are not calibrated probabilities. All 68 columns required; null cells use the fitted imputer.')
preprocessor = joblib.load(ARTIFACTS / 'preprocessor.pkl')
model = XGBClassifier()
model.load_model(ARTIFACTS / 'xgboost_model.json')
reference = pd.read_csv(ARTIFACTS / 'reference.csv')
THRESHOLD = float(CONFIG['cost_threshold'])

class TransactionRequest(BaseModel):
    features: dict = Field(description='Required keys from config.json; extra metadata ignored.')
class BatchTransactionRequest(BaseModel):
    transactions: list[dict] = Field(min_length=1, max_length=MAX_BATCH)

def transform(items):
    for index, item in enumerate(items):
        missing = [f for f in FEATURES if f not in item]
        if missing:
            raise HTTPException(422, detail={'row': index, 'missing_features': missing})
    try:
        frame = validate_frame(pd.DataFrame(items))
        encoder = preprocessor.named_transformers_['cat']
        for name, categories in zip(CONFIG['categorical'], encoder.categories_):
            if categories.dtype.kind in 'iuf':
                frame[name] = pd.to_numeric(frame[name], errors='raise').astype(float)
            else:
                frame[name] = frame[name].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        return preprocessor.transform(frame)
    except (ValueError, TypeError, OverflowError):
        try:
            validate_frame(pd.DataFrame(items))
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from None
        raise HTTPException(422, detail='Invalid categorical values; consult the example payload.') from None

def result(score):
    return dict(risk_score=float(score), threshold=THRESHOLD, flagged=bool(score >= THRESHOLD), decision='REVIEW' if score >= THRESHOLD else 'PASS')

@app.exception_handler(Exception)
async def unexpected(request, exc):
    from fastapi.responses import JSONResponse
    logging.exception('Inference failure', exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': 'Inference failed. Check server logs.'})

@app.get('/')
def root():
    return {'message': 'Explainable Fraud Detection API', 'documentation': '/docs'}
@app.get('/health')
def health():
    return dict(status='ok', model='XGBoost', features=len(FEATURES), threshold=THRESHOLD, max_batch=MAX_BATCH)
@app.get('/example-payload')
def example_payload():
    return {'features': records(reference[FEATURES].iloc[:1])[0]}
@app.get('/predict/example')
def predict_example():
    return predict(TransactionRequest(**example_payload()))
@app.post('/predict')
def predict(request: TransactionRequest):
    return result(model.predict_proba(transform([request.features]))[0, 1])
@app.post('/predict/batch')
def predict_batch(request: BatchTransactionRequest):
    scores = model.predict_proba(transform(request.transactions))[:, 1]
    return dict(total=len(scores), alerts=int((scores >= THRESHOLD).sum()), predictions=[dict(index=i, **result(s)) for i, s in enumerate(scores)])
@app.post('/explain')
def explain_transaction(request: TransactionRequest):
    X = transform([request.features])
    score = model.predict_proba(X)[0, 1]
    matrix = DMatrix(X)
    booster = model.get_booster()
    iterations = (0, int(CONFIG['best_iteration']) + 1)
    values = booster.predict(matrix, pred_contribs=True, iteration_range=iterations)[0]
    margin = float(booster.predict(matrix, output_margin=True, iteration_range=iterations)[0])
    contributions = [dict(feature=str(n), shap_value=float(v), direction='INCREASES_RISK' if v > 0 else 'DECREASES_RISK') for n, v in zip(preprocessor.get_feature_names_out(), values[:-1])]
    strongest = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
    return dict(**result(score), reason_codes=strongest[:5], contributions=contributions, baseline_margin=float(values[-1]), raw_margin=margin, reconstructed_margin=float(np.sum(values, dtype=np.float64)), reconstructed_score=float(1 / (1 + np.exp(-margin))), shap_units='log-odds')
