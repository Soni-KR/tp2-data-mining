"""Shared validation and deterministic decision policies."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, fbeta_score, matthews_corrcoef, precision_score, recall_score
BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / 'fraud_artifacts' if (BASE / 'fraud_artifacts/config.json').exists() else BASE
CONFIG = json.loads((ARTIFACTS / 'config.json').read_text())
FEATURES = CONFIG['features']
MAX_BATCH = 1000

def validate_frame(frame):
    if frame.empty:
        raise ValueError('No transactions supplied.')
    if not frame.columns.is_unique:
        raise ValueError('Duplicate column names are not allowed.')
    missing = [f for f in FEATURES if f not in frame]
    if missing:
        raise ValueError('Missing required features: ' + ', '.join(missing))
    result = frame[FEATURES].copy()
    result = result.where(result.notna(), np.nan)
    for name in CONFIG['numerical']:
        try:
            values = pd.to_numeric(result[name], errors='raise')
        except (ValueError, TypeError):
            raise ValueError(f'{name}: expected numeric values or empty cells.') from None
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f'{name}: infinite values are not allowed.')
        result[name] = values.astype(float)
    for name in CONFIG['categorical']:
        for value in result[name].dropna():
            if isinstance(value, (dict, list, bool)) or not isinstance(value, (str, int, float, np.number)):
                raise ValueError(f'{name}: expected a scalar category or null.')
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f'{name}: nonfinite category.')
    if 'isFraud' in frame and not frame['isFraud'].isin([0, 1]).all():
        raise ValueError('isFraud must contain only 0 or 1; remove the column for unlabeled data.')
    return result

def decisions(scores, mode='threshold', threshold=.39, budget=10000):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite vector.')
    if mode == 'threshold':
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between 0 and 1.')
        return scores >= threshold
    if mode != 'budget' or budget < 0 or int(budget) != budget:
        raise ValueError('Use threshold or a nonnegative integer budget.')
    result = np.zeros(len(scores), dtype=bool)
    result[np.argsort(-scores, kind='stable')[:min(int(budget), len(scores))]] = True
    return result

def metrics(frame, flagged, investigation=5):
    if not np.isfinite(investigation) or investigation < 0:
        raise ValueError('Investigation cost must be finite and nonnegative.')
    flagged = np.asarray(flagged, dtype=bool)
    if len(frame) != len(flagged):
        raise ValueError('Row alignment mismatch.')
    alerts = int(flagged.sum())
    out = dict(transactions=len(frame), alerts=alerts, alert_rate=alerts / len(frame) if len(frame) else 0,
               investigation_cost=float(investigation * alerts))
    if 'isFraud' not in frame or not frame.isFraud.isin([0, 1]).all():
        return out
    y = frame.isFraud.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    out.update(precision=float(precision_score(y, flagged, zero_division=0)),
               recall=float(recall_score(y, flagged, zero_division=0)),
               f2=float(fbeta_score(y, flagged, beta=2, zero_division=0)),
               mcc=float(matthews_corrcoef(y, flagged)) if len(np.unique(y)) > 1 and len(np.unique(flagged)) > 1 else 0.,
               tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    amounts = pd.to_numeric(frame.TransactionAmt, errors='coerce').to_numpy(dtype=float)
    if np.isfinite(amounts).all() and (amounts >= 0).all():
        loss = float(amounts[(y == 1) & ~flagged].sum())
        baseline = float(amounts[y == 1].sum())
        total = loss + out['investigation_cost']
        out.update(missed_fraud_loss=loss, total_cost=total, no_model_cost=baseline,
                   cost_reduction=1-total/baseline if baseline else None)
    return out

def records(frame):
    return json.loads(frame.to_json(orient='records'))
