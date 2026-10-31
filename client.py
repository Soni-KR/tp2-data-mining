"""Chunked HTTP scoring with positional alignment."""
import requests
from fraud_core import FEATURES, MAX_BATCH, records, validate_frame

def score_frame(frame, api_url, progress=None, session=requests):
    validate_frame(frame)
    output = frame.copy().reset_index(drop=True)
    scores, flags = [], []
    for start in range(0, len(frame), MAX_BATCH):
        batch = frame.iloc[start:start + MAX_BATCH][FEATURES]
        response = session.post(api_url + '/predict/batch', json={'transactions': records(batch)}, timeout=120)
        response.raise_for_status()
        predictions = response.json()['predictions']
        if len(predictions) != len(batch) or [p['index'] for p in predictions] != list(range(len(batch))):
            raise ValueError('API returned misaligned predictions.')
        scores.extend(p['risk_score'] for p in predictions)
        flags.extend(p['flagged'] for p in predictions)
        if progress:
            progress(min((start + len(batch)) / len(frame), 1.))
    output['xgb_score'] = scores
    output['flagged'] = flags
    output['row_position'] = range(len(frame))
    return output
