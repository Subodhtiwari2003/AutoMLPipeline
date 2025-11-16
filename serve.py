from fastapi import FastAPI
import joblib, pandas as pd, uvicorn
app = FastAPI()
MODEL_PATH = 'best_pipeline.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print('Model loaded.')
except Exception as e:
    model = None
    print('Could not load model at start:', e)

@app.post('/predict')
def predict(payload: dict):
    if model is None:
        return {'error': 'Model not loaded'}
    df = pd.DataFrame(payload.get('rows', []))
    preds = model.predict(df)
    proba = model.predict_proba(df).tolist() if hasattr(model, 'predict_proba') else None
    return {'predictions': preds.tolist(), 'probabilities': proba}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
