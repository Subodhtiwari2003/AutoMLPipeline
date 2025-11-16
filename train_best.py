import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def build_final_model_from_params(best_params, preprocessor, X_train, y_train):
    model = None
    model_type = best_params.get('model', 'xgb')
    if model_type == 'xgb':
        params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    else:
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    pipe = Pipeline([('pre', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    return pipe

def save_pipeline(pipe, path='best_pipeline.pkl'):
    joblib.dump(pipe, path)
    print('Saved pipeline to', path)
