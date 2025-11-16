import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import mlflow
from sklearn.pipeline import Pipeline

def objective(trial, X, y, preprocessor):
    model_choice = trial.suggest_categorical('model', ['logreg','rf','xgb','lgb','cat'])
    params = {}
    if model_choice == 'logreg':
        C = trial.suggest_loguniform('logreg_C', 1e-4, 1e2)
        model = LogisticRegression(C=C, max_iter=1000, solver='lbfgs')
        params['model'] = 'logreg'; params['logreg_C'] = C
    elif model_choice == 'rf':
        n_estimators = trial.suggest_int('rf_n', 50, 200)
        max_depth = trial.suggest_int('rf_depth', 3, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        params['model']='rf'; params['rf_n']=n_estimators; params['rf_depth']=max_depth
    elif model_choice == 'xgb':
        n_estimators = trial.suggest_int('xgb_n', 50, 200)
        max_depth = trial.suggest_int('xgb_depth', 3, 10)
        lr = trial.suggest_loguniform('xgb_lr', 1e-3, 0.3)
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
                              use_label_encoder=False, eval_metric='logloss', n_jobs=1)
        params['model']='xgb'; params.update({'xgb_n':n_estimators,'xgb_depth':max_depth,'xgb_lr':lr})
    elif model_choice == 'lgb':
        n_estimators = trial.suggest_int('lgb_n', 50, 200)
        num_leaves = trial.suggest_int('lgb_leaves', 8, 64)
        lr = trial.suggest_loguniform('lgb_lr', 1e-3, 0.3)
        model = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=lr)
        params['model']='lgb'; params.update({'lgb_n':n_estimators,'lgb_leaves':num_leaves,'lgb_lr':lr})
    else:
        iters = trial.suggest_int('cat_n', 50, 200)
        depth = trial.suggest_int('cat_depth', 3, 8)
        model = CatBoostClassifier(iterations=iters, depth=depth, verbose=0)
        params['model']='cat'; params.update({'cat_n':iters,'cat_depth':depth})

    pipe = Pipeline([('pre', preprocessor), ('model', model)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
    mean_score = float(np.mean(scores))

    # Log to MLflow current trial params and score
    mlflow.log_params(params)
    mlflow.log_metric('f1', mean_score)

    return 1.0 - mean_score

def run_study(X, y, preprocessor, n_trials=20):
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    func = lambda trial: objective(trial, X, y, preprocessor)
    study.optimize(func, n_trials=n_trials, n_jobs=1)
    return study
