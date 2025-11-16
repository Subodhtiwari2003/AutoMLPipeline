# Demo script: uses provided CSV dataset, runs an Optuna study (longer), fits final model, saves pipeline, logs to MLflow.
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import quick_profile
from preprocess import build_preprocessor
from model_search import run_study
from train_best import build_final_model_from_params, save_pipeline
import mlflow
import joblib

def main():
    data_path = 'data/binary_classification_sample.csv'
    df = pd.read_csv(data_path)
    quick_profile(df, 'target')
    preprocessor, num_cols, cat_cols = build_preprocessor(df, 'target')
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Starting Optuna study (n_trials=60)...')
    mlflow.set_experiment('automl_demo_full')
    study = run_study(X_train, y_train, preprocessor, n_trials=60)
    print('Best trial params:', study.best_trial.params)
    best_params = study.best_trial.params

    # Fit on full training data
    pipe = build_final_model_from_params(best_params, preprocessor, X_train, y_train)
    save_pipeline(pipe, path='best_pipeline.pkl')

    # Evaluate on holdout
    preds = pipe.predict(X_hold)
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
    print('F1 on holdout:', f1_score(y_hold, preds))
    print('Accuracy:', accuracy_score(y_hold, preds))

    # Log final model to MLflow
    with mlflow.start_run(run_name='final_model'):
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(pipe, artifact_path='model')
        mlflow.log_metric('holdout_f1', float(f1_score(y_hold, preds)))
        mlflow.log_metric('holdout_accuracy', float(accuracy_score(y_hold, preds)))

if __name__ == '__main__':
    main()
