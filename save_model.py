import mlflow, joblib
def save_model_mlflow(pipeline, run_name='automl_run'):
    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.log_model(pipeline, artifact_path='model')
def save_local(pipeline, path='best_pipeline.pkl'):
    joblib.dump(pipeline, path)
    print('Saved local pipeline to', path)
