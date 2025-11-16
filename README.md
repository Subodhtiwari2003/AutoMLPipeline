# AutoML-Pipeline

AutoML Pipeline (tabular classification) — ready for experimentation and production.
Prepared for Subodh Tiwari.

## Features
- Modular AutoML pipeline using Optuna for hyperparameter search
- Support for XGBoost, LightGBM, CatBoost, RandomForest, Logistic Regression
- MLflow tracking for experiments and artifact storage
- SHAP for explainability
- FastAPI minimal serving example
- Dockerfile & GitHub Actions CI for testing

## Quick start (local)
```bash
# clone
git clone https://github.com/Subodhtiwari2003/AutoML-Pipeline.git
cd AutoML-Pipeline

# create venv and install
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run demo (short)
python run_demo.py
# serves best_pipeline.pkl after run
python serve.py
```

## MLflow
To enable MLflow tracking locally, run:
```bash
mlflow ui --port 5000
```

## Docker
Build and run container:
```bash
docker build -t automl-pipeline:latest .
docker run -p 8000:8000 automl-pipeline:latest
```

## CI
A GitHub Actions workflow runs linting and unit tests on push (Python 3.10).

## License
