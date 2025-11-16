import shap
def explain_tree_model(pipeline, X_sample):
    pre = pipeline.named_steps['pre']
    model = pipeline.named_steps['model']
    X_trans = pre.transform(X_sample)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    return shap_values
