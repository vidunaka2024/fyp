import numpy as np
import shap
import lime
import lime.lime_tabular
from IPython.display import display, HTML
import matplotlib.pyplot as plt

# -------------------------
# Assume X_train, feature_names, and your trained model are already defined.
# For demonstration, let X_new be a single new instance (as a 2D NumPy array)
# e.g., X_new = X_test.values[0:1]  (change as needed)
# -------------------------
X_new = X_test.values[0:1]   # new instance for inference

# Perform inference
y_pred = model.predict(X_new)
print("Model Prediction:", y_pred)

# -------------------------------------------
# SHAP Explanation for the new instance
# -------------------------------------------
# Create a masker using the training data.
masker = shap.maskers.Independent(X_train)

# Create an explainer with the unified SHAP API.
explainer = shap.Explainer(model, masker)

# Compute SHAP values for the new instance.
shap_values_new = explainer(X_new)

# Display a force plot for the new instance.
# (Note: You can choose to display interactively in a Jupyter Notebook.)
shap.initjs()  # Initialize JS visualization for interactive plots
force_plot = shap.force_plot(
    shap_values_new.base_values[0],
    shap_values_new.values[0],
    X_new,
    feature_names=feature_names.tolist()
)
display(force_plot)

# Alternatively, you can generate a summary plot if you want to explain a batch:
# shap.summary_plot(shap_values_new.values, X_new, feature_names=feature_names.tolist())

# -------------------------------------------
# LIME Explanation for the new instance
# -------------------------------------------
# Define a custom prediction function that returns probabilities for both classes.
def predict_proba(x):
    preds = model.predict(x)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    # For binary classification: first column is probability of negative, second column is positive.
    return np.hstack([1 - preds, preds])

# Create a LIME explainer for tabular data.
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names.tolist(),
    class_names=["Negative", "Positive"],
    mode="classification"
)

# LIME expects a 1D array for a single instance.
instance_to_explain = X_new[0]  
lime_exp = lime_explainer.explain_instance(instance_to_explain, predict_proba, num_features=len(feature_names))
display(HTML(lime_exp.as_html()))
