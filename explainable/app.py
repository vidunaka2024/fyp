#!/usr/bin/env python
# app.py

import base64
import io
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import OpenAI and set your API key
from openai import OpenAI

client = OpenAI(api_key="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")
  # Replace with your actual API key

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Data Loading and Feature Calculation
#
# Load data from the CSV file "pairs_with_label.csv"
#
pairs = pd.read_csv("pairs_with_label.csv")
feature_cols = ["date_diff", "desc_similarity", "debit_diff", "credit_diff", "balance_diff"]

# Use the "label" column from the CSV as the target
X = pairs[feature_cols]
y = pairs["label"]

# -----------------------------------------------------------------------------
# Train a model (use your actual training process in production)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Define class names (adapt for your use case)
class_names = ["NoMatch", "Match"]

# -----------------------------------------------------------------------------
# Prepare LIME Explainer (we reinitialize SHAP per request)
# -----------------------------------------------------------------------------
def predict_proba(df: pd.DataFrame):
    return model.predict_proba(df)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    class_names=class_names,
    mode="classification"
)

# -----------------------------------------------------------------------------
# Helper: Convert matplotlib figure to a base64-encoded PNG string
# -----------------------------------------------------------------------------
def fig_to_base64_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded

# -----------------------------------------------------------------------------
# Helper: Generate LIME explanation text using the ChatGPT API
# -----------------------------------------------------------------------------
def generate_lime_explanation_with_chatgpt(lime_exp):
    """
    Extracts LIME explanation details and sends them to ChatGPT to obtain
    a short, human-readable explanation.
    """
    # Extract details from the LIME explanation object (list of tuples)
    lime_details = lime_exp.as_list()  # e.g., [("balance_diff <= 20307500.00", -0.007), ...]
    explanation_str = "LIME Explanation Details:\n"
    for feature_range, weight in lime_details:
        explanation_str += f"- {feature_range} (weight={weight:.4f})\n"

    prompt = f"""
You are a helpful AI that explains machine learning model explanations in simple terms.

Given the following LIME explanation details for a prediction (class 'Match'), please provide a short, concise explanation of what the features and their weights indicate about the model's decision:

{explanation_str}

Please keep the explanation brief and easy to understand.
    """

    # Call the ChatGPT API
    response = client.chat.completions.create(model="gpt-3.5-turbo",  # or another available ChatGPT model
    messages=[
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=200,
    temperature=0.7)

    chatgpt_explanation = response.choices[0].message.content.strip()
    return chatgpt_explanation

# -----------------------------------------------------------------------------
# Flask Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return (
        "<h1>Welcome to the local SHAP/LIME Flask app!</h1>"
        "<p>POST to /explain with JSON like: { \"data\": [date_diff, desc_similarity, debit_diff, credit_diff, balance_diff] }</p>"
    )

@app.route("/explain", methods=["POST"])
def explain():
    """
    Expects JSON like:
      {
        "data": [date_diff, desc_similarity, debit_diff, credit_diff, balance_diff]
      }
    Returns SHAP & LIME explanation plots as base64 strings, along with a ChatGPT-generated explanation for LIME.
    """
    try:
        content = request.get_json()
        raw_features = content.get("data", [])
        if not raw_features:
            return jsonify({"error": "No feature data provided"}), 400
        if len(raw_features) != len(feature_cols):
            return jsonify({"error": "Wrong number of features"}), 400

        # Create a DataFrame using the incoming data
        instance_df = pd.DataFrame([raw_features], columns=feature_cols)
        instance_df = instance_df.fillna(0.0).astype(float)

        # Predict class for the instance
        predicted_class = model.predict(instance_df)[0]

        # Reinitialize SHAP explainer for fresh explanations (avoids caching issues)
        masker = shap.maskers.Independent(X_train)
        shap_explainer_instance = shap.Explainer(model, masker)
        # Use the raw numpy array (same as LIME) for SHAP input
        raw_instance = instance_df.iloc[0].values.reshape(1, -1)
        shap_values = shap_explainer_instance(raw_instance, check_additivity=False)
        ex_single = shap_values[0]  # shape: (n_features, n_classes)
        ex_for_pred_class = ex_single[..., predicted_class]

        # Generate SHAP bar plot using a new figure (like LIME's figure generation)
        fig_shap = plt.figure()
        shap.plots.bar(ex_for_pred_class, show=False)
        shap_plot_b64 = fig_to_base64_img(fig_shap)
        plt.close(fig_shap)

        # Compute LIME explanation using the same raw data
        lime_exp = lime_explainer.explain_instance(
            instance_df.iloc[0].values,
            predict_proba,
            num_features=len(feature_cols)
        )
        fig_lime = lime_exp.as_pyplot_figure()
        lime_plot_b64 = fig_to_base64_img(fig_lime)
        plt.close(fig_lime)

        # Generate a text explanation for the LIME plot using ChatGPT API
        lime_explanation_text = generate_lime_explanation_with_chatgpt(lime_exp)

        return jsonify({
            "shap_plot_b64": shap_plot_b64,
            "lime_plot_b64": lime_plot_b64,
            "lime_explanation": lime_explanation_text
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
