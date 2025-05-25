import os
from io import BytesIO
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import base64


from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np

# FuzzyWuzzy requires python-Levenshtein installed for best performance
from fuzzywuzzy import fuzz
from itertools import product

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import SHAP and matplotlib for feature importance plotting
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "some_random_secret_key"

# Global variables to store data in memory
df1 = None
df2 = None
col_matches = {}    # df1_col -> df2_col
col_roles = {}      # df1_col -> role (date, text, numeric, ignore)
pairs = None        # DataFrame of all row-pairs
threshold_inputs = {}  # col -> threshold
trained_model = None
selected_fuzzy_cols = None  # New: user-selected fuzzy matching columns

##############################################################################
# 1) Home page
##############################################################################
@app.route("/")
def home():
    return render_template("index.html")


##############################################################################
# 2) File upload route
##############################################################################
@app.route("/upload", methods=["GET", "POST"])
def upload_files():
    global df1, df2

    if request.method == "POST":
        # We expect two files: file1, file2
        if "file1" not in request.files or "file2" not in request.files:
            flash("Please upload both XLSX files.")
            return redirect(request.url)

        file1 = request.files["file1"]
        file2 = request.files["file2"]

        if file1.filename == "" or file2.filename == "":
            flash("Please select files before uploading.")
            return redirect(request.url)

        # Read into DataFrame
        try:
            df1 = pd.read_excel(BytesIO(file1.read()))
            df2 = pd.read_excel(BytesIO(file2.read()))
        except Exception as e:
            flash(f"Error reading Excel files: {e}")
            return redirect(request.url)

        flash("DataFrames loaded successfully!")
        return redirect(url_for("select_fuzzy_features"))

    return render_template("upload.html")


##############################################################################
# New: Select Fuzzy Matching Columns
##############################################################################
@app.route("/select_fuzzy_features", methods=["GET", "POST"])
def select_fuzzy_features():
    global selected_fuzzy_cols, df1
    if df1 is None:
        flash("Please upload data first.")
        return redirect(url_for("upload_files"))
    if request.method == "POST":
        # Retrieve list of columns that the user wants to use for fuzzy matching
        selected_fuzzy_cols = request.form.getlist("fuzzy_cols")
        flash("Fuzzy matching columns updated!")
        return redirect(url_for("match_columns"))
    return render_template("select_fuzzy_features.html", df1_columns=df1.columns)


##############################################################################
# 3) Fuzzy-match columns route
##############################################################################
@app.route("/match_columns", methods=["GET", "POST"])
def match_columns():
    global df1, df2, col_matches, selected_fuzzy_cols

    if df1 is None or df2 is None:
        flash("Please upload data first.")
        return redirect(url_for("upload_files"))

    if request.method == "POST":
        # On POST, we confirm the matches from the user
        overridden_matches = {}
        for c1 in df1.columns:
            override_val = request.form.get(f"match_{c1}", "")
            if override_val == "None" or override_val == "":
                overridden_matches[c1] = None
            else:
                overridden_matches[c1] = override_val

        col_matches = overridden_matches
        flash("Column matching updated!")
        return redirect(url_for("guess_roles"))

    # On GET, do the fuzzy matching (or re-do it each time)
    col_matches = fuzzy_match_columns(df1.columns, df2.columns, threshold=60)
    return render_template(
        "match_columns.html",
        df1_columns=df1.columns,
        df2_columns=df2.columns,
        col_matches=col_matches,
    )


def fuzzy_match_columns(list1, list2, threshold=60):
    """
    Fuzzy-match columns in list1 to columns in list2 using FuzzyWuzzy ratio.
    If selected_fuzzy_cols is set and non-empty, only match those columns.
    Otherwise, match all columns in list1.
    """
    global selected_fuzzy_cols
    matched = {}

    # Determine which columns to match
    if not selected_fuzzy_cols:  # None or empty list
        columns_to_match = list1
    else:
        columns_to_match = selected_fuzzy_cols

    for c1 in columns_to_match:
        best_match = None
        best_score = -1
        for c2 in list2:
            score = fuzz.ratio(str(c1).lower(), str(c2).lower())
            if score > best_score:
                best_score = score
                best_match = c2

        if best_score >= threshold:
            matched[c1] = best_match
        else:
            matched[c1] = None

    return matched


##############################################################################
# 4) Guess roles
##############################################################################
@app.route("/guess_roles", methods=["GET", "POST"])
def guess_roles():
    global df1, df2, col_matches, col_roles

    if df1 is None or df2 is None:
        flash("Please upload data first.")
        return redirect(url_for("upload_files"))

    if not col_matches:
        flash("No fuzzy matches found. Please match columns first.")
        return redirect(url_for("match_columns"))

    if request.method == "POST":
        # On POST, the user might override roles
        new_roles = {}
        for c1 in col_matches.keys():
            role_val = request.form.get(f"role_{c1}", "ignore")
            new_roles[c1] = role_val
        col_roles = new_roles
        flash("Column roles updated!")
        return redirect(url_for("create_pairs"))

    # On GET, guess roles automatically
    col_roles = {}
    for c1, c2 in col_matches.items():
        if c2 is None:
            col_roles[c1] = "ignore"
            continue
        combined_name = (c1 + c2).lower()
        if "date" in combined_name:
            col_roles[c1] = "date"
        elif any(x in combined_name for x in ["desc", "memo", "text"]):
            col_roles[c1] = "text"
        elif any(x in combined_name for x in ["amount", "debit", "credit", "balance", "amt"]):
            col_roles[c1] = "numeric"
        else:
            col_roles[c1] = "ignore"

    return render_template(
        "guess_roles.html",
        col_matches=col_matches,
        col_roles=col_roles
    )


##############################################################################
# 5) Create pairs
##############################################################################
@app.route("/create_pairs", methods=["GET", "POST"])
def create_pairs():
    global df1, df2, col_matches, col_roles, pairs

    if df1 is None or df2 is None:
        flash("Please upload data first.")
        return redirect(url_for("upload_files"))
    if not col_matches:
        flash("No column matches found.")
        return redirect(url_for("match_columns"))
    if not col_roles:
        flash("Column roles not determined.")
        return redirect(url_for("guess_roles"))

    # Creating pairs
    pairs_list = list(product(df1.index, df2.index))
    data_for_pairs = []
    for (i, j) in pairs_list:
        row1 = df1.loc[i]
        row2 = df2.loc[j]
        row_dict = {}
        for c1, c2 in col_matches.items():
            if c2 is None:
                continue
            row_dict[f"df1_{c1}"] = row1[c1]
            row_dict[f"df2_{c2}"] = row2[c2]
        data_for_pairs.append(row_dict)

    pairs = pd.DataFrame(data_for_pairs)

    flash("Pairs created successfully.")
    return render_template("create_pairs.html", pair_count=len(pairs))


##############################################################################
# 6) Compute features
##############################################################################
@app.route("/compute_features", methods=["GET", "POST"])
def compute_features():
    global pairs, col_roles, col_matches

    if pairs is None or pairs.empty:
        flash("Pairs DataFrame does not exist or is empty.")
        return redirect(url_for("create_pairs"))

    # Compute features for each matched column based on its role
    for c1, role in col_roles.items():
        c2 = col_matches[c1]
        if c2 is None:
            continue

        col_df1 = f"df1_{c1}"
        col_df2 = f"df2_{c2}"

        if role == "date":
            # Convert to datetime, compute days difference
            pairs[col_df1] = pd.to_datetime(pairs[col_df1], errors="coerce")
            pairs[col_df2] = pd.to_datetime(pairs[col_df2], errors="coerce")
            diff_col = f"{c1}_diff"
            pairs[diff_col] = (pairs[col_df1] - pairs[col_df2]).abs().dt.days

        elif role == "text":
            sim_col = f"{c1}_sim"
            pairs[sim_col] = pairs.apply(
                lambda row: fuzz.ratio(str(row[col_df1]), str(row[col_df2]))
                if pd.notnull(row[col_df1]) and pd.notnull(row[col_df2]) else 0,
                axis=1
            )

        elif role == "numeric":
            diff_col = f"{c1}_diff"
            pairs[diff_col] = (
                pairs[col_df1].astype(float, errors="ignore")
                - pairs[col_df2].astype(float, errors="ignore")
            ).abs()

        else:
            # ignore
            pass

    flash("Features computed (columns ending in _diff or _sim).")
    return render_template("compute_features.html", columns=pairs.columns)


##############################################################################
# 7) Set thresholds & Labeling
##############################################################################
@app.route("/set_thresholds", methods=["GET", "POST"])
def set_thresholds():
    global pairs

    if pairs is None or pairs.empty:
        flash("Pairs DataFrame does not exist or is empty.")
        return redirect(url_for("create_pairs"))

    # Identify diff vs sim columns
    diff_cols = [c for c in pairs.columns if c.endswith("_diff")]
    sim_cols = [c for c in pairs.columns if c.endswith("_sim")]

    if request.method == "GET":
        return render_template("set_thresholds.html", diff_cols=diff_cols, sim_cols=sim_cols)

    global threshold_inputs
    threshold_inputs.clear()

    for col in diff_cols:
        val = request.form.get(col, "10.0")  # default
        threshold_inputs[col] = float(val)

    for col in sim_cols:
        val = request.form.get(col, "60.0")  # default
        threshold_inputs[col] = float(val)

    flash("Thresholds set successfully!")
    return redirect(url_for("label_pairs"))


@app.route("/label_pairs")
def label_pairs():
    global pairs, threshold_inputs

    if pairs is None or pairs.empty:
        flash("Pairs DataFrame does not exist or is empty.")
        return redirect(url_for("create_pairs"))
    if not threshold_inputs:
        flash("No thresholds set. Please set thresholds first.")
        return redirect(url_for("set_thresholds"))

    labels = []
    for idx, row in pairs.iterrows():
        violation = False
        for col, thr in threshold_inputs.items():
            val = row[col]
            if col.endswith("_diff"):
                if val > thr:
                    violation = True
                    break
            elif col.endswith("_sim"):
                if val < thr:
                    violation = True
                    break
        labels.append(1 if violation else 0)

    pairs["label"] = labels
    flash("Labeling done. 'label' column created (1=violation, 0=within thresholds).")
    return render_template("label_pairs.html", labeled_count=sum(labels))


##############################################################################
# 8) Train RandomForest model & save as .pkl
##############################################################################
@app.route("/train_model")
def train_model():
    global pairs, trained_model

    if pairs is None or pairs.empty:
        flash("Pairs DataFrame does not exist.")
        return redirect(url_for("create_pairs"))

    if "label" not in pairs.columns:
        flash("No 'label' column found in pairs. Please label pairs first.")
        return redirect(url_for("label_pairs"))

    feature_cols = [
        col for col in pairs.columns
        if (col.endswith("_diff") or col.endswith("_sim"))
        and pd.api.types.is_numeric_dtype(pairs[col])
    ]

    if not feature_cols:
        flash("No numeric feature columns found to train on.")
        return redirect(url_for("compute_features"))

    X = pairs[feature_cols]
    y = pairs["label"].astype(int)

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    trained_model = model

    # Save model to .pkl
    model_path = "trained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    flash(f"Model trained! Test accuracy: {accuracy:.3f} | Saved to {model_path}")
    return render_template("train_model.html", accuracy=accuracy)


@app.route("/shap_importance")
def shap_importance():
    global trained_model, pairs, feature_importance_global
    global trained_model, pairs
    if trained_model is None:
        flash("Please train the model first.")
        return redirect(url_for("train_model"))

    # Identify feature columns ending with _diff or _sim that are numeric.
    feature_cols = [
        col for col in pairs.columns
        if (col.endswith("_diff") or col.endswith("_sim"))
        and pd.api.types.is_numeric_dtype(pairs[col])
    ]
    X = pairs[feature_cols]

    # Create SHAP explainer and compute values.
    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Compute mean absolute SHAP values.
    if shap_values.ndim == 3:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    sorted_idx = np.argsort(mean_shap)[::-1]
    raw_features_sorted = np.array(X.columns)[sorted_idx]
    features_sorted = [str(x) for x in raw_features_sorted]
    importance_sorted = mean_shap[sorted_idx]
    feature_importance = list(zip(features_sorted, importance_sorted))

    # --- Generate SHAP Summary Plot Image ---
    summary_img = BytesIO()
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(summary_img, format="png", bbox_inches="tight")
    plt.close()
    summary_img.seek(0)

    # --- Generate Feature Importance Bar Chart Image ---
    bar_img = BytesIO()
    plt.figure()
    plt.barh(features_sorted, importance_sorted)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance Ranking")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(bar_img, format="png", bbox_inches="tight")
    plt.close()
    bar_img.seek(0)

    # --- Combine the Two Images Side by Side ---
    summary_plot_img = Image.open(summary_img)
    bar_chart_img = Image.open(bar_img)
    width1, height1 = summary_plot_img.size
    width2, height2 = bar_chart_img.size
    total_width = width1 + width2
    max_height = max(height1, height2)
    combined_img = Image.new("RGB", (total_width, max_height))
    combined_img.paste(summary_plot_img, (0, 0))
    combined_img.paste(bar_chart_img, (width1, 0))

    # Convert the combined image to a base64 string.
    final_img = BytesIO()
    combined_img.save(final_img, format="PNG")
    final_img.seek(0)
    base64_img = base64.b64encode(final_img.read()).decode('utf-8')
    shap_img_data = f"data:image/png;base64,{base64_img}"
    feature_importance_global = list(zip(features_sorted, importance_sorted))

    return render_template(
        "shap_importance.html",
        shap_img_data=shap_img_data,
        feature_importance=feature_importance_global
    )




##############################################################################
# Run the Flask application
##############################################################################
if __name__ == "__main__":
    app.run(debug=True)
