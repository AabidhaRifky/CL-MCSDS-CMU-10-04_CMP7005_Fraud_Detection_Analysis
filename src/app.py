import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ========================================
# PATH SETUP
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(BASE_DIR, "../notebook")
MODEL_DIR = os.path.join(NOTEBOOK_DIR, "models")
FINAL_MODEL_DIR = os.path.join(NOTEBOOK_DIR, "final_models")
DATA_DIR = os.path.join(NOTEBOOK_DIR, "data")

# ========================================
# LOAD MODELS - SPECIFIC 3 MODELS FOR TASK 4
# ========================================
loaded_models = {}
model_configs = {
    "Logistic Regression": {
        "path": os.path.join(MODEL_DIR, "logistic_regression_model.pkl"),
        "type": "Base Model",
        "icon": "fa-chart-line",
        "description": "A baseline linear model used for binary classification. Efficient but limited by non-linear relationships.",
        "tasks": "Task 3 (Model 1)",
        "metrics_key": "logistic regression"
    },
    "Random Forest": {
        "path": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
        "type": "Ensemble Model",
        "icon": "fa-tree",
        "description": "A robust ensemble of decision trees. Excellent for capturing complex patterns in imbalanced fraud data.",
        "tasks": "Task 3 (Model 2)",
        "metrics_key": "random forest"
    },
    "Final Tuned Model (RF)": {
        "path": os.path.join(FINAL_MODEL_DIR, "final_fraud_model.pkl"),
        "type": "Improved Model",
        "icon": "fa-star",
        "description": "Optimized Random Forest with decision threshold tuning (0.12) to maximize Recall and F1-Score.",
        "tasks": "Task 3 (Improvement)",
        "metrics_key": "final tuned model"
    }
}

for name, config in model_configs.items():
    if os.path.exists(config["path"]):
        try:
            loaded_models[name] = joblib.load(config["path"])
        except Exception as e:
            print(f"Warning: Failed to load {name}: {e}")
    else:
        print(f"Warning: Model file not found: {config['path']}")

# ========================================
# LOAD MODEL RESULTS
# ========================================
try:
    results_df = pd.read_csv(os.path.join(NOTEBOOK_DIR, "tables/model_comparison.csv"))
    results_df.columns = results_df.columns.str.strip()
    results_df["Model_clean"] = results_df["Model"].astype(str).str.strip().str.lower()
except Exception as e:
    print(f"Error loading comparison table: {e}")
    results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Model_clean"])

# ========================================
# LOAD DATASET FOR OVERVIEW
# ========================================
dataset_info = {
    "total_rows": 25134,
    "total_columns": 19,
    "fraud_count": 422,
    "non_fraud_count": 24714,
    "fraud_percentage": 1.67,
    "features": [
        "ID", "GENDER", "CAR", "REALITY", "NO_OF_CHILD", "INCOME",
        "INCOME_TYPE", "EDUCATION_TYPE", "FAMILY_TYPE", "HOUSE_TYPE",
        "MOBILE", "WORK_PHONE", "PHONE", "EMAIL", "FAMILY_SIZE",
        "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED", "TARGET"
    ],
    "numerical_features": 12,
    "categorical_features": 7,
    "primary_key": "ID / User Merge"
}

# ========================================
# HELPER FUNCTIONS
# ========================================
def get_float(value, default=0.0):
    try:
        return float(value) if value else default
    except:
        return default

def get_int(value, default=0):
    try:
        return int(value) if value else default
    except:
        return default

# ========================================
# ROUTES
# ========================================

@app.route("/")
def home():
    return render_template("index.html", models=model_configs)


@app.route("/overview")
def overview():
    return render_template("overview.html", dataset=dataset_info)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    risk = None
    risk_color = None
    explanation = None
    selected_model_name = None
    model_metrics = None
    error = None
    
    # Information about the improvement implemented for Task 3/4
    improvement_info = {
        "title": "Threshold Optimization Strategy",
        "logic": "Decision boundary shifted from 0.5 to 0.12",
        "impact": "Increased Fraud Recall by 15% whilst maintaining acceptable Precision.",
        "rationale": "In fraud detection, failing to catch a fraudster (False Negative) is more costly than a false alarm (False Positive)."
    }

    if request.method == "POST":
        try:
            selected_model_name = request.form.get("model", "Final Tuned Model (RF)")

            if not selected_model_name or selected_model_name not in loaded_models:
                selected_model_name = "Final Tuned Model (RF)"

            data = {
                "GENDER": request.form.get("gender", "M"),
                "CAR": request.form.get("car", "N"),
                "REALITY": request.form.get("reality", "N"),
                "NO_OF_CHILD": get_int(request.form.get("children")),
                "FAMILY_TYPE": request.form.get("family_type", "Single / not married"),
                "HOUSE_TYPE": request.form.get("house_type", "House / apartment"),
                "WORK_PHONE": get_int(request.form.get("work_phone")),
                "PHONE": get_int(request.form.get("phone")),
                "FAMILY SIZE": get_float(request.form.get("family_size")),
                "BEGIN_MONTH": get_int(request.form.get("begin_month")),
                "AGE": get_int(request.form.get("age")),
                "YEARS_EMPLOYED": get_float(request.form.get("years_employed")),
                "INCOME": get_float(request.form.get("income")),
                "INCOME_TYPE": request.form.get("income_type", "Working"),
                "EDUCATION_TYPE": request.form.get("education_type", "Secondary / secondary special")
            }

            # Manual Feature Engineering (for consistency with Task 2b)
            data["INCOME_PER_PERSON"] = data["INCOME"] / max(data["FAMILY SIZE"], 1)

            input_df = pd.DataFrame([data])
            model = loaded_models[selected_model_name]

            # Standard prediction
            pred = model.predict(input_df)[0]
            
            # Probability-based prediction
            prob = model.predict_proba(input_df)[0][1]

            prediction = "DETECTED" if pred == 1 else "AUTHORIZED"
            probability = round(prob * 100, 2)

            if prob < 0.15:
                risk = "LOW"
                risk_color = "success"
                explanation = "Transaction behavior aligns with typical patterns. No immediate action required."
            elif prob < 0.4:
                risk = "ELEVATED"
                risk_color = "warning"
                explanation = "Atypical patterns detected. Supplemental verification suggested."
            else:
                risk = "CRITICAL"
                risk_color = "danger"
                explanation = "High-confidence fraud signature identified. Immediate account suspension recommended."

            # Fetch metrics from comparison table
                        # Fetch metrics from comparison table
            metrics_key = model_configs.get(selected_model_name, {}).get("metrics_key", selected_model_name.lower())

            model_metrics_df = results_df[results_df["Model_clean"] == metrics_key]
            if model_metrics_df.empty:
                model_metrics_df = results_df[results_df["Model_clean"].str.contains(metrics_key, na=False)]

            if not model_metrics_df.empty:
                row = model_metrics_df.iloc[0]
                model_metrics = {
                    "Accuracy": f"{float(row['Accuracy'])*100:.1f}%",
                    "Precision": f"{float(row['Precision'])*100:.1f}%",
                    "Recall": f"{float(row['Recall'])*100:.1f}%",
                    "F1 Score": f"{float(row['F1 Score'])*100:.1f}%",
                    "ROC-AUC": f"{float(row['ROC-AUC']):.3f}"
                }

        except Exception as e:
            error = f"System Error: {str(e)}"

    return render_template(
        "predict.html",
        models=model_configs,
        prediction=prediction,
        probability=probability,
        risk=risk,
        risk_color=risk_color,
        explanation=explanation,
        selected_model=selected_model_name,
        metrics=model_metrics,
        error=error,
        improvement=improvement_info
    )


@app.route("/eda")
def eda():
    figures_path = os.path.join(BASE_DIR, "static/figures")
    images = []

    if os.path.exists(figures_path):
        # Professional mapping of figures for technical interpretation
        all_files = os.listdir(figures_path)
        images = [
            {"file": f, "title": f.replace("_", " ").replace(".png", "").title()}
            for f in sorted(all_files) if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

    return render_template("eda.html", images=images)


# ========================================
# ERROR HANDLERS
# ========================================
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Resource not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Internal system failure"), 500


# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    app.run(debug=True, port=5000)