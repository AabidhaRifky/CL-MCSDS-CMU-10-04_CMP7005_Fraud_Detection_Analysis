import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# -----------------------------
# PATH SETUP (ROBUST)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "../notebook/models")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "../notebook/final_models")
TABLES_DIR = os.path.join(BASE_DIR, "../notebook/tables")

# -----------------------------
# LOAD MODELS
# -----------------------------
models = {
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(MODEL_DIR, "gradient_boosting_model.pkl")),
    "Final Tuned Model": joblib.load(os.path.join(FINAL_MODEL_DIR, "final_fraud_model.pkl"))
}

# -----------------------------
# LOAD RESULTS TABLE (FIXED)
# -----------------------------
results_df = pd.read_csv(os.path.join(TABLES_DIR, "model_comparison.csv"))

# Clean columns
results_df.columns = results_df.columns.str.strip()

# Add cleaned model column (for safe matching)
results_df["Model_clean"] = results_df["Model"].astype(str).str.strip().str.lower()

# -----------------------------
# SAFE INPUT FUNCTIONS
# -----------------------------
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

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    risk = None
    explanation = None
    selected_model = None
    model_metrics = None
    error = None

    if request.method == "POST":
        try:
            # -----------------------------
            # SAFE MODEL SELECTION
            # -----------------------------
            selected_model = request.form.get("model")

            if not selected_model or selected_model not in models:
                selected_model = "Final Tuned Model"

            # -----------------------------
            # INPUT DATA
            # -----------------------------
            data = {
                "GENDER": request.form.get("gender"),
                "CAR": request.form.get("car"),
                "REALITY": request.form.get("reality"),
                "NO_OF_CHILD": get_int(request.form.get("children")),
                "FAMILY_TYPE": request.form.get("family_type"),
                "HOUSE_TYPE": request.form.get("house_type"),
                "WORK_PHONE": get_int(request.form.get("work_phone")),
                "PHONE": get_int(request.form.get("phone")),
                "FAMILY SIZE": get_float(request.form.get("family_size")),
                "BEGIN_MONTH": get_int(request.form.get("begin_month")),
                "AGE": get_int(request.form.get("age")),
                "YEARS_EMPLOYED": get_float(request.form.get("years_employed")),
                "INCOME": get_float(request.form.get("income")),
                "INCOME_TYPE": request.form.get("income_type"),
                "EDUCATION_TYPE": request.form.get("education_type")
            }

            # Derived feature
            data["INCOME_PER_PERSON"] = data["INCOME"] / max(data["FAMILY SIZE"], 1)

            input_df = pd.DataFrame([data])

            # -----------------------------
            # PREDICTION
            # -----------------------------
            model = models[selected_model]

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            prediction = "Fraud" if pred == 1 else "Not Fraud"
            probability = round(prob, 3)

            # -----------------------------
            # RISK LOGIC
            # -----------------------------
            if prob < 0.2:
                risk = "Low Risk"
                explanation = "Low likelihood of fraud."
            elif prob < 0.5:
                risk = "Medium Risk"
                explanation = "Moderate risk — review recommended."
            else:
                risk = "High Risk"
                explanation = "High fraud likelihood — immediate action required."

            # -----------------------------
            # METRICS (FIXED MATCHING)
            # -----------------------------
            selected_model_clean = str(selected_model).strip().lower()

            model_metrics_df = results_df[
                results_df["Model_clean"] == selected_model_clean
            ]

            if not model_metrics_df.empty:
                model_metrics = model_metrics_df.iloc[0].to_dict()
            else:
                model_metrics = None

        except Exception as e:
            error = str(e)

    return render_template(
        "predict.html",
        models=models.keys(),
        prediction=prediction,
        probability=probability,
        risk=risk,
        explanation=explanation,
        selected_model=selected_model,
        metrics=model_metrics,
        error=error
    )


@app.route("/eda")
def eda():
    figures_path = os.path.join(BASE_DIR, "static/figures")

    # Prevent crash if folder missing
    if not os.path.exists(figures_path):
        images = []
    else:
        images = os.listdir(figures_path)

    return render_template("eda.html", images=images)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)