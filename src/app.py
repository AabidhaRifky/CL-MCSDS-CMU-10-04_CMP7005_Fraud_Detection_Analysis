from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from flask import Flask, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "tables"

DATASET_1_PATTERNS = ["*Credit*Sept_1*.csv", "*Credir*Sept_1*.csv"]
DATASET_2_PATTERNS = ["*Credit*Sept_2*.csv", "*Credir*Sept_2*.csv"]

PREDICTION_COLUMNS = [
    "NO_OF_CHILD",
    "WORK_PHONE",
    "PHONE",
    "E_MAIL",
    "FAMILY SIZE",
    "BEGIN_MONTH",
    "AGE",
    "YEARS_EMPLOYED",
    "INCOME",
    "GENDER",
    "CAR",
    "REALITY",
    "FAMILY_TYPE",
    "HOUSE_TYPE",
    "INCOME_TYPE",
    "EDUCATION_TYPE",
]


def find_file(folder: Path, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(folder.glob(pattern))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        return None
    matches = sorted(matches, key=lambda p: (len(str(p)), str(p)))
    return matches[0]


def load_and_merge_data():
    file1 = find_file(DATA_DIR, DATASET_1_PATTERNS)
    file2 = find_file(DATA_DIR, DATASET_2_PATTERNS)

    if file1 is None or file2 is None:
        raise FileNotFoundError(
            "Dataset files were not found inside the data folder. "
            "Please place both Sept_1 and Sept_2 CSV files inside the data folder."
        )

    data_1 = pd.read_csv(file1)
    data_2 = pd.read_csv(file2)

    data_raw = pd.merge(
        data_1,
        data_2,
        left_on="ID",
        right_on="User",
        how="outer",
        validate="one_to_one"
    )

    data = data_raw.drop(columns=[c for c in ["User", "Unnamed: 0"] if c in data_raw.columns]).copy()
    return data, file1.name, file2.name


def preprocess_for_display(data: pd.DataFrame) -> pd.DataFrame:
    data_clean = data.copy()

    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data_clean.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_columns:
        data_clean[col] = data_clean[col].fillna(data_clean[col].median())

    for col in categorical_columns:
        mode_value = data_clean[col].mode(dropna=True)
        fill_value = mode_value.iloc[0] if len(mode_value) > 0 else "Unknown"
        data_clean[col] = data_clean[col].fillna(fill_value)

    candidate_numeric_cols = [col for col in ["AGE", "YEARS_EMPLOYED", "INCOME"] if col in data_clean.columns]

    for col in candidate_numeric_cols:
        q1 = data_clean[col].quantile(0.25)
        q3 = data_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clean[col] = data_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return data_clean


def add_engineered_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"INCOME", "FAMILY SIZE"}.issubset(df.columns):
        family_size = pd.to_numeric(df["FAMILY SIZE"], errors="coerce").replace(0, np.nan)
        income = pd.to_numeric(df["INCOME"], errors="coerce")
        df["INCOME_PER_FAMILY_MEMBER"] = income / family_size

    contact_cols = [col for col in ["PHONE", "WORK_PHONE", "E_MAIL"] if col in df.columns]
    if contact_cols:
        contact_data = df[contact_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["HAS_CONTACT_INFO"] = (contact_data > 0).sum(axis=1)

    if "FAMILY SIZE" in df.columns:
        family_size_raw = pd.to_numeric(df["FAMILY SIZE"], errors="coerce")
        df["IS_LARGE_FAMILY"] = (family_size_raw >= 4).astype(int)

    return df


def get_category_options(data_clean: pd.DataFrame):
    category_options = {}
    for col in ["GENDER", "CAR", "REALITY", "FAMILY_TYPE", "HOUSE_TYPE", "INCOME_TYPE", "EDUCATION_TYPE"]:
        if col in data_clean.columns:
            category_options[col] = sorted(data_clean[col].dropna().astype(str).unique().tolist())
    return category_options


def load_model_comparison():
    comparison_file = TABLES_DIR / "final_model_comparison.csv"
    if comparison_file.exists():
        return pd.read_csv(comparison_file)

    return pd.DataFrame([
        {
            "Model": "Random Forest",
            "Version": "Threshold tuned on validation (0.12)",
            "Accuracy": 0.976129,
            "Precision": 0.308511,
            "Recall": 0.345238,
            "F1_Score": 0.325843,
            "ROC_AUC": 0.773008,
            "Average_Precision": 0.198068,
            "Threshold": 0.12,
        },
        {
            "Model": "Random Forest (RandomizedSearchCV)",
            "Version": "Threshold tuned on validation (0.43)",
            "Accuracy": 0.980107,
            "Precision": 0.366667,
            "Recall": 0.261905,
            "F1_Score": 0.305556,
            "ROC_AUC": 0.780965,
            "Average_Precision": 0.187064,
            "Threshold": 0.43,
        },
        {
            "Model": "Random Forest (RandomizedSearchCV)",
            "Version": "Default threshold (0.50)",
            "Accuracy": 0.982296,
            "Precision": 0.435897,
            "Recall": 0.202381,
            "F1_Score": 0.276423,
            "ROC_AUC": 0.780965,
            "Average_Precision": 0.187064,
            "Threshold": 0.50,
        },
        {
            "Model": "Random Forest",
            "Version": "Default threshold (0.50)",
            "Accuracy": 0.981699,
            "Precision": 0.346154,
            "Recall": 0.107143,
            "F1_Score": 0.163636,
            "ROC_AUC": 0.773008,
            "Average_Precision": 0.198068,
            "Threshold": 0.50,
        },
        {
            "Model": "Logistic Regression",
            "Version": "Threshold tuned on validation (0.76)",
            "Accuracy": 0.960215,
            "Precision": 0.097222,
            "Recall": 0.166667,
            "F1_Score": 0.122807,
            "ROC_AUC": 0.668636,
            "Average_Precision": 0.091635,
            "Threshold": 0.76,
        },
        {
            "Model": "Logistic Regression",
            "Version": "Default threshold (0.50)",
            "Accuracy": 0.654466,
            "Precision": 0.029060,
            "Recall": 0.607143,
            "F1_Score": 0.055465,
            "ROC_AUC": 0.668636,
            "Average_Precision": 0.091635,
            "Threshold": 0.50,
        },
    ])


def get_models():
    loaded_models = {}

    rf_model_path = MODELS_DIR / "random_forest_balanced_pipeline.pkl"
    rf_threshold_path = MODELS_DIR / "random_forest_best_threshold.pkl"
    final_model_path = MODELS_DIR / "final_selected_model_pipeline.pkl"
    final_threshold_path = MODELS_DIR / "final_selected_threshold.pkl"

    rf_rs_model_path = MODELS_DIR / "random_forest_randomized_search_pipeline.pkl"
    rf_rs_threshold_path = MODELS_DIR / "random_forest_randomized_search_best_threshold.pkl"

    lr_model_path = MODELS_DIR / "logistic_regression_pipeline.pkl"
    lr_threshold_path = MODELS_DIR / "logistic_regression_best_threshold.pkl"

    rf_pipeline = None
    rf_tuned_threshold = 0.12

    rf_rs_pipeline = None
    rf_rs_tuned_threshold = 0.43

    lr_pipeline = None
    lr_tuned_threshold = 0.76

    # Base Random Forest
    if rf_model_path.exists():
        rf_pipeline = joblib.load(rf_model_path)
    elif final_model_path.exists():
        rf_pipeline = joblib.load(final_model_path)

    if rf_threshold_path.exists():
        rf_tuned_threshold = float(joblib.load(rf_threshold_path))
    elif final_threshold_path.exists():
        rf_tuned_threshold = float(joblib.load(final_threshold_path))

    if rf_pipeline is not None:
        loaded_models[f"Random Forest - Threshold tuned on validation ({rf_tuned_threshold:.2f})"] = {
            "pipeline": rf_pipeline,
            "threshold": rf_tuned_threshold,
            "family": "Random Forest",
            "variant": f"Threshold tuned on validation ({rf_tuned_threshold:.2f})",
            "description": "Random Forest using the validation-tuned threshold selected from Task 3."
        }

        loaded_models["Random Forest - Default threshold (0.50)"] = {
            "pipeline": rf_pipeline,
            "threshold": 0.50,
            "family": "Random Forest",
            "variant": "Default threshold (0.50)",
            "description": "Random Forest using the default classification threshold."
        }

    # Random Forest (RandomizedSearchCV)
    if rf_rs_model_path.exists():
        rf_rs_pipeline = joblib.load(rf_rs_model_path)

        if rf_rs_threshold_path.exists():
            rf_rs_tuned_threshold = float(joblib.load(rf_rs_threshold_path))

        loaded_models[f"Random Forest (RandomizedSearchCV) - Threshold tuned on validation ({rf_rs_tuned_threshold:.2f})"] = {
            "pipeline": rf_rs_pipeline,
            "threshold": rf_rs_tuned_threshold,
            "family": "Random Forest (RandomizedSearchCV)",
            "variant": f"Threshold tuned on validation ({rf_rs_tuned_threshold:.2f})",
            "description": "Random Forest after RandomizedSearchCV hyperparameter tuning and validation-based threshold tuning."
        }

        loaded_models["Random Forest (RandomizedSearchCV) - Default threshold (0.50)"] = {
            "pipeline": rf_rs_pipeline,
            "threshold": 0.50,
            "family": "Random Forest (RandomizedSearchCV)",
            "variant": "Default threshold (0.50)",
            "description": "Random Forest after RandomizedSearchCV hyperparameter tuning using the default classification threshold."
        }

    # Logistic Regression
    if lr_model_path.exists():
        lr_pipeline = joblib.load(lr_model_path)

        if lr_threshold_path.exists():
            lr_tuned_threshold = float(joblib.load(lr_threshold_path))

        loaded_models[f"Logistic Regression - Threshold tuned on validation ({lr_tuned_threshold:.2f})"] = {
            "pipeline": lr_pipeline,
            "threshold": lr_tuned_threshold,
            "family": "Logistic Regression",
            "variant": f"Threshold tuned on validation ({lr_tuned_threshold:.2f})",
            "description": "Logistic Regression using the validation-tuned threshold selected from Task 3."
        }

        loaded_models["Logistic Regression - Default threshold (0.50)"] = {
            "pipeline": lr_pipeline,
            "threshold": 0.50,
            "family": "Logistic Regression",
            "variant": "Default threshold (0.50)",
            "description": "Logistic Regression using the default classification threshold."
        }

    return loaded_models

def get_eda_figures():
    figures = [
        {
            "filename": "missing_values_before_treatment.png",
            "title": "Missing Values Before Treatment",
            "caption": "This figure shows that only a small number of values were missing, mainly in INCOME_TYPE, YEARS_EMPLOYED and FAMILY SIZE."
        },
        {
            "filename": "boxplots_after_outlier_capping.png",
            "title": "Boxplots After Outlier Treatment",
            "caption": "Only AGE, YEARS_EMPLOYED and INCOME were capped for EDA. Count variables were not capped to avoid unrealistic fractional values."
        },
        {
            "filename": "target_class_distribution.png",
            "title": "Target Class Distribution",
            "caption": "The target distribution is highly imbalanced, so accuracy alone is not a reliable performance metric."
        },
        {
            "filename": "numerical_distributions.png",
            "title": "Numerical Distributions",
            "caption": "These histograms show how the main continuous variables are distributed across the dataset."
        },
        {
    "filename": "categorical_distributions.png",
    "title": "Categorical Distributions",
    "caption": "These charts summarise the frequency of the main categorical variables."
},
{
    "filename": "fraud_rate_by_category.png",
    "title": "Fraud Rate by Category",
    "caption": "These charts compare fraud-rate percentages across selected categorical variables, which is more informative than raw counts for an imbalanced fraud dataset."
},
{
    "filename": "bivariate_numeric_by_target.png",
    "title": "AGE and INCOME by TARGET",
    "caption": "These boxplots compare AGE and INCOME across non-fraud and fraud cases and show that the classes overlap rather than separating cleanly."
},
{
    "filename": "correlation_heatmap.png",
    "title": "Correlation Heatmap",
    "caption": "The heatmap excludes identifier-like variables and focuses only on analytical numeric columns."
},
    ]

    return [fig for fig in figures if (FIGURES_DIR / fig["filename"]).exists()]


def get_model_figures():
    figure_candidates = [
        {
            "filenames": ["final_selected_model_confusion_matrix.png", "random_forest_balanced_tuned_confusion_matrix.png"],
            "title": "Final Selected Model Confusion Matrix",
            "caption": "This matrix shows the final fraud predictions after threshold tuning."
        },
        {
            "filenames": ["roc_curve_comparison.png"],
            "title": "ROC Curve Comparison",
            "caption": "The RandomizedSearchCV Random Forest achieved the highest ROC-AUC, although the threshold-tuned Random Forest was selected as the final model because it achieved the best F1-score."
        },
        {
            "filenames": ["precision_recall_curve_comparison.png"],
            "title": "Precision-Recall Curve Comparison",
            "caption": "The tree-based models outperform Logistic Regression on this imbalanced dataset; however, the threshold-tuned Random Forest achieved the strongest final F1-score."
        },
        {
            "filenames": ["top15_feature_importance_final_model.png", "top15_feature_importance_random_forest.png"],
            "title": "Top 15 Feature Importances",
            "caption": "BEGIN_MONTH, AGE, YEARS_EMPLOYED and INCOME_PER_FAMILY_MEMBER are among the most influential features in the final model."
        },
    ]

    resolved = []
    for item in figure_candidates:
        chosen = next((name for name in item["filenames"] if (FIGURES_DIR / name).exists()), None)
        if chosen is not None:
            resolved.append({
                "filename": chosen,
                "title": item["title"],
                "caption": item["caption"],
            })
    return resolved


raw_data, file1_name, file2_name = load_and_merge_data()
clean_data = preprocess_for_display(raw_data)
category_options = get_category_options(clean_data)
models = get_models()
model_comparison_df = load_model_comparison()

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/figures/<path:filename>")
def figure_file(filename):
    return send_from_directory(FIGURES_DIR, filename)


@app.route("/")
def index():
    rows, cols = raw_data.shape
    fraud_count = int(raw_data["TARGET"].sum()) if "TARGET" in raw_data.columns else 0
    nonfraud_count = rows - fraud_count
    missing_count = int(raw_data.isnull().sum().sum())

    preview_df = raw_data.head(15).reset_index(drop=True)

    return render_template(
        "index.html",
        file1_name=file1_name,
        file2_name=file2_name,
        rows=rows,
        cols=cols,
        fraud_count=fraud_count,
        nonfraud_count=nonfraud_count,
        missing_count=missing_count,
        preview_columns=preview_df.columns.tolist(),
        preview_rows=preview_df.values.tolist(),
    )


@app.route("/eda")
def eda():
    numeric_cols = clean_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = clean_data.select_dtypes(exclude=[np.number]).columns.tolist()

    findings = [
        "The analytical dataset contains 25,134 rows and 19 columns after removing redundant merge keys.",
        "Missing values were limited and were mainly found in INCOME_TYPE, YEARS_EMPLOYED and FAMILY SIZE.",
        "Count variables such as NO_OF_CHILD and FAMILY SIZE were not outlier-capped because fractional values would be unrealistic.",
        "The target variable is highly imbalanced, so precision, recall, F1-score and Average Precision are more informative than accuracy alone.",
        "Category-level fraud rates should be interpreted cautiously where subgroup sizes are very small.",
        "For modelling, preprocessing was handled inside sklearn pipelines to avoid data leakage."
    ]

    return render_template(
        "eda.html",
        numeric_count=len(numeric_cols),
        categorical_count=len(categorical_cols),
        eda_figures=get_eda_figures(),
        findings=findings
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model_names = list(models.keys())
    model_figures = get_model_figures()

    preferred_rf_name = next(
        (name for name in model_names if name.startswith("Random Forest - Threshold tuned on validation")),
        model_names[0] if model_names else ""
    )

    defaults = {
        "NO_OF_CHILD": "0",
        "FAMILY SIZE": "2",
        "BEGIN_MONTH": "24",
        "AGE": "40",
        "YEARS_EMPLOYED": "5",
        "INCOME": "180000",
        "WORK_PHONE": "0",
        "PHONE": "0",
        "E_MAIL": "0",
        "GENDER": "F",
        "CAR": "N",
        "REALITY": "Y",
        "FAMILY_TYPE": "Married",
        "HOUSE_TYPE": "House / apartment",
        "INCOME_TYPE": "Working",
        "EDUCATION_TYPE": "Secondary / secondary special",
        "selected_model": preferred_rf_name,
    }

    result = None
    form_data = defaults.copy()

    if request.method == "POST":
        form_data.update(request.form.to_dict())
        selected_model = form_data.get("selected_model", "")

        try:
            if selected_model not in models:
                raise ValueError(
                    "Selected model is not available. Please make sure the saved model files exist in the models folder."
                )

            input_row = {
                "NO_OF_CHILD": float(form_data["NO_OF_CHILD"]),
                "WORK_PHONE": int(float(form_data["WORK_PHONE"])),
                "PHONE": int(float(form_data["PHONE"])),
                "E_MAIL": int(float(form_data["E_MAIL"])),
                "FAMILY SIZE": float(form_data["FAMILY SIZE"]),
                "BEGIN_MONTH": float(form_data["BEGIN_MONTH"]),
                "AGE": float(form_data["AGE"]),
                "YEARS_EMPLOYED": float(form_data["YEARS_EMPLOYED"]),
                "INCOME": float(form_data["INCOME"]),
                "GENDER": form_data["GENDER"].strip(),
                "CAR": form_data["CAR"].strip(),
                "REALITY": form_data["REALITY"].strip(),
                "FAMILY_TYPE": form_data["FAMILY_TYPE"].strip(),
                "HOUSE_TYPE": form_data["HOUSE_TYPE"].strip(),
                "INCOME_TYPE": form_data["INCOME_TYPE"].strip(),
                "EDUCATION_TYPE": form_data["EDUCATION_TYPE"].strip(),
            }

            input_df = pd.DataFrame([input_row], columns=PREDICTION_COLUMNS)
            input_df = add_engineered_features_for_prediction(input_df)

            model_bundle = models[selected_model]
            pipeline = model_bundle["pipeline"]
            threshold = float(model_bundle["threshold"])

            if not hasattr(pipeline, "predict_proba"):
                raise ValueError("The selected model does not support probability prediction.")

            probability = float(pipeline.predict_proba(input_df)[0][1])
            prediction = int(probability >= threshold)

            label = "Fraudulent Transaction" if prediction == 1 else "Non-Fraudulent Transaction"

            interpretation = [
                "A predicted class of 1 means the application flags the record as potentially fraudulent under the selected decision threshold.",
                "A predicted class of 0 means the application does not flag the record as fraudulent under the selected decision threshold.",
                f"The probability threshold used for this model is {threshold:.2f}.",
                "Engineered features such as income per family member, contact-info count and large-family flag are calculated automatically before prediction.",
            ]

            if "Threshold tuned" in selected_model:
                interpretation.append(
                    "This version uses a validation-tuned threshold to improve the balance between recall and precision on an imbalanced fraud dataset."
                )
            else:
                interpretation.append(
                    "This version uses the default threshold of 0.50, which is useful as a baseline comparison."
                )

            if selected_model.startswith("Random Forest"):
                if "Threshold tuned" in selected_model:
                    interpretation.append(
                        "This was the best-performing Task 3 model based on F1-score and was selected as the recommended model."
                    )
                else:
                    interpretation.append(
                        "This is the Random Forest baseline version using the default decision threshold."
                    )
            elif selected_model.startswith("Logistic Regression"):
                if "Threshold tuned" in selected_model:
                    interpretation.append(
                        "This is the Logistic Regression version with validation-based threshold tuning applied."
                    )
                else:
                    interpretation.append(
                        "This is the Logistic Regression baseline using the default decision threshold."
                    )

            result = {
                "selected_model": selected_model,
                "predicted_class": prediction,
                "prediction_label": label,
                "fraud_probability": f"{probability:.4f}",
                "threshold": f"{threshold:.2f}",
                "interpretation": interpretation,
            }

        except Exception as e:
            result = {"error": str(e)}

    comparison_rows = model_comparison_df.to_dict(orient="records")

    return render_template(
        "predict.html",
        model_names=model_names,
        category_options=category_options,
        form_data=form_data,
        result=result,
        model_figures=model_figures,
        comparison_rows=comparison_rows,
        recommended_model_name=preferred_rf_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
