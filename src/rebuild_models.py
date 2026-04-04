from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_1_PATTERNS = ["*Credit*Sept_1*.csv", "*Credir*Sept_1*.csv"]
DATASET_2_PATTERNS = ["*Credit*Sept_2*.csv", "*Credir*Sept_2*.csv"]


def find_file(patterns):
    search_roots = [
        BASE_DIR,
        BASE_DIR / "data",
        Path.cwd(),
        Path.cwd() / "data",
    ]

    matches = []
    for root in search_roots:
        if root.exists():
            for pattern in patterns:
                matches.extend(root.glob(pattern))
                matches.extend(root.glob(f"**/{pattern}"))

    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No file found for patterns: {patterns}\n"
            f"Searched roots: {[str(p) for p in search_roots]}"
        )

    matches = sorted(set(matches), key=lambda p: (len(str(p)), str(p)))
    return matches[0]


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def add_engineered_features(df):
    df = df.copy()

    if {"INCOME", "FAMILY SIZE"}.issubset(df.columns):
        income = pd.to_numeric(df["INCOME"], errors="coerce")
        family_size = pd.to_numeric(df["FAMILY SIZE"], errors="coerce").replace(0, np.nan)
        df["INCOME_PER_FAMILY_MEMBER"] = income / family_size

    contact_cols = [col for col in ["PHONE", "WORK_PHONE", "E_MAIL"] if col in df.columns]
    if contact_cols:
        contact_data = df[contact_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["HAS_CONTACT_INFO"] = (contact_data > 0).sum(axis=1)

    if "FAMILY SIZE" in df.columns:
        family_size_raw = pd.to_numeric(df["FAMILY SIZE"], errors="coerce")
        df["IS_LARGE_FAMILY"] = (family_size_raw >= 4).astype(int)

    return df


print("BASE_DIR:", BASE_DIR)
print("CWD:", Path.cwd())

file1 = find_file(DATASET_1_PATTERNS)
file2 = find_file(DATASET_2_PATTERNS)

print("Dataset 1 found at:", file1)
print("Dataset 2 found at:", file2)

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

# Match the final notebook logic
model_data = add_engineered_features(data.copy())

drop_cols = ["TARGET", "ID", "FLAG_MOBIL"]
X = model_data.drop(columns=drop_cols, errors="ignore")
y = model_data["TARGET"].astype(int)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

lr_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

lr_model = Pipeline([
    ("preprocessor", lr_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

rf_model = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Fitting training models...")
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# RandomizedSearchCV-tuned Random Forest
rf_param_distributions = {
    "classifier__n_estimators": [200, 300, 400, 500],
    "classifier__max_depth": [None, 10, 15, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": ["balanced", "balanced_subsample"]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_param_distributions,
    n_iter=12,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_random_search.fit(X_train, y_train)

rf_tuned_model = rf_random_search.best_estimator_
rf_tuned_best_params = rf_random_search.best_params_

lr_val_prob = lr_model.predict_proba(X_val)[:, 1]
rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
rf_tuned_val_prob = rf_tuned_model.predict_proba(X_val)[:, 1]

lr_best_threshold, lr_best_f1 = find_best_threshold(y_val, lr_val_prob)
rf_best_threshold, rf_best_f1 = find_best_threshold(y_val, rf_val_prob)
rf_tuned_best_threshold, rf_tuned_best_f1 = find_best_threshold(y_val, rf_tuned_val_prob)

print(f"Best LR threshold: {lr_best_threshold:.2f} | Validation F1: {lr_best_f1:.6f}")
print(f"Best RF threshold: {rf_best_threshold:.2f} | Validation F1: {rf_best_f1:.6f}")
print(f"Best RF (RandomizedSearchCV) threshold: {rf_tuned_best_threshold:.2f} | Validation F1: {rf_tuned_best_f1:.6f}")
print("Best RandomizedSearchCV parameters:", rf_tuned_best_params)

# Final deployment RF model retrained on full development data
final_rf_deployment_model = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

final_rf_deployment_model.fit(X_train_full, y_train_full)

# Final deployment LR model retrained on full development data
final_lr_deployment_model = Pipeline([
    ("preprocessor", lr_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42
    ))
])

final_lr_deployment_model.fit(X_train_full, y_train_full)

# Final deployment RandomizedSearchCV RF model retrained on full development data
rf_tuned_classifier_params = {
    key.replace("classifier__", ""): value
    for key, value in rf_tuned_best_params.items()
    if key.startswith("classifier__")
}

final_rf_random_search_model = Pipeline([
    ("preprocessor", rf_preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **rf_tuned_classifier_params
    ))
])

final_rf_random_search_model.fit(X_train_full, y_train_full)

# Save generic final-selected model files for GUI compatibility
joblib.dump(final_rf_deployment_model, MODELS_DIR / "final_selected_model_pipeline.pkl")
joblib.dump(rf_best_threshold, MODELS_DIR / "final_selected_threshold.pkl")

# Save base RF files expected by GUI
joblib.dump(final_rf_deployment_model, MODELS_DIR / "random_forest_balanced_pipeline.pkl")
joblib.dump(rf_best_threshold, MODELS_DIR / "random_forest_best_threshold.pkl")

# Save RandomizedSearchCV RF files for GUI
joblib.dump(final_rf_random_search_model, MODELS_DIR / "random_forest_randomized_search_pipeline.pkl")
joblib.dump(rf_tuned_best_threshold, MODELS_DIR / "random_forest_randomized_search_best_threshold.pkl")

# Save LR files expected by GUI
joblib.dump(final_lr_deployment_model, MODELS_DIR / "logistic_regression_pipeline.pkl")
joblib.dump(lr_best_threshold, MODELS_DIR / "logistic_regression_best_threshold.pkl")

print("\nModel files rebuilt successfully.")
print("Saved to:", MODELS_DIR)
print("Files created:")
for p in sorted(MODELS_DIR.glob("*.pkl")):
    print("-", p.name)