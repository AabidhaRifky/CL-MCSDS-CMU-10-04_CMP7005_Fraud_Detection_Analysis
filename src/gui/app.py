import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

DATASET_1_PATTERNS = ["*Credit*Sept_1*.csv", "*Credir*Sept_1*.csv"]
DATASET_2_PATTERNS = ["*Credit*Sept_2*.csv", "*Credir*Sept_2*.csv"]


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
            "Dataset files were not found inside the data folder.\n"
            "Please place both Sept_1 and Sept_2 CSV files inside the data folder."
        )

    data_1 = pd.read_csv(file1)
    data_2 = pd.read_csv(file2)
    data = pd.merge(data_1, data_2, left_on="ID", right_on="User", how="outer")
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

    candidate_numeric_cols = [
        col for col in ["NO_OF_CHILD", "FAMILY SIZE", "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED", "INCOME"]
        if col in data_clean.columns
    ]

    for col in candidate_numeric_cols:
        q1 = data_clean[col].quantile(0.25)
        q3 = data_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_clean[col] = data_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return data_clean


def prepare_model_features(data: pd.DataFrame):
    data_model = pd.get_dummies(data, drop_first=True)

    drop_candidate_cols = ["TARGET", "ID", "User", "Unnamed: 0"]
    X = data_model.drop(columns=[c for c in drop_candidate_cols if c in data_model.columns], errors="ignore")
    return X


class FraudApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("CMP7005 PRAC1 - Credit Card Fraud Detection GUI")
        self.geometry("1200x760")
        self.minsize(1100, 700)

        self.raw_data = None
        self.clean_data = None
        self.file1_name = ""
        self.file2_name = ""

        self.scaler = None
        self.models = {}

        self._load_assets()
        self._build_ui()

    def _load_assets(self):
        try:
            self.raw_data, self.file1_name, self.file2_name = load_and_merge_data()
            self.clean_data = preprocess_for_display(self.raw_data)

            scaler_path = MODELS_DIR / "standard_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            model_files = {
                "Logistic Regression Baseline": "logistic_regression_baseline.pkl",
                "Logistic Regression Weighted": "logistic_regression_weighted.pkl",
                "Decision Tree Baseline": "decision_tree_baseline.pkl",
                "Decision Tree Weighted": "decision_tree_weighted.pkl",
                "Decision Tree Tuned": "decision_tree_tuned.pkl",
            }

            for display_name, filename in model_files.items():
                model_path = MODELS_DIR / filename
                if model_path.exists():
                    self.models[display_name] = joblib.load(model_path)

        except Exception as e:
            messagebox.showerror("Startup Error", str(e))

    def _build_ui(self):
        title = tk.Label(
            self,
            text="Credit Card Fraud Detection System",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=12)

        subtitle = tk.Label(
            self,
            text="Programming for Data Analysis - CMP7005",
            font=("Arial", 11)
        )
        subtitle.pack()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        self.page_overview = ttk.Frame(notebook)
        self.page_eda = ttk.Frame(notebook)
        self.page_model = ttk.Frame(notebook)

        notebook.add(self.page_overview, text="Data Overview")
        notebook.add(self.page_eda, text="EDA")
        notebook.add(self.page_model, text="Modeling & Prediction")

        self._build_overview_page()
        self._build_eda_page()
        self._build_model_page()

    def _build_overview_page(self):
        frame = self.page_overview

        summary_frame = ttk.LabelFrame(frame, text="Dataset Summary", padding=12)
        summary_frame.pack(fill="x", padx=10, pady=10)

        if self.raw_data is not None:
            rows, cols = self.raw_data.shape
            fraud_count = int(self.raw_data["TARGET"].sum()) if "TARGET" in self.raw_data.columns else 0
            nonfraud_count = rows - fraud_count
            missing_count = int(self.raw_data.isnull().sum().sum())

            summary_text = (
                f"Dataset 1: {self.file1_name}\n"
                f"Dataset 2: {self.file2_name}\n\n"
                f"Merged Rows: {rows}\n"
                f"Merged Columns: {cols}\n"
                f"Fraudulent Transactions (Target=1): {fraud_count}\n"
                f"Non-Fraudulent Transactions (Target=0): {nonfraud_count}\n"
                f"Total Missing Values Before Treatment: {missing_count}"
            )
        else:
            summary_text = "Dataset could not be loaded."

        lbl = tk.Label(summary_frame, text=summary_text, justify="left", anchor="w", font=("Arial", 11))
        lbl.pack(fill="x")

        table_frame = ttk.LabelFrame(frame, text="Preview of Merged Dataset", padding=8)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = [col for col in self.raw_data.columns if col != "Unnamed: 0"][:10] if self.raw_data is not None else []
        self.overview_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)

        for col in columns:
            self.overview_tree.heading(col, text=col)
            self.overview_tree.column(col, width=120, anchor="center")

        if self.raw_data is not None:
            preview = self.raw_data[columns].head(15).reset_index(drop=True)
            for _, row in preview.iterrows():
                self.overview_tree.insert("", "end", values=list(row))

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.overview_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.overview_tree.xview)
        self.overview_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.overview_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

    def _build_eda_page(self):
        frame = self.page_eda

        top_frame = ttk.LabelFrame(frame, text="EDA Summary", padding=12)
        top_frame.pack(fill="x", padx=10, pady=10)

        if self.clean_data is not None:
            numeric_cols = self.clean_data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.clean_data.select_dtypes(exclude=[np.number]).columns.tolist()

            eda_text = (
                f"Numerical Columns: {len(numeric_cols)}\n"
                f"Categorical Columns: {len(categorical_cols)}\n"
                f"Main Target Column: TARGET\n"
                f"Class imbalance is severe, therefore precision, recall and F1-score are more useful than accuracy."
            )
        else:
            eda_text = "EDA data is not available."

        tk.Label(top_frame, text=eda_text, justify="left", anchor="w", font=("Arial", 11)).pack(fill="x")

        middle_frame = ttk.LabelFrame(frame, text="Saved EDA Figures", padding=10)
        middle_frame.pack(fill="both", expand=True, padx=10, pady=10)

        figure_files = sorted([f.name for f in FIGURES_DIR.glob("*.png")])

        figure_files = sorted([f.name for f in FIGURES_DIR.glob("*.png")])

        if figure_files:
            text = f"Total saved figures found: {len(figure_files)}\n\n" + "\n".join(figure_files)
        else:
            text = "No saved figures found yet."

        self.eda_textbox = tk.Text(middle_frame, wrap="word", height=20)
        self.eda_textbox.insert("1.0", text)
        self.eda_textbox.config(state="disabled")
        self.eda_textbox.pack(fill="both", expand=True)

        bottom_frame = ttk.LabelFrame(frame, text="Key EDA Findings", padding=10)
        bottom_frame.pack(fill="x", padx=10, pady=10)

        findings = (
            "- The dataset is highly imbalanced, with fraud cases forming a very small minority.\n"
            "- Some missing values were identified in INCOME_TYPE, YEARS_EMPLOYED, and FAMILY SIZE.\n"
            "- Outliers were analyzed using the IQR method and capped conservatively.\n"
            "- Important numerical drivers include BEGIN_MONTH, AGE, YEARS_EMPLOYED, and INCOME.\n"
            "- Identifier columns such as ID and User were excluded from modeling."
        )

        tk.Label(bottom_frame, text=findings, justify="left", anchor="w", font=("Arial", 10)).pack(fill="x")

    def _build_model_page(self):
        frame = self.page_model

        info_frame = ttk.LabelFrame(frame, text="Model Selection", padding=12)
        info_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(info_frame, text="Choose a model and enter customer details for a fraud prediction.", font=("Arial", 11)).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 10)
        )

        tk.Label(info_frame, text="Model:").grid(row=1, column=0, sticky="w", padx=4, pady=4)

        self.model_var = tk.StringVar()
        model_names = list(self.models.keys()) if self.models else ["No model found"]
        self.model_combo = ttk.Combobox(info_frame, textvariable=self.model_var, values=model_names, state="readonly", width=35)
        self.model_combo.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        if model_names:
            self.model_combo.current(0)

        form_frame = ttk.LabelFrame(frame, text="Prediction Input Fields", padding=12)
        form_frame.pack(fill="x", padx=10, pady=10)

        self.entries = {}

        fields = [
            ("NO_OF_CHILD", "0"),
            ("FAMILY SIZE", "2"),
            ("BEGIN_MONTH", "24"),
            ("AGE", "40"),
            ("YEARS_EMPLOYED", "5"),
            ("INCOME", "180000"),
            ("GENDER", "F"),
            ("CAR", "N"),
            ("REALITY", "Y"),
            ("FAMILY_TYPE", "Married"),
            ("HOUSE_TYPE", "House / apartment"),
            ("INCOME_TYPE", "Working"),
            ("EDUCATION_TYPE", "Secondary / secondary special"),
            ("WORK_PHONE", "0"),
            ("PHONE", "0"),
            ("E_MAIL", "0"),
            ("FLAG_MOBIL", "1"),
        ]

        for idx, (label_text, default_val) in enumerate(fields):
            r = idx // 2
            c = (idx % 2) * 2

            tk.Label(form_frame, text=label_text + ":").grid(row=r, column=c, sticky="w", padx=5, pady=5)
            entry = ttk.Entry(form_frame, width=30)
            entry.insert(0, default_val)
            entry.grid(row=r, column=c + 1, sticky="w", padx=5, pady=5)
            self.entries[label_text] = entry

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", padx=10, pady=10)

        ttk.Button(button_frame, text="Predict", command=self.predict_transaction).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_inputs).pack(side="left", padx=5)

        result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding=12)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.result_text = tk.Text(result_frame, height=12, wrap="word")
        self.result_text.pack(fill="both", expand=True)

    def clear_inputs(self):
        defaults = {
            "NO_OF_CHILD": "0",
            "FAMILY SIZE": "2",
            "BEGIN_MONTH": "24",
            "AGE": "40",
            "YEARS_EMPLOYED": "5",
            "INCOME": "180000",
            "GENDER": "F",
            "CAR": "N",
            "REALITY": "Y",
            "FAMILY_TYPE": "Married",
            "HOUSE_TYPE": "House / apartment",
            "INCOME_TYPE": "Working",
            "EDUCATION_TYPE": "Secondary / secondary special",
            "WORK_PHONE": "0",
            "PHONE": "0",
            "E_MAIL": "0",
            "FLAG_MOBIL": "1",
        }

        for key, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, defaults[key])

        self.result_text.delete("1.0", tk.END)

    def predict_transaction(self):
        try:
            selected_model = self.model_var.get()
            if selected_model not in self.models:
                messagebox.showerror("Model Error", "Selected model is not available.")
                return

            input_row = {
                "NO_OF_CHILD": float(self.entries["NO_OF_CHILD"].get()),
                "FAMILY SIZE": float(self.entries["FAMILY SIZE"].get()),
                "BEGIN_MONTH": float(self.entries["BEGIN_MONTH"].get()),
                "AGE": float(self.entries["AGE"].get()),
                "YEARS_EMPLOYED": float(self.entries["YEARS_EMPLOYED"].get()),
                "INCOME": float(self.entries["INCOME"].get()),
                "GENDER": self.entries["GENDER"].get().strip(),
                "CAR": self.entries["CAR"].get().strip(),
                "REALITY": self.entries["REALITY"].get().strip(),
                "FAMILY_TYPE": self.entries["FAMILY_TYPE"].get().strip(),
                "HOUSE_TYPE": self.entries["HOUSE_TYPE"].get().strip(),
                "INCOME_TYPE": self.entries["INCOME_TYPE"].get().strip(),
                "EDUCATION_TYPE": self.entries["EDUCATION_TYPE"].get().strip(),
                "WORK_PHONE": int(float(self.entries["WORK_PHONE"].get())),
                "PHONE": int(float(self.entries["PHONE"].get())),
                "E_MAIL": int(float(self.entries["E_MAIL"].get())),
                "FLAG_MOBIL": int(float(self.entries["FLAG_MOBIL"].get())),
            }

            temp_df = pd.DataFrame([input_row])

            # add dropped/unused columns with placeholders so preprocessing structure matches
            temp_df["ID"] = 0
            temp_df["User"] = 0
            temp_df["Unnamed: 0"] = 0
            temp_df["TARGET"] = 0

            combined = pd.concat([self.clean_data.copy(), temp_df], ignore_index=True)
            combined_encoded = pd.get_dummies(combined, drop_first=True)

            drop_candidate_cols = ["TARGET", "ID", "User", "Unnamed: 0"]
            X_all = combined_encoded.drop(columns=[c for c in drop_candidate_cols if c in combined_encoded.columns], errors="ignore")

            input_features = X_all.tail(1)

            model = self.models[selected_model]

            if "Logistic Regression" in selected_model:
                if self.scaler is None:
                    messagebox.showerror("Scaler Error", "Standard scaler file was not found.")
                    return
                input_processed = self.scaler.transform(input_features)
            else:
                input_processed = input_features

            prediction = model.predict(input_processed)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_processed)[0][1]
            else:
                probability = None

            label = "Fraudulent Transaction" if prediction == 1 else "Non-Fraudulent Transaction"

            explanation = (
                f"Selected Model: {selected_model}\n"
                f"Predicted Class: {int(prediction)}\n"
                f"Prediction Label: {label}\n"
            )

            if probability is not None:
                explanation += f"Fraud Probability: {probability:.4f}\n"

            explanation += (
                "\nInterpretation:\n"
                "- A predicted class of 1 means the transaction is likely fraudulent.\n"
                "- A predicted class of 0 means the transaction is likely non-fraudulent.\n"
                "- For this coursework, Decision Tree Weighted was the best-performing model by F1-score.\n"
            )

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", explanation)

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


if __name__ == "__main__":
    app = FraudApp()
    app.mainloop()