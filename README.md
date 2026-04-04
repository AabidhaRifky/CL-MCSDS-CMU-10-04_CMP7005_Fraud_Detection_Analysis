# CL-MCSDS-CMU-10-04_CMP7005_Fraud_Detection_Analysis

## Credit Card Fraud Detection Analysis and Web Application  
**Programming for Data Analysis - CMP7005**

This project contains a complete fraud detection submission for CMP7005, including:

- a **Jupyter notebook** for Tasks 1 to 3  
- a **Flask-based multipage web application** for Task 4  
- saved **models, tables, figures, and logs** generated from the final notebook  
- supporting datasets and project files

The project analyses merged credit card customer data, performs exploratory data analysis, builds fraud detection models, compares performance on an imbalanced dataset, and provides an interactive web interface for prediction.

---

## Project contents

### Main components
- **Notebook analysis** for:
  - Task 1: Data handling
  - Task 2: Data understanding, preprocessing, statistics, and visualisation
  - Task 3: Model building, comparison, and improvement

- **Web application** for:
  - Data Overview
  - EDA
  - Modelling & Prediction

---

## Folder structure

- `data/` — input CSV datasets  
- `models/` — saved machine learning pipelines and thresholds  
- `figures/` — exported EDA and modelling figures  
- `tables/` — exported tables from the notebook  
- `logs/` — saved logs and text-based outputs  
- `reports/` — summary report outputs  
- `src/app.py` — Flask web application backend  
- `src/templates/` — HTML templates for the GUI  
- `src/static/style.css` — web application styling  
- `.ipynb` notebook — final analysis notebook

---

## Datasets used

The project uses two CSV files:

- `Credit_Card_Dataset_2025_Sept_1.csv`
- `Credit_Card_Dataset_2025_Sept_2.csv`

These datasets are merged using the shared identifiers **ID** and **User**.  
After removing redundant merge columns and non-analytical artefacts, the final analytical dataset contains:

- **25,134 rows**
- **19 analytical columns**
- **binary fraud target (`TARGET`)**

---

## Final modelling summary

Two classification models were developed and compared:

- **Logistic Regression**
- **Random Forest**

Because the dataset is highly imbalanced, model selection was based mainly on:

- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Average Precision**

The final selected model was:

- **Random Forest – Threshold tuned on validation (0.12)**

This model achieved the strongest final **F1-score** among the evaluated model variants and was used as the recommended model in the web application.

---

## Engineered features

The final notebook and GUI use a small set of interpretable engineered features:

- `INCOME_PER_FAMILY_MEMBER`
- `HAS_CONTACT_INFO`
- `IS_LARGE_FAMILY`

These are calculated to improve the representation of household burden and contactability while keeping the model explainable.

In the web application, these engineered features are calculated automatically from the raw user input before prediction.

---

## Install dependencies

Create and activate your virtual environment, then run:

pip install -r requirements.txt

## Run app.py
Run python src/app.py and then open the local Flask address shown in the terminal, usually:

http://127.0.0.1:5000/