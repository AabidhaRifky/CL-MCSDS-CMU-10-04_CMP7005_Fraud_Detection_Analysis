Credit Card Fraud Detection Analysis \& Web Application

Module: Programming for Data Analysis (CMP7005)

Project ID: CL-MCSDS-CMU-10-04

Author: Aabidha



Project Overview:

This project presents a complete end-to-end fraud detection system combining data analysis, machine learning, and a Flask web application.



Key Features:

\- Data merging and validation

\- Exploratory Data Analysis (EDA)

\- Data preprocessing and feature engineering

\- Multiple ML models (LR, DT, RF, GB)

\- Hyperparameter tuning

\- Threshold tuning for imbalanced data

\- Interactive Flask web app



Project Structure:

\- notebook/: data, figures, models, tables, reports

\- src/: Flask app, templates, static files

\- requirements.txt

\- README.txt



Dataset:

\- Two CSV files merged using ID and User

\- Final dataset: 25,134 rows, binary TARGET variable



Class Imbalance:

\- Fraud: 1.68%

\- Non-Fraud: 98.32%



Models:

\- Logistic Regression

\- Decision Tree

\- Random Forest

\- Gradient Boosting

\- Final Tuned Model (Best)



Best Model:

Final Tuned Random Forest:

\- Best F1 Score

\- Best balance of precision \& recall

\- Highest ROC-AUC



Threshold:

Optimized threshold = 0.45



Run App:

pip install -r requirements.txt

python src/app.py



Open:

http://127.0.0.1:5000/



Pages:

\- Home

\- Predict

\- EDA



Outputs:

\- Prediction (Fraud / Not Fraud)

\- Probability

\- Risk Level

\- Model Metrics



Conclusion:

This project demonstrates a complete ML pipeline with deployment and UI, handling real-world class imbalance problems effectively.



