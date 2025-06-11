Cancer Prediction using Machine Learning

This project focuses on predicting the likelihood of cancer in patients based on various medical and lifestyle attributes using ensemble machine learning techniques.
Dataset

The dataset contains 1,500 patient records, each with the following features:

    Age

    Gender

    BMI

    Smoking

    Genetic Risk

    Physical Activity

    Alcohol Intake

    Cancer History

    Diagnosis (Target Label)

Models Used

The following machine learning models were trained and evaluated:

    Logistic Regression

    Random Forest Classifier

    XGBoost Classifier

AUC Scores (Individual Models)
Model	AUC Score
Logistic Regression	0.92
Random Forest	0.96
XGBoost	0.96
Ensemble Learning

To improve prediction accuracy, two ensemble techniques were implemented:

    Voting Classifier (Soft Voting)

    Stacked Classifier (using Logistic Regression as meta-learner)

Accuracy (Ensemble Models)
Ensemble Model	Accuracy
Voting Classifier	94.67%
Stacked Classifier	95.00%
ROC Curve

ROC Curve
How to Use the Trained Model

import joblib
import numpy as np

# Load the trained model
model = joblib.load("cancer_prediction_model.pkl")

# Sample input format: [Age, Gender, BMI, Smoking, Genetic Risk, Physical Activity, Alcohol Intake, Cancer History]
sample = np.array([[48, 1, 28.5, 0, 1, 6.0, 3.2, 0]])

# If a scaler was used during training, scale the input as well:
# sample = scaler.transform(sample)

# Make prediction
prediction = model.predict(sample)
print("Cancer Prediction:", prediction)

Trained Model

You can download the trained ensemble model:

File: cancer_prediction_model.pkl
Future Work

    Deploy as a web application using Flask, Streamlit, or FastAPI

    Integrate with real-time hospital data systems

    Add SHAP or LIME for model interpretability and explanation
