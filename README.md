# Job Screening Model Bias Analysis Project

This project focuses on analyzing bias and explaining decisions in job screening models, comparing Logistic Regression and XGBoost. The goal is to evaluate fairness metrics and interpret model predictions using SHAP values, with a focus on gender as the sensitive attribute.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results & Evaluation](#results--evaluation)

---

## Project Overview

This project investigates bias in hiring decisions using Logistic Regression and XGBoost models. It includes data preprocessing, fairness metric evaluation (e.g., Demographic Parity, Equal Opportunity), and explainability analysis using SHAP to interpret model predictions.


2. **Notebook Workflow:**
   - **Data Preprocessing:** Scales features using StandardScaler.
   - **Bias Analysis:** Evaluates fairness metrics (Demographic Parity, Equal Opportunity, False Positive Rate, Average Odds Difference).
   - **Model Training:** Trains Logistic Regression (from scratch) and loads pretrained XGBoost model.
   - **Explainability:** Uses SHAP to explain predictions for selected Hire/No-Hire cases.
   - **Evaluation:** Compares model performance and bias metrics.

---

## Modeling Approach

- **Feature Engineering:** Uses raw features, scaled with StandardScaler.
- **Class Imbalance Handling:** Accounts for imbalanced dataset (90% male) during training.
- **Models Used:**
  - Logistic Regression (`LogisticRegression`)
  - XGBoost (`XGBClassifier`, pretrained)
- **Fairness Metrics:** Demographic Parity, Equal Opportunity, False Positive Rate, Average Odds Difference.
- **Explainability:** SHAP values to interpret feature contributions for predictions.

---

## Results & Evaluation

- **Model Performance:** Both models show comparable bias (Average Odds Difference: 0.00). XGBoost has slightly higher disparities in Equal Opportunity (Male: 0.770, Female: 0.99) and Demographic Parity (Male: 0.318, Female: 0.287).
- **Explainability Results:**
  - **Logistic Regression:** High EducationLevel (SHAP: 1.574) and PersonalityScore (SHAP: 0.0088) drive Hire decisions; low PersonalityScore (SHAP: -0.0077) and InterviewScore (SHAP: -0.0077) influence No-Hire.
  - **XGBoost:** High PersonalityScore drives Hire decisions.
- **Visualizations:**
  - **Fairness Metrics Plot:** Visualizes Demographic Parity and Equal Opportunity for both models.
    ![Fairness Metrics Plot](images/fairness_metrics_plot.png)
  - **SHAP Explainability Plot:** Shows SHAP values for feature contributions in Hire/No-Hire predictions.
    ![SHAP Explainability Plot](images/shap_explainability_plot.png)

---


