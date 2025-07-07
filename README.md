# Job Screening Model Bias Project

This project focuses on analyzing bias and explaining decisions in job screening models, comparing Logistic Regression and XGBoost. The goal is to evaluate fairness metrics and interpret model predictions using SHAP values, with a focus on gender as the sensitive attribute.


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Models](#running-the-models)
  - [Codebase Structure](#codebase-structure)
- [Results & Evaluation](#results--evaluation)
- [Improvements](#improvements)

## Overview
This project evaluates bias in job screening models using logistic regression and XGBoost on a dataset of 500 applicant records. The dataset includes features such as Age, Gender (0: Female, 1: Male), EducationLevel (1-4), ExperienceYears, PreviousCompanies, DistanceFromCompany, InterviewScore, SkillsScore, PersonalityScore, RecruitmentStrategy (1-3), and HiringDecision (0: No-Hire, 1: Hire). Gender is the sensitive attribute, and the dataset is imbalanced (80% male, 40% female resumes).

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/model-bias-analysis.git
   cd model-bias-analysis
   ```

2. **Set Up Environment**
   - Install Python 3.8+.
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install scikit-learn xgboost pandas numpy
     ```

3. **Data Preparation**
   - Place the dataset (e.g., `hiring_data.csv`) in the `data/` directory.
   - Ensure features are preprocessed (scaled using `StandardScaler`).

## Usage

### Running the Models
1. **Logistic Regression**
   - Execute the script to train and evaluate:
     ```bash
     python scripts/logistic_regression.py
     ```
   - Output: Accuracy (0.86), Precision (0.83), Recall (0.71), Fairness metrics (Demographic Parity: Male 0.304, Female 0.269; Equal Opportunity: Male 0.761, Female 0.668; False Positive Rate: Male 0.098, Female 0.062; AOD: 0.05).

2. **XGBoost**
   - Load the pretrained model and evaluate:
     ```bash
     python scripts/xgboost.py
     ```
   - Output: Accuracy (0.92), Precision (0.91), Recall (0.83), Fairness metrics (Demographic Parity: Male 0.318, Female 0.287; Equal Opportunity: Male 0.570, Female 0.519; False Positive Rate: Male 0.000, Female 0.029; AOD: 0.05).

### Codebase Structure
- `data/`: Contains the dataset (e.g., `hiring_data.csv`).
- `scripts/`: 
  - `logistic_regression.py`: Implements and trains logistic regression.
  - `xgboost.py`: Loads pretrained XGBoost model (`xgboost_model.json`) and evaluates.
- `notebooks/`: Jupyter notebooks for exploratory analysis (if applicable).

- **Model Performance:** Both models show comparable bias (Average Odds Difference: 0.00). XGBoost has slightly higher disparities in Equal Opportunity (Male: 0.770, Female: 0.99) and Demographic Parity (Male: 0.318, Female: 0.287).
- **Explainability Results:**
  - **Logistic Regression:** High EducationLevel (SHAP: 1.574) and PersonalityScore (SHAP: 0.0088) drive Hire decisions; low PersonalityScore (SHAP: -0.0077) and InterviewScore (SHAP: -0.0077) influence No-Hire.
  - **XGBoost:** High PersonalityScore drives Hire decisions.
- **Visualizations:**
  - **Fairness Metrics Plot:** Visualizes Demographic Parity and Equal Opportunity for both models.
    ![Fairness Metrics Plot](file:///C:/Users/dell/OneDrive/Documents/AI%20Bias%20Detection/results/logistic_fairness_plot.png)
  - **SHAP Explainability Plot:** Shows SHAP values for feature contributions in Hire/No-Hire predictions.
    ![SHAP Explainability Plot](images/shap_explainability_plot.png)

## Improvements
- Rebalance the dataset (e.g., oversample female resumes) to reduce bias.
- Apply fairness constraints during training.
- Analyze feature importance in XGBoost to identify bias sources.



---


