import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import shap
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from google.colab import files


def load_data(file_path='data.csv'):
    data = pd.read_csv(file_path)
    data['Gender'] = data['Gender'].map({0: 'Female', 1: 'Male'})
    return data

def train_xgboost_classifier(data, save_path='xgboost_model.json'):
    X = data[['Age', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies', 
              'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']]
    y = data['HiringDecision']
    
    male_data = data[data['Gender'] == 'Male'].sample(frac=0.8, random_state=42)
    female_data = data[data['Gender'] == 'Female'].sample(frac=0.4, random_state=42)
    train_data = pd.concat([male_data, female_data])
    X_train = train_data[X.columns]
    y_train = train_data['HiringDecision']
    
    X_test = data.loc[~data.index.isin(train_data.index), X.columns]
    y_test = data.loc[~data.index.isin(train_data.index), 'HiringDecision']
    test_gender = data.loc[~data.index.isin(train_data.index), 'Gender']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    
    model.save_model(save_path)
    
    return model, scaler, X_train_scaled, y_train, X_test_scaled, y_test, test_gender, X_test

def load_xgboost_classifier(save_path='xgboost_model.json'):
    model = xgb.XGBClassifier()
    model.load_model(save_path)
    return model

def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    metrics = {
        'demographic_parity': selection_rate,
        'equal_opportunity': true_positive_rate,
        'false_positive_rate': false_positive_rate
    }
    metric_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    
    fpr_diff = metric_frame.by_group['false_positive_rate'].diff().abs().mean()
    tpr_diff = metric_frame.by_group['equal_opportunity'].diff().abs().mean()
    avg_odds_diff = (fpr_diff + tpr_diff) / 2
    
    return metric_frame, avg_odds_diff

def plot_disparities(metric_frame, filename='xgb_fairness_plot.png'):
    plt.figure(figsize=(10, 6))
    metric_frame.by_group.plot(kind='bar')
    plt.title('Fairness Metrics by Gender (XGBoost)')
    plt.ylabel('Metric Value')
    plt.savefig(filename)
    plt.close()
    files.download(filename)

def explain_predictions(model, X_test_scaled, X_test, test_gender):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    y_pred = model.predict(X_test_scaled)
    hire_indices = np.where(y_pred == 1)[0][:3]
    no_hire_indices = np.where(y_pred == 0)[0][:2]
    
    explanations = []
    for idx in np.concatenate([hire_indices, no_hire_indices]):
        explanation = {
            'index': idx,
            'prediction': 'Hire' if y_pred[idx] == 1 else 'No-Hire',
            'gender': test_gender.iloc[idx],
            'shap_values': dict(zip(X_test.columns, shap_values[idx]))
        }
        explanations.append(explanation)
    
    return explanations

def mitigate_bias(data, model, scaler):
    X = data[['Age', 'EducationLevel', 'ExperienceYears', 'PreviousCompanies', 
              'DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore', 'RecruitmentStrategy']]
    y = data['HiringDecision']
    sensitive_features = data['Gender']
    
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive_features, test_size=0.2, random_state=42)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    baseline_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    baseline_model.fit(X_train_scaled, y_train)
    baseline_pred = baseline_model.predict(X_test_scaled)
    
    exp_grad = ExponentiatedGradient(
        xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        constraints=DemographicParity(),
        max_iter=50
    )
    exp_grad.fit(X_train_scaled, y_train, sensitive_features=sensitive_train)
    mitigated_pred = exp_grad.predict(X_test_scaled)
    
    baseline_metrics, baseline_aod = calculate_fairness_metrics(y_test, baseline_pred, sensitive_test)
    mitigated_metrics, mitigated_aod = calculate_fairness_metrics(y_test, mitigated_pred, sensitive_test)
    
    baseline_acc = accuracy_score(y_test, baseline_pred)
    mitigated_acc = accuracy_score(y_test, mitigated_pred)
    
    return baseline_metrics, mitigated_metrics, baseline_acc, mitigated_acc, baseline_aod, mitigated_aod