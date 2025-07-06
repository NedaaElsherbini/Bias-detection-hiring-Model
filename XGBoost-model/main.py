def main():
  
    data = load_data()
    xgb_model, xgb_scaler, X_train_scaled, y_train, X_test_scaled, y_test, test_gender, X_test = train_xgboost_classifier(data, save_path='xgboost_model.json')
    
    pretrained_xgb_model = load_xgboost_classifier('xgboost_model.json')
    xgb_pred = pretrained_xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    
    xgb_metric_frame, xgb_aod = calculate_fairness_metrics(y_test, xgb_pred, test_gender)
    plot_disparities(xgb_metric_frame, 'xgb_fairness_plot.png')
    
    xgb_explanations = explain_predictions(pretrained_xgb_model, X_test_scaled, X_test, test_gender)
    
    xgb_baseline_metrics, xgb_mitigated_metrics, xgb_baseline_acc, xgb_mitigated_acc, xgb_baseline_aod, xgb_mitigated_aod = mitigate_bias(data, pretrained_xgb_model, xgb_scaler)
    
    with open('results.txt', 'w') as f:
        f.write("XGBoost (Pretrained) Results:\n")
        f.write(f"Accuracy: {xgb_accuracy:.2f}\nPrecision: {xgb_precision:.2f}\nRecall: {xgb_recall:.2f}\n")
        f.write(f"Fairness Metrics:\n{xgb_metric_frame.by_group}\nAverage Odds Difference: {xgb_aod:.2f}\n")
        f.write("Explanations:\n")
        for exp in xgb_explanations:
            f.write(f"Prediction {exp['index']}: {exp['prediction']} (Gender: {exp['gender']})\n")
            f.write(f"SHAP Values: {exp['shap_values']}\n")
        f.write(f"Baseline Accuracy: {xgb_baseline_acc:.2f}\nMitigated Accuracy: {xgb_mitigated_acc:.2f}\n")
        f.write(f"Baseline AOD: {xgb_baseline_aod:.2f}\nMitigated AOD: {xgb_mitigated_aod:.2f}\n")
    
    files.download('results.txt')
    files.download('xgb_fairness_plot.png')
    files.download('xgboost_model.json')

if __name__ == "__main__":
    main()