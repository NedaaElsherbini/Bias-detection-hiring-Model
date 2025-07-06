# Main execution
def main():
    data = load_data()
    model, scaler, X_train_scaled, y_train, X_test_scaled, y_test, test_gender, X_test = train_classifier(data)
    
    # Model performance
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Fairness analysis
    metric_frame, avg_odds_diff = calculate_fairness_metrics(y_test, y_pred, test_gender)
    plot_disparities(metric_frame)
    
    # Explainability
    explanations = explain_predictions(model, X_test_scaled, X_test, test_gender)
    
    # Bias mitigation
    baseline_metrics, mitigated_metrics, baseline_acc, mitigated_acc, baseline_aod, mitigated_aod = mitigate_bias(data, model, scaler)
    
    # Save results
    with open('results.txt', 'w') as f:
        f.write(f"Model Performance:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\n")
        f.write(f"Fairness Metrics:\n{metric_frame.by_group}\nAverage Odds Difference: {avg_odds_diff:.2f}\n")
        f.write("Explanations:\n")
        for exp in explanations:
            f.write(f"Prediction {exp['index']}: {exp['prediction']} (Gender: {exp['gender']})\n")
            f.write(f"SHAP Values: {exp['shap_values']}\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.2f}\nMitigated Accuracy: {mitigated_acc:.2f}\n")
        f.write(f"Baseline AOD: {baseline_aod:.2f}\nMitigated AOD: {mitigated_aod:.2f}\n")
    
    files.download('results.txt')  # Download results

if __name__ == "__main__":
    main()