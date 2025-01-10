import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.tree import plot_tree

def load_and_prepare_data(file_path, features, target):
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna(subset=features + [target])
    X = df_cleaned[features]
    y = df_cleaned[target]
    return train_test_split(X, y, test_size=0.6, random_state=42), df_cleaned

def linear_regression(X_train, X_test, y_train, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_lin)
    r2 = r2_score(y_test, y_pred_lin)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_lin)
    print()
    print(f"Linear Regression R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    plot_actual_vs_predicted(y_test, y_pred_lin, "Linear Regression: Actual vs Predicted")

def random_forest_regression(X_train, X_test, y_train, y_test, features):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    print()
    print(f"Random Forest R-squared: {r2_rf}")
    print(f"Mean Squared Error: {mse_rf}")
    print(f"Root Mean Squared Error: {rmse_rf}")
    print(f"Mean Absolute Error: {mae_rf}")
    plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest: Actual vs Predicted")
    plot_feature_importances(rf_model, features)
    plot_decision_tree(rf_model, features)

def logistic_regression(X_train, X_test, y_train, y_test):
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train, y_train)
    y_pred_log = log_reg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_log)
    conf_matrix = confusion_matrix(y_test, y_pred_log)
    class_report = classification_report(y_test, y_pred_log, zero_division=0)
    print()
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='green', label='Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Fit Line')
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, features):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Feature Importances from Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_decision_tree(model, features):
    tree = model.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()

def plot_correlation_matrix(df_cleaned, features):
    correlation_matrix = df_cleaned[features].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()

def plot_pca(X, df_cleaned):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Features')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=df_cleaned['department_group'], palette='deep', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Features Colored by Department Group')
    plt.legend(title='Department Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_mutual_information(X, y, features):
    mi = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({'Feature': features, 'Mutual Information': mi}).sort_values(by='Mutual Information', ascending=False)
    print()
    print(f"Mutual Information Scores: {mi_df}")

def main():
    file_path = 'cleaned_survey_data.csv'
    features = [
        'experience_years', 'ai_usage_rate', 'pa_frequency', 'confidence_influence',
        'stakeholder_communication_effectiveness', 'ai_confidence_increase',
        'ai_uncertainty_reduction', 'ai_forecast_accuracy',
        'pm_efficiency_ai', 'ai_risk_assessment', 'ai_data_visualization',
        'ai_tool_preference'
    ]
    target = 'stakeholder_trust'
    (X_train, X_test, y_train, y_test), df_cleaned = load_and_prepare_data(file_path, features, target)
    linear_regression(X_train, X_test, y_train, y_test)
    random_forest_regression(X_train, X_test, y_train, y_test, features)
    logistic_regression(X_train, X_test, y_train, y_test)
    plot_correlation_matrix(df_cleaned, features)
    calculate_mutual_information(X_train, y_train, features)
    plot_pca(X_train, df_cleaned)

if __name__ == "__main__":
    main()
