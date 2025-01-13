import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Load existing data from Excel
file_path = "*/Generated_Relational_Database.xlsx"
progress_data = pd.read_excel(file_path, sheet_name="Project Progress Data")
risk_logs = pd.read_excel(file_path, sheet_name="Risk Issue Logs")
performance_metrics = pd.read_excel(file_path, sheet_name="Performance Metrics")

# Print features of each table
def print_data_features(data, name):
    print(f"Features of {name}:")
    print(data.columns.tolist())
    print("\n")

print_data_features(progress_data, "Project Progress Data")
print_data_features(risk_logs, "Risk Issue Logs")
print_data_features(performance_metrics, "Performance Metrics")

def generate_unsupervised_analytics(progress_data, risk_logs, metrics, output_path):
    # Combine data for clustering
    combined_data = progress_data.merge(risk_logs, on='Milestone_ID', how='left').merge(metrics, on='Milestone_ID', how='left')

    # Select features for clustering
    features = ['Timeline_Weeks', 'KPI']  # Adjust features as needed
    X = combined_data[features].dropna()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust number of clusters as needed
    clusters_kmeans = kmeans.fit_predict(X_scaled)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(X_scaled)

    # Apply Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(random_state=42, contamination=0.1)  # Adjust contamination as needed
    anomalies = isolation_forest.fit_predict(X_scaled)

    # Linear Regression for Timeline_Weeks
    regression_features = ['KPI']  # Add more features if needed
    X_regression = combined_data[regression_features].dropna()
    y_regression = combined_data.loc[X_regression.index, 'Timeline_Weeks']

    X_train, X_test, y_train, y_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    timeline_predictions = lin_reg.predict(X_regression)

    # Create Risk_Happened column based on Risk_Probability
    risk_logs['Risk_Happened'] = (risk_logs['Risk_Probability'] > 0.50).astype(int)

    # Logistic Regression for Risk Prediction
    risk_features = ['Risk_Severity', 'Risk_Probability']
    X_risk = risk_logs[risk_features].dropna()
    y_risk = risk_logs['Risk_Happened']

    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_risk, y_train_risk)
    risk_predictions = log_reg.predict(X_risk)

    # Add cluster, anomaly, regression, and risk prediction labels to the original data
    combined_data['KMeans_Cluster'] = pd.Series(clusters_kmeans, index=X.index)
    combined_data['DBSCAN_Cluster'] = pd.Series(clusters_dbscan, index=X.index)
    combined_data['IsolationForest_Anomaly'] = pd.Series(anomalies, index=X.index)
    combined_data['Predicted_Timeline_Weeks'] = pd.Series(timeline_predictions, index=X_regression.index)
    combined_data['Risk_Prediction'] = pd.Series(risk_predictions, index=X_risk.index)

    # Save the labeled data to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        combined_data.to_excel(writer, sheet_name='Clustered_Data', index=False)

# Generate unsupervised analytics
output_path = "*/Clustered_Output.xlsx"
generate_unsupervised_analytics(progress_data, risk_logs, performance_metrics, output_path)

print(f"Clustered data saved to {output_path}")
