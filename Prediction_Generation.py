import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# Load the dataset
file_path = 'C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/cleaned_survey_data.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering
features = ['confidence_influence', 'ai_confidence_increase']  # Ensure these columns exist
X = data[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Affinity Propagation clustering
affinity_propagation = AffinityPropagation(random_state=42)
clusters_affinity = affinity_propagation.fit_predict(X_scaled)

# Add cluster labels to the data
data['AI_Adopter_Cluster'] = pd.Series(clusters_affinity, index=X.index)

# Perform T-test between clusters
def perform_t_test(data, cluster_col, value_col, cluster1, cluster2):
    cluster1_data = data[data[cluster_col] == cluster1][value_col]
    cluster2_data = data[data[cluster_col] == cluster2][value_col]
    t_stat, p_value = ttest_ind(cluster1_data, cluster2_data, nan_policy='omit')
    print(f"T-test between Cluster {cluster1} and Cluster {cluster2} for {value_col}: T-statistic = {t_stat}, P-value = {p_value}")

unique_clusters = data['AI_Adopter_Cluster'].unique()
if len(unique_clusters) >= 2:
    perform_t_test(data, 'AI_Adopter_Cluster', 'numeric_column', unique_clusters[0], unique_clusters[1])
else:
    print("Not enough clusters formed for T-test.")
