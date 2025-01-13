import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Load and preprocess the dataset
file_path = 'C:/Users/x/OneDrive - Berlin School of Business and Innovation (BSBI)/Desktop/BSBI/Dissertation/Data-Cleaning/cleaned_survey_data.csv'
data = pd.read_csv(file_path)
features = ['confidence_influence', 'ai_confidence_increase']
X = data[features].dropna()
X_scaled = StandardScaler().fit_transform(X)

# Apply Affinity Propagation clustering with adjusted preference
preference_value = -50  # Adjust this value to influence the number of clusters
affinity_propagation = AffinityPropagation(preference=preference_value, random_state=42)
clusters_affinity = affinity_propagation.fit_predict(X_scaled)

# Check the number of clusters formed
num_clusters = len(set(clusters_affinity))
print(f"Number of clusters formed: {num_clusters}")

# Evaluate Affinity Propagation clustering
silhouette_affinity = silhouette_score(X_scaled, clusters_affinity)
davies_bouldin_affinity = davies_bouldin_score(X_scaled, clusters_affinity)
calinski_harabasz_affinity = calinski_harabasz_score(X_scaled, clusters_affinity)

print("\nAffinity Propagation Clustering Performance:")
print(f"Silhouette Score: {silhouette_affinity}")
print(f"Davies-Bouldin Index: {davies_bouldin_affinity}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_affinity}")

# Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_affinity, cmap='viridis', alpha=0.7)
plt.title('PCA of Features - Affinity Propagation with Adjusted Preference')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()
