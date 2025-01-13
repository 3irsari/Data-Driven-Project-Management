from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '*/cleaned_survey_data.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering
features = ['confidence_influence', 'ai_confidence_increase']  # Example features
X = data[features].dropna()  # Drop rows with missing values in selected features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters_kmeans = kmeans.fit_predict(X_scaled)

# Evaluate K-Means clustering
silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans)
calinski_harabasz_kmeans = calinski_harabasz_score(X_scaled, clusters_kmeans)

print("K-Means Clustering Performance:")
print(f"Silhouette Score: {silhouette_kmeans}")
print(f"Davies-Bouldin Index: {davies_bouldin_kmeans}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_kmeans}")

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# Evaluate DBSCAN clustering
# Note: Silhouette score and other metrics may not be applicable if DBSCAN assigns all points to noise (-1)
if len(set(clusters_dbscan)) > 1:
    silhouette_dbscan = silhouette_score(X_scaled, clusters_dbscan)
    davies_bouldin_dbscan = davies_bouldin_score(X_scaled, clusters_dbscan)
    calinski_harabasz_dbscan = calinski_harabasz_score(X_scaled, clusters_dbscan)

    print("\nDBSCAN Clustering Performance:")
    print(f"Silhouette Score: {silhouette_dbscan}")
    print(f"Davies-Bouldin Index: {davies_bouldin_dbscan}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_dbscan}")
else:
    print("\nDBSCAN did not form valid clusters for evaluation.")

# Visualize the clusters (optional)
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_kmeans, cmap='viridis', marker='o')
plt.title('K-Means Clustering of AI Adopters')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.colorbar(label='Cluster')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_dbscan, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering of AI Adopters')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.colorbar(label='Cluster')
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '*/cleaned_survey_data.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering
features = ['confidence_influence', 'ai_confidence_increase']  # Example features
X = data[features].dropna()  # Drop rows with missing values in selected features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=2)
clusters_hierarchical = hierarchical.fit_predict(X_scaled)

# Evaluate Hierarchical Clustering
silhouette_hierarchical = silhouette_score(X_scaled, clusters_hierarchical)
davies_bouldin_hierarchical = davies_bouldin_score(X_scaled, clusters_hierarchical)
calinski_harabasz_hierarchical = calinski_harabasz_score(X_scaled, clusters_hierarchical)

print("\nHierarchical Clustering Performance:")
print(f"Silhouette Score: {silhouette_hierarchical}")
print(f"Davies-Bouldin Index: {davies_bouldin_hierarchical}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_hierarchical}")

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters_hierarchical, cmap='viridis', marker='o')
plt.title('Hierarchical Clustering of AI Adopters')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.colorbar(label='Cluster')
plt.show()


from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, random_state=42)
clusters_gmm = gmm.fit_predict(X_scaled)

# Evaluate GMM Clustering
silhouette_gmm = silhouette_score(X_scaled, clusters_gmm)
davies_bouldin_gmm = davies_bouldin_score(X_scaled, clusters_gmm)
calinski_harabasz_gmm = calinski_harabasz_score(X_scaled, clusters_gmm)

print("\nGMM Clustering Performance:")
print(f"Silhouette Score: {silhouette_gmm}")
print(f"Davies-Bouldin Index: {davies_bouldin_gmm}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_gmm}")

from sklearn.cluster import MeanShift

mean_shift = MeanShift()
clusters_meanshift = mean_shift.fit_predict(X_scaled)

# Evaluate Mean-Shift Clustering
silhouette_meanshift = silhouette_score(X_scaled, clusters_meanshift)
davies_bouldin_meanshift = davies_bouldin_score(X_scaled, clusters_meanshift)
calinski_harabasz_meanshift = calinski_harabasz_score(X_scaled, clusters_meanshift)

print("\nMean-Shift Clustering Performance:")
print(f"Silhouette Score: {silhouette_meanshift}")
print(f"Davies-Bouldin Index: {davies_bouldin_meanshift}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_meanshift}")

from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters=2, random_state=42, affinity='nearest_neighbors')
clusters_spectral = spectral.fit_predict(X_scaled)

# Evaluate Spectral Clustering
silhouette_spectral = silhouette_score(X_scaled, clusters_spectral)
davies_bouldin_spectral = davies_bouldin_score(X_scaled, clusters_spectral)
calinski_harabasz_spectral = calinski_harabasz_score(X_scaled, clusters_spectral)

print("\nSpectral Clustering Performance:")
print(f"Silhouette Score: {silhouette_spectral}")
print(f"Davies-Bouldin Index: {davies_bouldin_spectral}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_spectral}")


from minisom import MiniSom

som = MiniSom(x=2, y=1, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, num_iteration=100)

clusters_som = [som.winner(x) for x in X_scaled]
clusters_som = [c[0] for c in clusters_som]  # Convert tuples to cluster labels
silhouette_som = silhouette_score(X_scaled, clusters_som)

print("\nSOM Clustering Performance:")
print(f"Silhouette Score: {silhouette_som}")

import hdbscan

hdb = hdbscan.HDBSCAN(min_samples=5)
clusters_hdbscan = hdb.fit_predict(X_scaled)

# Evaluate HDBSCAN (only if valid clusters are formed)
if len(set(clusters_hdbscan)) > 1:
    silhouette_hdbscan = silhouette_score(X_scaled, clusters_hdbscan)
    davies_bouldin_hdbscan = davies_bouldin_score(X_scaled, clusters_hdbscan)
    print("\nHDBSCAN Clustering Performance:")
    print(f"Silhouette Score: {silhouette_hdbscan}")
    print(f"Davies-Bouldin Index: {davies_bouldin_hdbscan}")
else:
    print("HDBSCAN did not form valid clusters.")

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch, AffinityPropagation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '*/cleaned_survey_data.csv'
data = pd.read_csv(file_path)

# Select relevant features for clustering
features = ['confidence_influence', 'ai_confidence_increase']
X = data[features].dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply OPTICS clustering
optics = OPTICS(min_samples=5)
clusters_optics = optics.fit_predict(X_scaled)

# Evaluate OPTICS clustering
if len(set(clusters_optics)) > 1:
    silhouette_optics = silhouette_score(X_scaled, clusters_optics)
    davies_bouldin_optics = davies_bouldin_score(X_scaled, clusters_optics)
    calinski_harabasz_optics = calinski_harabasz_score(X_scaled, clusters_optics)

    print("\nOPTICS Clustering Performance:")
    print(f"Silhouette Score: {silhouette_optics}")
    print(f"Davies-Bouldin Index: {davies_bouldin_optics}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_optics}")
else:
    print("\nOPTICS did not form valid clusters for evaluation.")

# Apply BIRCH clustering
birch = Birch(n_clusters=2)
clusters_birch = birch.fit_predict(X_scaled)

# Evaluate BIRCH clustering
silhouette_birch = silhouette_score(X_scaled, clusters_birch)
davies_bouldin_birch = davies_bouldin_score(X_scaled, clusters_birch)
calinski_harabasz_birch = calinski_harabasz_score(X_scaled, clusters_birch)

print("\nBIRCH Clustering Performance:")
print(f"Silhouette Score: {silhouette_birch}")
print(f"Davies-Bouldin Index: {davies_bouldin_birch}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_birch}")

# Apply Affinity Propagation clustering
affinity_propagation = AffinityPropagation(random_state=42)
clusters_affinity = affinity_propagation.fit_predict(X_scaled)

# Evaluate Affinity Propagation clustering
silhouette_affinity = silhouette_score(X_scaled, clusters_affinity)
davies_bouldin_affinity = davies_bouldin_score(X_scaled, clusters_affinity)
calinski_harabasz_affinity = calinski_harabasz_score(X_scaled, clusters_affinity)

print("\nAffinity Propagation Clustering Performance:")
print(f"Silhouette Score: {silhouette_affinity}")
print(f"Davies-Bouldin Index: {davies_bouldin_affinity}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_affinity}")

# Visualize the clusters for each algorithm
def plot_clusters(X_scaled, clusters, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.colorbar(label='Cluster')
    plt.show()

plot_clusters(X_scaled, clusters_optics, 'OPTICS Clustering of AI Adopters')
plot_clusters(X_scaled, clusters_birch, 'BIRCH Clustering of AI Adopters')
plot_clusters(X_scaled, clusters_affinity, 'Affinity Propagation Clustering of AI Adopters')

# Note: Deep Clustering and Ensemble Clustering require custom implementations or additional libraries.
