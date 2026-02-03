"""
HIERARCHICAL CLUSTERING WITH GOWER DISTANCE
===========================================

Purpose:
    Performs hierarchical clustering on mixed‑type donor data using the 
    animal_charity_donation_records.csv dataset (publicly available via Kaggle).

    The workflow:
        (1) Loads and preprocesses categorical and numeric features
        (2) Computes Gower distance to handle mixed data types
        (3) Applies hierarchical clustering and evaluates cluster solutions
        (4) Selects the optimal number of clusters using silhouette analysis
        (5) Generates a radar chart to visualise cluster profiles

Key Features:
    - Uses Gower distance for mixed categorical/numeric variables
    - Supports multiple linkage methods
    - Includes silhouette‑based guidance for selecting k
    - Produces a radar chart summarising cluster characteristics

Output:
    Cluster assignments, silhouette metrics, and a radar chart visualising 
    the defining attributes of each cluster.

Author: ktevon
Date: November 2025
"""

# --- Imports ---

# Import modules
import pandas as pd
import numpy as np
import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# To make the code reproducable
np.random.seed(42)

# Get all keys from the rcParams dictionary
plt.rcParams.keys()

# Global font change
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

# Import data
data_full = pd.read_csv("animal_charity_donation_records.csv")

# Inspect data frame
print(data_full.head())
print(data_full.info())

# data_full.groupby("country").size()

# --- Preprocessing mixed data ---

# 1. Convert date field
data_full['donation_date'] = pd.to_datetime(data_full['donation_date'])

# 2. Drop ID field
data_clust = data_full.drop(columns=['donor_id'])

# 3. Ensure booleans are actual bool
data_clust["newsletter_opt_in"] = data_clust["newsletter_opt_in"].astype(bool)

# --- Compute Gower distance matrix ---

distance_matrix = gower.gower_matrix(data_clust)
print(distance_matrix)

# --- Convert full distance matrix to condensed form ---

# SciPy’s linkage expects a condensed (upper triangle) distance vector.
# SciPy needs condensed form, but Gower returns a square matrix.
distance_condensed = squareform(distance_matrix, checks=False)
print(distance_condensed)

# --- Perform hierarchical clustering ---

# You cannot use "ward" with Gower distances. Use:
# - "average" (UPGMA)
# - "complete" (max distance)
# - "single" (not recommended — chaining)
# - "weighted"
Z = linkage(distance_condensed, method='average')

# --- Plot dendrogram ---

plt.figure(figsize=(14, 6))
dendrogram(Z, truncate_mode='level', p=5) # Condenses a large dendrogram for readability
plt.title("Hierarchical Clustering Dendrogram (Gower Distance)")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# --- Silhouette Score for Gower Distance ---

# Function to compute silhouette score for any number of clusters
def silhouette_for_k(Z, distance_matrix, k):
    clusters = fcluster(Z, k, criterion='maxclust')
    score = silhouette_score(distance_matrix, clusters, metric='precomputed')
    return score

# Try several values of k - Generally 2-10 clusters

scores = {}

for k in range(2, 11):
    score = silhouette_for_k(Z, distance_matrix, k)
    scores[k] = score
    print(f"k={k}, silhouette={score:.4f}")

# Find the best k
best_k = max(scores, key=scores.get)
best_k, scores[best_k]

# Plot silhouette scores
plt.plot(list(scores.keys()), list(scores.values()), marker='o')
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()

# --- Choose number of clusters and assign them ---

k = 5 # Although best_k is 2, 5 is more useful
clusters = fcluster(Z, k, criterion='maxclust')

# Add clusters back into the dataset:
data_full['cluster'] = clusters

# --- Inspect cluster sizes ---

cluster_count = data_full['cluster'].value_counts().sort_index()
cluster_count

# Check the total
total_sum = cluster_count.sum()
total_sum

# Compute cluster percentages
cluster_percentage = (cluster_count / total_sum)*100
cluster_percentage

# Visualise

categories = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"]
values = cluster_percentage.values.astype(str) + "%"
print(values)

fig, ax = plt.subplots()
ax.bar(categories, values)
ax.set_ylim(bottom = -0.5)
plt.show()

# --- Get cluster profiles ---

cluster_profiles = data_full.groupby('cluster').agg({
    'donation_amount': ['mean', 'median'],
    'gender': lambda x: x.value_counts().index[0],
    'age_group': lambda x: x.value_counts().index[0],
    'country': lambda x: x.value_counts().index[0],
    'donation_type': lambda x: x.value_counts().index[0],
    'newsletter_opt_in': 'mean'
})

cluster_profiles

cluster_data = data_full.groupby('cluster').agg({
    'donation_amount': 'mean',
    'gender': lambda x: x.value_counts().get("Male", 0) / len(x),
    'donation_type': lambda x: x.value_counts().get("Monthly", 0) / len(x),
    'newsletter_opt_in': 'mean'
})

cluster_data

# --- Create a radar chart ---

# Min-max scale each column to 0-1
cluster_data_scaled = (cluster_data - cluster_data.min()) / (cluster_data.max() - cluster_data.min())
print(cluster_data_scaled.describe())

# Convert to dictionary for the radar chart loop
cluster_data_scaled = {
    cluster: cluster_data_scaled.loc[cluster].tolist()
    for cluster in cluster_data_scaled.index
}

cluster_data_scaled

num_vars = cluster_data.shape[1]
num_vars

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # complete the loop - This adds 0.0 at the end.
angles

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for cluster, values in cluster_data_scaled.items():
    values += values[:1] # complete the loop
    ax.plot(angles, values, linewidth = 1, label = cluster)
    ax.fill(angles, values, alpha = 0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(cluster_data.columns)
ax.set_title("Cluster Profiles Radar Chart", size=15)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.show()
