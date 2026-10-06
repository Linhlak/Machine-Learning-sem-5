import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==========================================================
# 1. Load and Explore the Dataset
# ==========================================================
print("--- Step 1: Loading Dataset ---")
df_mall = pd.read_csv('/Mall_Customers.csv')
print(f"Dataset loaded with shape: {df_mall.shape}\n")
display(df_mall.head())

# ==========================================================
# 2. Extract Clustering Features
# ==========================================================
# Selecting 'Annual Income (k$)' and 'Spending Score (1-100)'
X_clusters = df_mall.iloc[:, [3, 4]].values

# ==========================================================
# 3. Determine Optimal Clusters with the Elbow Method
# ==========================================================
print("\n--- Step 2: Running Elbow Method ---")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_clusters)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

ax1.plot(range(1, 11), wcss, marker='o', linestyle='--')
ax1.set_title('The Elbow Method')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('WCSS')
ax1.grid(True)

# ==========================================================
# 4. Train K-Means Model with Optimal Clusters (k=5)
# ==========================================================
print("--- Step 3: Training K-Means Model (k=5) ---")
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_clusters)
print("Model clustering complete.\n")

# ==========================================================
# 5. Visualize the Segmented Clusters
# ==========================================================
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Standard Users', 'Careful Users', 'Target Group', 'Spendthrifts', 'Sensible Users']

for cluster_idx in range(k_optimal):
    ax2.scatter(X_clusters[y_kmeans == cluster_idx, 0], 
                X_clusters[y_kmeans == cluster_idx, 1], 
                s=100, 
                c=colors[cluster_idx], 
                label=labels[cluster_idx])

# Plot cluster centroids
ax2.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1], 
            s=300, 
            c='yellow', 
            label='Centroids', 
            edgecolor='black')

ax2.set_title('Clusters of Mall Customers')
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
