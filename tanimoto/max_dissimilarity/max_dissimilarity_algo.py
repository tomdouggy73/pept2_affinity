import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Step 1: Load your distance matrix
file_path = 'tanimoto/tanimoto_similarity_matrix.csv'
df = pd.read_csv(file_path, index_col=0)
labels = df.index.tolist()
dissimilarity_matrix = 1 - df.values  # 1 - similarity = dissimilarity

# Step 2: Initialize clusters
n_clusters = 9
clusters = {i: [] for i in range(n_clusters)}
unassigned = list(range(len(dissimilarity_matrix)))

# Step 3: Assign one random point to each cluster
for cluster_id in range(n_clusters):
    seed = np.random.choice(unassigned)
    clusters[cluster_id].append(seed)
    unassigned.remove(seed)

# Step 4: Iteratively assign the most dissimilar points to each cluster
while unassigned:
    for cluster_id in range(n_clusters):
        if not unassigned: break
        # Calculate distance of unassigned points to current cluster members
        current_cluster_points = clusters[cluster_id]
        max_avg_dist = -1
        next_point = None
        
        for point in unassigned:
            avg_dist = np.mean([dissimilarity_matrix[point][p] for p in current_cluster_points])
            if avg_dist > max_avg_dist:
                max_avg_dist = avg_dist
                next_point = point
        
        if next_point is not None:
            clusters[cluster_id].append(next_point)
            unassigned.remove(next_point)

# Step 5: Convert cluster assignments to labels and save
cluster_labels = np.zeros(len(labels), dtype=int)
for cluster_id, points in clusters.items():
    for p in points:
        cluster_labels[p] = cluster_id

# Save cluster labels
clustered_data = pd.DataFrame({'Label': labels, 'Cluster': cluster_labels})
clustered_data.to_csv('tanimoto/max_dissimilarity_clusters.csv', index=False)