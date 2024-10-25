import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Step 1: Load your distance matrix
file_path = 'tanimoto/tanimoto_similarity_matrix.csv'
df = pd.read_csv(file_path, index_col=0)
labels = df.index.tolist()

# The similarity matrix (Tanimoto)
similarity_matrix = df.values
dissimilarity_matrix = 1 - similarity_matrix  

# Function to calculate the average similarity for a given clustering
def calculate_average_similarity(clusters, similarity_matrix):
    total_similarity = 0
    total_pairs = 0

    for cluster_points in clusters.values():
        # Get pairwise similarities for all points within the cluster
        if len(cluster_points) > 1:
            cluster_similarities = similarity_matrix[np.ix_(cluster_points, cluster_points)]
            total_similarity += np.sum(cluster_similarities) - np.trace(cluster_similarities)  # Exclude self-similarity
            total_pairs += len(cluster_points) * (len(cluster_points) - 1)  # Number of pairwise comparisons

    # Return the average similarity across all clusters
    if total_pairs == 0:
        return 0
    return total_similarity / total_pairs

# Set the seed 
np.random.seed(101885)

# Number of clusters
n_clusters = 9

# Initialize clusters
clusters = {i: [] for i in range(n_clusters)}
unassigned = list(range(len(dissimilarity_matrix)))

# Assign one random point to each cluster
for cluster_id in range(n_clusters):
    seed = np.random.choice(unassigned)
    clusters[cluster_id].append(seed)
    unassigned.remove(seed)

# Iteratively assign the most dissimilar points to each cluster
while unassigned:
    for cluster_id in range(n_clusters):
        if not unassigned:
            break
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

# Step 3: Prepare the output DataFrame with cluster assignments
cluster_assignments = []
for cluster_id, points in clusters.items():
    for point in points:
        cluster_assignments.append([labels[point], cluster_id])

# Convert to DataFrame
output_df = pd.DataFrame(cluster_assignments, columns=["Label", "Cluster"])

# Step 4: Save the clusters to a CSV file
output_df.to_csv('clusters_seed_101885.csv', index=False)

print("Cluster assignments saved to 'clusters_seed_101885.csv'.")

# Step 5: Load the affinity data from dataset_smiles.csv
affinity_file_path = 'dataset_smiles.csv'  # Replace with your actual file path
affinity_data = pd.read_csv(affinity_file_path)

# Step 6: Merge the cluster assignments with the affinity data, keeping only 'SKPT (PEPT2)' without 'Compound'
final_data = pd.merge(output_df, affinity_data[['Compound', 'SKPT (PEPT2)']], 
                      left_on='Label', right_on='Compound', how='left')

# Remove column
final_data = final_data.drop(columns=['Label'])

# Step 7: Save to CSV file
output_file_path = 'tanimoto/seed_101855_affinity.csv'  # Replace with your desired output path
final_data.to_csv(output_file_path, index=False)

print(f"Affinity data combined and saved to {output_file_path}.")