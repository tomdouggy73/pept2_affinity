import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm  # Import tqdm for progress tracking

# Step 1: Load your distance matrix
file_path = '../tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv'
df = pd.read_csv(file_path, index_col=0)
labels = df.index.tolist()

# The similarity matrix (Tanimoto)
similarity_matrix = df.values
dissimilarity_matrix = 1 - similarity_matrix  # Dissimilarity matrix (1 - similarity)

# Function to calculate the average similarity for a given clustering
def calculate_average_similarity(clusters, dissimilarity_matrix):
    total_similarity = 0
    total_pairs = 0

    for cluster_points in clusters.values():
        # Get pairwise similarities for all points within the cluster
        if len(cluster_points) > 1:
            cluster_similarities = dissimilarity_matrix[np.ix_(cluster_points, cluster_points)]
            total_similarity += np.sum(cluster_similarities) - np.trace(cluster_similarities)
            total_pairs += len(cluster_points) * (len(cluster_points) - 1)

    # Return the average similarity across all clusters
    if total_pairs == 0:
        return 0
    return total_similarity / total_pairs

# Variables to store the best seeds and their corresponding average similarities
top_seeds_df = pd.DataFrame(columns=["Seed", "Average Similarity"])

# Number of clusters
n_clusters = 3

# Step 2: Iterate through 1,000 random seeds with progress tracking
for iteration in tqdm(range(1000), desc="Processing seeds", unit="seed"):
    np.random.seed(iteration)  # Set the seed for reproducibility

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

    # Step 3: Calculate the average similarity for this clustering
    avg_similarity = calculate_average_similarity(clusters, dissimilarity_matrix)

    # Step 4: Manage the top seeds
    if len(top_seeds_df) < 100:
        # Use pd.concat instead of append
        top_seeds_df = pd.concat([top_seeds_df, pd.DataFrame({"Seed": [iteration], "Average Similarity": [avg_similarity]})], ignore_index=True)
    else:
        # Find the index of the seed with the highest average similarity
        max_avg_index = top_seeds_df['Average Similarity'].idxmax()
        max_avg_value = top_seeds_df['Average Similarity'].max()

        # Check if the current seed has a lower average similarity
        if avg_similarity < max_avg_value:
            top_seeds_df.loc[max_avg_index] = [iteration, avg_similarity]  # Replace the worst seed

# Step 5: Save the top seeds to a CSV file
top_seeds_df.sort_values(by="Average Similarity", inplace=True)  # Sort by Average Similarity
top_seeds_df.to_csv('../tanimoto_based_clusters/top_100_seeds.csv', index=False)

print("Top 100 seeds and their average similarities saved to 'top_100_seeds_3_clusters.csv'.")