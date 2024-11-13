import pandas as pd
import numpy as np
from vendi_score import vendi
import matplotlib.pyplot as plt

# Load the CSV, fill any missing values, and convert to a NumPy array
ts_matrix_df = pd.read_csv(
    "tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv",
    header=0
).iloc[:, 1:].fillna(0)
ts_matrix = ts_matrix_df.to_numpy(dtype=float)

# Parameters
num_iterations = 1000  # Number of random iterations
cluster_range = range(2, 21)  # Clusters from 2 to 20

# Store mean and variance for each cluster setup
random_vendi_means = []
random_vendi_variances = []

# Random Splits and Vendi Score Calculation for each cluster number
for n_clusters in cluster_range:
    random_scores, random_variance = [], []

    for iteration in range(num_iterations):
        indices = np.arange(ts_matrix.shape[0])
        np.random.shuffle(indices)
        clusters = np.array_split(indices, n_clusters)

        # Compute Vendi scores for each cluster subset
        vendi_scores = []
        for cluster_indices in clusters:
            ts_matrix_cluster = ts_matrix[np.ix_(cluster_indices, cluster_indices)]
            vendi_score = vendi.score_K(ts_matrix_cluster)
            vendi_scores.append(vendi_score)

        # Calculate the average Vendi score for this random split
        avg_vendi_score = np.mean(vendi_scores)
        avg_vendi_var = np.var(vendi_scores)
        random_scores.append(avg_vendi_score)
        random_variance.append(avg_vendi_var)

    # Mean and variance of Vendi scores for random splits
    random_mean = np.mean(random_scores)
    random_var = np.mean(random_variance)
    random_vendi_means.append(random_mean)
    random_vendi_variances.append(random_var)

    print(f"Clusters: {n_clusters}, Random mean Vendi = {random_mean:.5f}, Random variance Vendi = {random_var:.5f}")

print(f"Total clusters processed: {len(random_vendi_variances)}, {len(random_vendi_means)},")
assert len(random_vendi_variances) == len(cluster_range), "Mismatch in the expected number of variances."

best_vendi_scores = pd.read_csv('vendi/best_scores_vendi_max_dissim.csv')

# Extract cluster counts, scores, and variances from best-seed data
best_clusters = best_vendi_scores['Clusters'].values
best_means = best_vendi_scores['Score'].values  # Assuming 'Score' column holds the mean values
best_variances = best_vendi_scores['Variance'].values

# Define your cluster range and random mean/variance lists
cluster_range = list(range(2, 21))  # Generates list for clusters 2 to 20

# Ensure correct length for random data lists (should match the cluster range length)
if len(random_vendi_means) != len(cluster_range) or len(random_vendi_variances) != len(cluster_range):
    raise ValueError("Ensure random_vendi_means and random_vendi_variances have 19 entries each.")

# Plotting Vendi Score Mean with Variance (Error Bars) for Random vs. Best-seed Configuration
plt.figure(figsize=(10, 6))

# Plot random cluster Vendi scores with error bars (variance)
plt.errorbar(cluster_range, random_vendi_means, yerr=np.sqrt(random_vendi_variances), 
             label='Random Cluster Split', marker='o', color='blue', linestyle='--', capsize=5)

# Plot best-seed cluster Vendi scores with error bars (variance)
plt.errorbar(best_clusters, best_means, yerr=np.sqrt(best_variances), 
             label='Best-seed Clustering', marker='o', color='red', linestyle='-', capsize=5)

# Labeling and display options
plt.xlabel('Number of Clusters')
plt.ylabel('Vendi Score')
plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) 
plt.title('Mean Vendi Score with Variance for Random Splits vs. Best-seed Clustering')
plt.legend()

plt.savefig('vendi/vendi_score_variance_comparison.png', dpi=300)  # Save with high resolution (300 DPI)
plt.show()