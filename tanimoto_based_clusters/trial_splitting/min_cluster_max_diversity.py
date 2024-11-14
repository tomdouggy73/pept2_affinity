import pandas as pd
import numpy as np
from vendi_score import vendi  # Ensure vendi_score module is available
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn_extra.cluster import KMedoids
import hdbscan
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom function for calculating Vendi score for a cluster
def calculate_vendi_score_for_cluster(similarity_matrix, cluster_indices):
    if len(cluster_indices) < 2:
        # Skip calculation if the cluster has fewer than 2 points
        return 0  
    
    # Extract the sub-matrix for the given cluster directly from the similarity matrix
    K_ = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
    
    # Calculate and return the Vendi score if K_ is valid
    try:
        return vendi.score_K(K_)
    except Exception as e:
        print(f"Error calculating Vendi score for cluster: {e}")
        return 0  # Default or placeholder score if an error occurs

# Run multiple iterations for each clustering method
def run_clustering_and_vendi(distance_matrix, clustering_method, num_iterations=1000):
    vendi_scores_small = []
    vendi_scores_large = []
    small_cluster_sizes = []
    large_cluster_sizes = []
    small_cluster_indices = []  # Initialize the variable outside the loop

    for i in range(num_iterations):
        clusters, small_cluster, large_cluster = clustering_method(distance_matrix)
        
        # Calculate Vendi scores and sizes for both small and large clusters
        vendi_small = calculate_vendi_score_for_cluster(distance_matrix, small_cluster)
        vendi_large = calculate_vendi_score_for_cluster(distance_matrix, large_cluster)
        
        vendi_scores_small.append(vendi_small)
        vendi_scores_large.append(vendi_large)
        small_cluster_sizes.append(len(small_cluster))
        large_cluster_sizes.append(len(large_cluster))
        
        # Save indices from the last iteration
        if i == num_iterations - 1:
            small_cluster_indices = list(small_cluster)  # Save indices of the small cluster

    mean_vendi_small = np.mean(vendi_scores_small)
    mean_vendi_large = np.mean(vendi_scores_large)
    mean_small_size = int(np.mean(small_cluster_sizes))
    mean_large_size = int(np.mean(large_cluster_sizes))
    
    # Return the mean Vendi scores, sizes, and the small cluster indices
    return mean_vendi_small, mean_vendi_large, mean_small_size, mean_large_size, small_cluster_indices

# K-Center Clustering Implementation
def k_center_clustering(distance_matrix, k=2):
    n_points = distance_matrix.shape[0]
    centers = [np.random.choice(n_points)]
    for _ in range(1, k):
        dist_to_centers = distance_matrix[:, centers].min(axis=1)
        new_center = np.argmax(dist_to_centers)
        centers.append(new_center)
    
    assigned_clusters = np.argmin(distance_matrix[:, centers], axis=1)
    unique, counts = np.unique(assigned_clusters, return_counts=True)
    small_cluster_label = unique[np.argmin(counts)]
    small_cluster = np.where(assigned_clusters == small_cluster_label)[0]
    large_cluster = np.where(assigned_clusters != small_cluster_label)[0]
    
    return None, small_cluster, large_cluster

def save_indices_to_file(cluster_name, indices, compound_names):
    filename = f"{cluster_name}_small_cluster_indices_with_compounds.txt"
    with open(filename, 'w') as f:
        # Combine indices and compound names and save them
        for idx in indices:
            f.write(f"{idx}: {compound_names[idx]}\n")
    print(f"Small cluster indices with compound names for {cluster_name} saved to {filename}")

def main():
    # Load the similarity matrix CSV
    matrix_df = pd.read_csv("tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv", header=0, index_col=0).fillna(0)
    compound_names = matrix_df.index.tolist()  # Extract compound names from the index
    similarity_matrix = matrix_df.to_numpy(dtype=float)  # Convert matrix to numpy array for clustering
    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
    
    np.fill_diagonal(distance_matrix, 0)  # Ensure the distance matrix is symmetric
    
    # Initialize results storage
    methods = ['K-Center', 'Agglomerative', 'Spectral', 'HDBSCAN', 'K-Medoids']
    results = {
        'Method': [],
        'Small Cluster Size': [],
        'Large Cluster Size': [],
        'Vendi Score Small Cluster': [],
        'Vendi Score Large Cluster': []
    }

    whole_dataset_vendi_score = vendi.score_K(similarity_matrix)
    print(f"Vendi Score for the entire dataset: {whole_dataset_vendi_score:.5f}")

    # 1. K-Center Clustering
    mean_vendi_small_kc, mean_vendi_large_kc, small_size_kc, large_size_kc, cluster_indices_kc = run_clustering_and_vendi(similarity_matrix, k_center_clustering)
    results['Method'].append('K-Center')
    results['Small Cluster Size'].append(small_size_kc)
    results['Large Cluster Size'].append(large_size_kc)
    results['Vendi Score Small Cluster'].append(mean_vendi_small_kc)
    results['Vendi Score Large Cluster'].append(mean_vendi_large_kc)
    print(f"K-Center Clustering: Vendi Small={mean_vendi_small_kc:.5f}, Vendi Large={mean_vendi_large_kc:.5f}")
    
    # 2. Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
    def agglomerative_clustering_method(distance_matrix):
        labels_agg = agglomerative.fit_predict(distance_matrix)
        small_label_agg = 0 if np.sum(labels_agg == 0) < np.sum(labels_agg == 1) else 1
        small_agg = np.where(labels_agg == small_label_agg)[0]
        large_agg = np.where(labels_agg != small_label_agg)[0]
        return labels_agg, small_agg, large_agg
    
    mean_vendi_small_agg, mean_vendi_large_agg, small_size_agg, large_size_agg, cluster_indices_agg = run_clustering_and_vendi(similarity_matrix, agglomerative_clustering_method)
    results['Method'].append('Agglomerative')
    results['Vendi Score Small Cluster'].append(mean_vendi_small_agg)
    results['Vendi Score Large Cluster'].append(mean_vendi_large_agg)
    results['Small Cluster Size'].append(small_size_agg)
    results['Large Cluster Size'].append(large_size_agg)
    print(f"Agglomerative Clustering: Vendi Small={mean_vendi_small_agg:.5f}, Vendi Large={mean_vendi_large_agg:.5f}")
    
    # 3. Spectral Clustering
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans')
    def spectral_clustering_method(distance_matrix):
        labels_spectral = spectral.fit_predict(distance_matrix)
        small_label_spectral = 0 if np.sum(labels_spectral == 0) < np.sum(labels_spectral == 1) else 1
        small_spectral = np.where(labels_spectral == small_label_spectral)[0]
        large_spectral = np.where(labels_spectral != small_label_spectral)[0]
        return labels_spectral, small_spectral, large_spectral
    
    mean_vendi_small_spectral, mean_vendi_large_spectral, small_size_spectral, large_size_spectral, cluster_indices_spectral = run_clustering_and_vendi(similarity_matrix, spectral_clustering_method)
    results['Method'].append('Spectral')
    results['Vendi Score Small Cluster'].append(mean_vendi_small_spectral)
    results['Vendi Score Large Cluster'].append(mean_vendi_large_spectral)
    results['Small Cluster Size'].append(small_size_spectral)
    results['Large Cluster Size'].append(large_size_spectral)
    print(f"Spectral Clustering: Vendi Small={mean_vendi_small_spectral:.5f}, Vendi Large={mean_vendi_large_spectral:.5f}")
   
    if cluster_indices_spectral:
        print(f"Spectral Clustering Small Cluster Indices: {cluster_indices_spectral}")
        save_indices_to_file("vendi/test_set_clusters/Spectral_Clustering", cluster_indices_spectral, compound_names)
    else:
        print("No small cluster found for Spectral Clustering.")

    # 4. HDBSCAN
    def hdbscan_clustering_method(similarity_matrix):
        # Convert similarity to distance for clustering
        distance_matrix = 1 - similarity_matrix
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=10)
        labels_hdb = clusterer.fit_predict(distance_matrix)
        
        # Identify unique clusters (excluding noise, labeled as -1)
        unique_hdb, counts_hdb = np.unique(labels_hdb[labels_hdb != -1], return_counts=True)
        
        if len(unique_hdb) == 0:
            # No clusters were found, all points are labeled as noise
            print("No clusters found by HDBSCAN. All points classified as noise.")
            small_cluster = np.array([])
            large_cluster = np.array([])
        else:
            # Proceed with finding the smallest and largest clusters
            small_cluster_label_hdb = unique_hdb[np.argmin(counts_hdb)]
            small_cluster = np.where(labels_hdb == small_cluster_label_hdb)[0]
            large_cluster = np.where(labels_hdb != small_cluster_label_hdb)[0]
        
        return labels_hdb, small_cluster, large_cluster
    
    mean_vendi_small_hdb, mean_vendi_large_hdb, small_size_hdb, large_size_hdb, cluster_indices_hdb = run_clustering_and_vendi(similarity_matrix, hdbscan_clustering_method)
    results['Method'].append('HDBSCAN')
    results['Vendi Score Small Cluster'].append(mean_vendi_small_hdb)
    results['Vendi Score Large Cluster'].append(mean_vendi_large_hdb)
    results['Small Cluster Size'].append(small_size_hdb)
    results['Large Cluster Size'].append(large_size_hdb)
    print(f"HDBSCAN Clustering: Vendi Small={mean_vendi_small_hdb:.5f}, Vendi Large={mean_vendi_large_hdb:.5f}")
    
    # Print and save small cluster indices for HDBSCAN
    if cluster_indices_hdb:
        print(f"HDBSCAN Small Cluster Indices: {cluster_indices_hdb}")
        save_indices_to_file("vendi/test_set_clusters/HDBSCAN", cluster_indices_hdb, compound_names)
    else:
        print("No small cluster found for HDBSCAN.")
    

    # If lengths are mismatched, fix by adding placeholders
    max_len = max(len(lst) for lst in results.values())
    for key in results.keys():
        if len(results[key]) < max_len:
            results[key].extend([None] * (max_len - len(results[key])))

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    mean_vendi_small_spectral, mean_vendi_large_spectral, small_size_spectral, large_size_spectral, small_cluster_indices_spectral = run_clustering_and_vendi(similarity_matrix, spectral_clustering_method)
    print(f"Spectral Clustering Small Cluster Indices: {small_cluster_indices_spectral}")

    mean_vendi_small_hdb, mean_vendi_large_hdb, small_size_hdb, large_size_hdb, small_cluster_indices_hdb = run_clustering_and_vendi(similarity_matrix, hdbscan_clustering_method)
    print(f"HDBSCAN Small Cluster Indices: {small_cluster_indices_hdb}")

    print("\nClustering Results:")
    print(results_df)
    
    # Plotting Vendi Scores for Small and Large Clusters per Method
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot for Small Clusters
    axes[0].bar(results_df['Method'], results_df['Vendi Score Small Cluster'], color='skyblue')
    axes[0].set_xlabel('Clustering Method')
    axes[0].set_ylabel('Vendi Score (Small Cluster)')
    axes[0].set_title('Vendi Score for Small Clusters')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot for Large Clusters
    axes[1].bar(results_df['Method'], results_df['Vendi Score Large Cluster'], color='salmon')
    axes[1].set_xlabel('Clustering Method')
    axes[1].set_ylabel('Vendi Score (Large Cluster)')
    axes[1].set_title('Vendi Score for Large Clusters')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('vendi/vendi_score_comparison_methods.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()