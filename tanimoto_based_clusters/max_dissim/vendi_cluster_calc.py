import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from tqdm import tqdm
from vendi_score import vendi

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# Function to calculate average similarity for seed optimizer
def calculate_average_similarity(clusters, dissimilarity_matrix):
    total_similarity = 0
    total_pairs = 0

    for cluster_points in clusters.values():
        if len(cluster_points) > 1:
            cluster_similarities = dissimilarity_matrix[np.ix_(cluster_points, cluster_points)]
            total_similarity += np.sum(cluster_similarities) - np.trace(cluster_similarities)
            total_pairs += len(cluster_points) * (len(cluster_points) - 1)

    return total_similarity / total_pairs if total_pairs > 0 else 0

# Main loop for clusters n=2 to n=21
for n_clusters in tqdm(range(2,21), desc="Processing clusters", unit="cluster"):
    # Step 1: Load the distance matrix for each n_clusters
    file_path = 'tanimoto_based_clusters/tanimoto_matrix/tanimoto_similarity_matrix.csv'
    df = pd.read_csv(file_path, index_col=0)
    labels = df.index.tolist()
    similarity_matrix = df.values
    dissimilarity_matrix = 1 - similarity_matrix

    # Step 2: Seed optimizer to calculate average similarity for 1000 random seeds
    top_seeds_df = pd.DataFrame(columns=["Seed", "Average Similarity"])

    for iteration in range(1000):
        np.random.seed(iteration)
        clusters = {i: [] for i in range(n_clusters)}
        unassigned = list(range(len(dissimilarity_matrix)))

        for cluster_id in range(n_clusters):
            seed = np.random.choice(unassigned)
            clusters[cluster_id].append(seed)
            unassigned.remove(seed)

        while unassigned:
            for cluster_id in range(n_clusters):
                if not unassigned:
                    break
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

        avg_similarity = calculate_average_similarity(clusters, dissimilarity_matrix)

        # Track the best seeds based on average similarity
        top_seeds_df = pd.concat([top_seeds_df, pd.DataFrame({"Seed": [iteration], "Average Similarity": [avg_similarity]})], ignore_index=True)

    # Select the best seed (highest average similarity)
    top_seeds_df.sort_values(by="Average Similarity", ascending=False, inplace=True)
    top_seed_iteration = top_seeds_df.iloc[0]["Seed"]  # Best seed

    # Step 3: Generate cluster assignments using the best seed
    np.random.seed(top_seed_iteration)
    clusters = {i: [] for i in range(n_clusters)}
    unassigned = list(range(len(dissimilarity_matrix)))

    for cluster_id in range(n_clusters):
        seed = np.random.choice(unassigned)
        clusters[cluster_id].append(seed)
        unassigned.remove(seed)

    while unassigned:
        for cluster_id in range(n_clusters):
            if not unassigned:
                break
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

    # Save the cluster assignments to CSV
    cluster_assignments = []
    for cluster_id, points in clusters.items():
        for point in points:
            cluster_assignments.append([labels[point], cluster_id])

    # Convert to DataFrame and save
    output_df = pd.DataFrame(cluster_assignments, columns=["Label", "Cluster"])
    output_file_path = f'tanimoto_based_clusters/max_dissim/n_clusters/{n_clusters}_clusters/clusters_{n_clusters}_seed_{top_seed_iteration}.csv'
    output_df.to_csv(output_file_path, index=False)
    print(f"Cluster assignments saved to {output_file_path}.")

    # Step 4: Load the affinity data from dataset_smiles.csv and combine
    affinity_file_path = 'source_data/dataset_smiles.csv'
    affinity_data = pd.read_csv(affinity_file_path)
    final_data = pd.merge(output_df, affinity_data[['Compound', 'SKPT (PEPT2)']], 
                        left_on='Label', right_on='Compound', how='left')

    final_data = final_data.drop(columns=['Label'])

    output_file_path = f'tanimoto_based_clusters/max_dissim/n_clusters/{n_clusters}_clusters/clusters_{n_clusters}_seed_{top_seed_iteration}_affinity.csv'  # Replace with your desired output path
    final_data.to_csv(output_file_path, index=False)

    print(f"Affinity data combined and saved to {output_file_path}.")

    # Step 5: Merge with additional data
    seed_affinity = pd.read_csv(f'tanimoto_based_clusters/max_dissim/n_clusters/{n_clusters}_clusters/clusters_{n_clusters}_seed_{top_seed_iteration}_affinity.csv')
    dataset_smiles = pd.read_csv('source_data/dataset_smiles.csv')
    dataset_morganfingerprints = pd.read_csv('source_data/dataset_morganfingerprints.csv', header=None)
    
    dataset_morganfingerprints.columns = ['SMILES'] + [f'Fingerprint_{i}' for i in range(1, dataset_morganfingerprints.shape[1])]

    merged_data = pd.merge(seed_affinity, dataset_smiles, on='Compound', how='inner')
    final_merged_data = pd.merge(merged_data, dataset_morganfingerprints, on='SMILES', how='inner')

    final_merged_data = final_merged_data.loc[:, ~final_merged_data.columns.str.contains('^Unnamed')]

    # Save the final merged data
    final_output_file = f'tanimoto_based_clusters/max_dissim/morgan_fingerprints/final_merged_data_with_fingerprints_{n_clusters}_clusters_max_dissim.csv'
    final_merged_data.to_csv(final_output_file, index=False)
    print(f"Final merged data saved to {final_output_file}.")

    # Step 6: Calculate the Vendi score
    dataset = pd.read_csv(final_output_file)
    indices = dataset["SKPT (PEPT2)_x"].dropna().index
    smiles = dataset["SMILES"][indices].values
    Ki = dataset["SKPT (PEPT2)_x"].dropna().values
    compounds_names = dataset["Compound"][indices].values
    logKi = np.log10(Ki)
    clusters = dataset['Cluster'].values

    unique_clusters = np.unique(clusters)
    vendi_scores = []

    for test_cluster in unique_clusters:
        test_indices = np.where(clusters == test_cluster)[0]
        test_smiles = [smiles[idx] for idx in test_indices]
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2) for smi in test_smiles]

        tanimoto_matrix = np.zeros((len(fps), len(fps)))
        for i, fp1 in enumerate(fps):
            for j, fp2 in enumerate(fps):
                tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)


        vendi_score_test = vendi.score_K(tanimoto_matrix)
        vendi_scores.append(vendi_score_test)

    vendi_mean = np.mean(vendi_scores)
    vendi_variance = np.var(vendi_scores)
    print(f"For {n_clusters} clusters: Mean vendi score = {vendi_mean:.5f}, variance = {vendi_variance:.5f}")