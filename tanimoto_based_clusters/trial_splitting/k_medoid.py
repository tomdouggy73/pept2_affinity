import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids  # sklearn-extra library for KMedoids

# Step 1: Load the Tanimoto similarity matrix CSV into a DataFrame to retain labels
file_path = 'tanimoto/tanimoto_similarity_matrix.csv'  # Replace with your actual file path
df = pd.read_csv(file_path, index_col=0)  # Load CSV and set the first column as index

# Step 2: Extract the similarity matrix and labels
similarity_matrix = df.values  # Get the numerical values as a NumPy array
labels = df.index.tolist()      # Get the labels (row names)

# Step 3: Convert similarity matrix to dissimilarity (distance) matrix (1 - similarity)
dissimilarity_matrix = 1 - similarity_matrix

# Step 3: Apply K-medoids (better suited for distance matrices)
n_clusters = 9
kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed")  # Use dissimilarity matrix
clusters = kmedoids.fit_predict(dissimilarity_matrix)

# Step 4: Save the cluster assignments
clustered_data = pd.DataFrame({
    'Label': labels,
    'Cluster': clusters
})

# Step 7: Create a DataFrame to save the cluster assignments
clustered_data = pd.DataFrame({
    'Label': labels,
    'Cluster': clusters
})

# Step 8: Load the affinity data from dataset_smiles.csv
affinity_file_path = 'dataset_smiles.csv'  # Replace with your actual file path
affinity_data = pd.read_csv(affinity_file_path)

# Step 9: Merge the cluster assignments with the affinity data
final_data = pd.merge(clustered_data, affinity_data[['Compound', 'SKPT (PEPT2)']], 
                       left_on='Label', right_on='Compound', how='left')

# Step 10: Save to CSV file
output_file_path = 'tanimoto/k_medoid_clusters.csv'  # Replace with your desired output path
final_data.to_csv(output_file_path, index=False)