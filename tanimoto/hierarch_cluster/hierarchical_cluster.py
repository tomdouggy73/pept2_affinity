import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Tanimoto similarity matrix CSV into a DataFrame to retain labels
file_path = 'tanimoto/matrix/tanimoto_similarity_matrix.csv'
df = pd.read_csv(file_path, index_col=0)

# Step 2: Extract the similarity matrix and labels
similarity_matrix = df.values  # Get the numerical values as a NumPy array
labels = df.index.tolist()      # Get the labels (row names)

# Step 3: Convert similarity matrix to dissimilarity (distance) matrix (1 - similarity)
dissimilarity_matrix = 1 - similarity_matrix

# Step 4: Condense the dissimilarity matrix
condensed_dissimilarity_matrix = squareform(dissimilarity_matrix)

# Step 5: Perform hierarchical clustering using 'complete' linkage on the dissimilarity matrix
Z = linkage(condensed_dissimilarity_matrix, method='complete')

# Step 6: Form exactly 9 clusters
num_clusters = 9
clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

# Output clusters (the cluster assignments for each data point)
print(f"Cluster assignments for 9 clusters: {clusters}")

# Step 7: Create a DataFrame to save the cluster assignments
clustered_data = pd.DataFrame({
    'Label': labels,
    'Cluster': clusters
})

# Step 8: Load the affinity data from dataset_smiles.csv
affinity_file_path = 'dataset_smiles.csv' 
affinity_data = pd.read_csv(affinity_file_path)

# Step 9: Merge the cluster assignments with the affinity data
final_data = pd.merge(clustered_data, affinity_data[['Compound', 'SKPT (PEPT2)']], 
                       left_on='Label', right_on='Compound', how='left')

# Remove column
final_data = final_data.drop(columns=['Label'])

# Step 10: Save to CSV file
output_file_path = 'tanimoto/hierarch_cluster/output/complete_linkage_with_affinity.csv'
final_data.to_csv(output_file_path, index=False)
