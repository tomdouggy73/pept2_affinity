import numpy as np
import pandas as pd

# Load the dataset with cluster assignments
cluster_assignments_file = "tanimoto/hierarch_cluster/output/complete_linkage_with_affinity.csv"
cluster_data = pd.read_csv(cluster_assignments_file)

# Treat all 'inf' values as missing values
cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan)

# Load the dataset containing SMILES information
smiles_file = "dataset_smiles.csv"  # Replace with the actual path to your SMILES dataset
smiles_data = pd.read_csv(smiles_file)

# Ensure the 'Compound' column exists in both datasets
if 'Compound' not in cluster_data.columns or 'Compound' not in smiles_data.columns:
    raise ValueError("Both datasets must contain a 'Compound' column.")

# Merge the cluster assignments with the SMILES data based on the 'Compound' column
merged_data = pd.merge(cluster_data, smiles_data[['Compound', 'SMILES']],
                        on='Compound', how='inner')

print("Merged Data Columns:", merged_data.columns)

# Select relevant columns for the output DataFrame
output_data = merged_data[['Compound', 'SMILES', 'SKPT (PEPT2)', 'Cluster']]

# Save the merged data to a CSV file
output_file_path = 'merged_clusters_with_affinity.csv'  # Specify your output path
output_data.to_csv(output_file_path, index=False)

print("Merged data with SMILES and affinity information saved to 'merged_clusters_with_affinity.csv'.")
