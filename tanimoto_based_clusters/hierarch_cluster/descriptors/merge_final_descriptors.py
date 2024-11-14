import pandas as pd

# Load the merged dataset containing SMILES, cluster, and affinity information
data_raw = pd.read_csv('tanimoto/hierarch_cluster/merged_clusters_with_affinity_and_smiles.csv')

# Load the dataset containing Morgan fingerprints
descriptors = pd.read_csv('source_data/dataset_descriptors.csv', header = None)
descriptors.columns = ['SMILES'] + [f'Descriptor_{i}' for i in range(1, descriptors.shape[1])]

# Merge the datasets based on the SMILES column
merged_data_with_fingerprints = pd.merge(data_raw, descriptors, on='SMILES', how='inner')

# Check the columns of the merged dataframe
print("Columns in merged dataset with Morgan fingerprints:")
print(merged_data_with_fingerprints.columns)

# Save the new merged dataset with Morgan fingerprints to a CSV file
merged_output_file = 'tanimoto/hierarch_cluster/descriptors/merged_clusters_complete_descriptors.csv'
merged_data_with_fingerprints.to_csv(merged_output_file, index=False)

print(f"Final merged data with Morgan fingerprints saved to '{merged_output_file}'.")
