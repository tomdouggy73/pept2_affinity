import pandas as pd

# Load the merged dataset containing SMILES, cluster, and affinity information
data_raw = pd.read_csv('merged_clusters_with_affinity.csv')

# Load the dataset containing Morgan fingerprints
morgan_fingerprints = pd.read_csv('dataset_morganfingerprints.csv', header = None)
morgan_fingerprints.columns = ['SMILES'] + [f'Fingerprint_{i}' for i in range(1, morgan_fingerprints.shape[1])]

# Merge the datasets based on the SMILES column
merged_data_with_fingerprints = pd.merge(data_raw, morgan_fingerprints, on='SMILES', how='inner')

# Check the columns of the merged dataframe
print("Columns in merged dataset with Morgan fingerprints:")
print(merged_data_with_fingerprints.columns)

# Save the new merged dataset with Morgan fingerprints to a CSV file
merged_output_file = 'merged_clusters_complete.csv'
merged_data_with_fingerprints.to_csv(merged_output_file, index=False)

print(f"Final merged data with Morgan fingerprints saved to '{merged_output_file}'.")

