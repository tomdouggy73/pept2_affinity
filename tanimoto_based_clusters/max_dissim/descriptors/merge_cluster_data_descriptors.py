import pandas as pd

# Load the initial datasets
seed_affinity = pd.read_csv('tanimoto/max_dissim/seed_101855_affinity.csv')
dataset_smiles = pd.read_csv('source_data/dataset_smiles.csv')
dataset_descriptors = pd.read_csv('source_data/dataset_descriptors.csv', header=None)

# Name the columns in the morgan_fingerprints dataset, assuming the first column is SMILES
dataset_descriptors.columns = ['SMILES'] + [f'Descriptor_{i}' for i in range(1, dataset_descriptors.shape[1])]

# Step 1: Merge `seed_101855_affinity` with `dataset_smiles` on the 'Compound' column
merged_data = pd.merge(seed_affinity, dataset_smiles, on='Compound', how='inner')

# Step 2: Merge the result with `dataset_morganfingerprints` on the 'SMILES' column
final_merged_data = pd.merge(merged_data, dataset_descriptors, on='SMILES', how='inner')

# Drop any unnamed columns that may have been added
final_merged_data = final_merged_data.loc[:, ~final_merged_data.columns.str.contains('^Unnamed')]

# Check the columns of the final merged dataframe
print("Columns in the final merged dataset:")
print(final_merged_data.columns)

# Save the new merged dataset to a CSV file
final_output_file = 'tanimoto/max_dissim/descriptors/final_merged_data_with_descriptors.csv'
final_merged_data.to_csv(final_output_file, index=False)

print(f"Final merged data with descriptors saved to '{final_output_file}'.")
