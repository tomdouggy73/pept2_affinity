import numpy as np
import pandas as pd
from vendi_score import vendi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.DataStructs import FingerprintSimilarity
import random

# Load dataset and prepare compounds
dataset = pd.read_csv("source_data/dataset_smiles.csv")

# Replace inf values with NaN and get indices for available data
dataset = dataset.replace([np.inf, -np.inf], np.nan)
compound_smiles = dataset["SMILES"].dropna().values

# Generate fingerprints
rdkit_mols = [Chem.MolFromSmiles(smi) for smi in compound_smiles]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in rdkit_mols]

# Define similarity function for picking
def tanimoto_similarity(i, j):
    return 1 - FingerprintSimilarity(fps[i], fps[j])

# Specify the cluster size and indices
cluster_size = 18

# Function to select test indices and calculate vendi score for one iteration
def calculate_vendi_for_cluster(smiles, picker_method, cluster_size):
    # Generate test indices based on the picker method
    if picker_method == "random":
        test_indices = random.sample(range(len(smiles)), cluster_size)
    elif picker_method == "maxmin":
        picker = MaxMinPicker()
        initial_pick = random.randint(0, len(smiles) - 1)
        test_indices = picker.LazyPick(tanimoto_similarity, len(smiles), cluster_size, firstPicks=[initial_pick])

    # Get fingerprints for selected compounds
    selected_fps = [fps[i] for i in test_indices]

    # Calculate Tanimoto similarity matrix for the selected compounds
    tanimoto_matrix = np.zeros((cluster_size, cluster_size))
    for i, fp1 in enumerate(selected_fps):
        for j, fp2 in enumerate(selected_fps):
            tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)

    # Calculate Vendi score
    vendi_score = vendi.score_K(tanimoto_matrix)

    return test_indices, vendi_score

# Prepare to save results for both methods
methods = ["random", "maxmin"]
for method in methods:
    # Get test indices and vendi score
    test_indices, vendi_score = calculate_vendi_for_cluster(compound_smiles, method, cluster_size)
    
    # Label compounds by cluster
    cluster_labels = ["1" if i in test_indices else "2" for i in range(len(compound_smiles))]
    
    # Create a DataFrame with SMILES and cluster labels
    results_df = pd.DataFrame({
        "SMILES": compound_smiles,
        "Cluster": cluster_labels
    })
    
    # Save the DataFrame to a CSV file with a unique name for each method
    results_filename = f"vendi_cluster_{method}.csv"
    results_df.to_csv(results_filename, index=False)
    
    # Print the method and vendi score
    print(f"Method: {method}, Vendi Score: {vendi_score}")
    print(f"Results saved to {results_filename}")