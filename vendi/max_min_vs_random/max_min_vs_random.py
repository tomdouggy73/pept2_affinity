import numpy as np
import pandas as pd
import tqdm
from vendi_score import vendi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.DataStructs import FingerprintSimilarity
import random
import matplotlib.pyplot as plt

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# Load the dataset
dataset = pd.read_csv("source_data/dataset_smiles.csv")

# Replace inf values with NaN
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# Get indices where data exists for Caco-2 (PEPT2)
indices = dataset["SKPT (PEPT2)"].dropna().index
compound_smiles = dataset["SMILES"][indices].values

# Generate fingerprints for the 184 compounds
fps = []
rdkit_mols = [Chem.MolFromSmiles(smi) for smi in compound_smiles]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in rdkit_mols]

# Tanimoto similarity function
def tanimoto_similarity(i, j):
    return 1 - FingerprintSimilarity(fps[i], fps[j])

# Function to compute vendi score for a subset of smiles
def calculate_vendi_for_subset(smiles_subset, picker_method, num_picks, num_iterations=1000):
    vendi_scores = []

    for iteration in range(num_iterations):
        # Seed both random libraries for each iteration
        random.seed(iteration)
        np.random.seed(iteration)

        # Select a subset of indices based on the method
        if picker_method == "random":
            # Randomly pick `num_picks` indices
            selected_indices = random.sample(range(len(smiles_subset)), num_picks)
        elif picker_method == "maxmin":
            # Use MaxMinPicker with a random initial pick
            picker = MaxMinPicker()
            initial_pick = random.randint(0, len(smiles_subset) - 1)
            selected_indices = picker.LazyPick(tanimoto_similarity, len(smiles_subset), num_picks, firstPicks=[initial_pick])

        # Get the diverse subset of SMILES strings and corresponding fingerprints
        diverse_fps = [fps[i] for i in selected_indices]

        # Now calculate the similarity matrix for the diverse subset based on fingerprints
        tanimoto_matrix = np.zeros((num_picks, num_picks))

        for i, fp1 in enumerate(diverse_fps):
            for j, fp2 in enumerate(diverse_fps):
                tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)

        # Calculate the Vendi score using the similarity matrix
        vendi_score = vendi.score_K(tanimoto_matrix)
        vendi_scores.append(vendi_score)

    # Calculate the mean and variance of the Vendi scores
    mean_vendi_score = np.mean(vendi_scores)
    variance_vendi_score = np.var(vendi_scores)
    std_error = np.sqrt(variance_vendi_score / num_iterations)

    return mean_vendi_score, variance_vendi_score, std_error

# Prepare to collect results
num_picks_range = list(range(2, 93))   # From 2 to 92 and 184
methods = ["random", "maxmin"]
results = {"num_picks": [], "method": [], "mean_vendi": [], "variance_vendi": [], "std_error": []}

# Iterate for each num_picks size and each method
for num_picks in num_picks_range:
    for method in methods:
        # Calculate vendi score mean, variance, and standard error for this subset and method
        mean_vendi, variance_vendi, std_error = calculate_vendi_for_subset(compound_smiles, method, num_picks)

        # Print results for each iteration
        print(f"Num Picks: {num_picks}, Method: {method}")
        print(f"  Mean Vendi Score: {mean_vendi:.5f}, Variance: {variance_vendi:.5f}")

        # Append the results
        results["num_picks"].append(num_picks)
        results["method"].append(method)
        results["mean_vendi"].append(mean_vendi)
        results["variance_vendi"].append(variance_vendi)
        results["std_error"].append(std_error)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot results
plt.figure(figsize=(10, 6))

# Plot Mean Vendi Score with Standard Error as error bars
for method in methods:
    subset_results = results_df[results_df["method"] == method]
    plt.errorbar(subset_results["num_picks"], subset_results["mean_vendi"], 
                 yerr=subset_results["std_error"], label=method, fmt='-o', capsize=5)

plt.title('Mean Vendi Score vs. Number of Picks with Standard Error')
plt.xlabel('Number of Picks')
plt.ylabel('Mean Vendi Score')
plt.legend()

plt.tight_layout()
plt.show()