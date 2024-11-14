import pandas as pd
import numpy as np
from vendi_score import vendi
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# Load the CSV with pandas
dataset = pd.read_csv(
    'tanimoto_based_clusters/hierarch_cluster/morgan_fingerprints/merged_clusters_complete_morgan_fingerprints.csv')

# Data preprocessing
indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)
clusters = dataset['Cluster'].values

unique_clusters = np.unique(clusters)

cluster_matrix, vendi_scores = [], []

for test_cluster in unique_clusters:
    test_indices = np.where(clusters == test_cluster)[0]
    test_smiles = [smiles[idx] for idx in test_indices]
    rdkit_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]

    # Calculate Tanimoto matrix for the test SMILES
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in rdkit_mols]

    tanimoto_matrix = np.zeros((len(fps), len(fps)))
    for i, fp1 in enumerate(fps):
        for j, fp2 in enumerate(fps):
            tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)

    vendi_score_test = vendi.score_K(tanimoto_matrix)
    vendi_scores.append(vendi_score_test)

print("Vendi Scores:", vendi_scores)

vendi_variance = np.var(vendi_scores)  
print(f"Vendi variance: {vendi_variance:.5f}")