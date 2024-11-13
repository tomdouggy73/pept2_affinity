import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity

# Filter for PepT2 and generate fingerprint
dataset = pd.read_csv("dataset_smiles.csv")

dataset = dataset.replace([np.inf, -np.inf], np.nan)

indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)

rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]

# Calculate Morgan fingerprints (radius 2 and bit size 2048)
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in rdkit_mols]

# Generate pairwise matrix
tanimoto_matrix = np.zeros((len(fingerprints), len(fingerprints)))

for i, fp1 in enumerate(fingerprints):
    for j, fp2 in enumerate(fingerprints):
        tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)

# Convert the Tanimoto matrix to a DataFrame for easier handling
tanimoto_df = pd.DataFrame(tanimoto_matrix, index=compounds_names, columns=compounds_names)

# Display the Tanimoto coefficients
print(tanimoto_df)

# Save the Tanimoto similarity matrix to a CSV file
tanimoto_df.to_csv("tanimoto_similarity_matrix.csv")
