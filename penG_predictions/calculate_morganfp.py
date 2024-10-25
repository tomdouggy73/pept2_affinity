#%%
 
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt


dataset = pd.read_csv("ligand_smiles.dat")

# Treat all 'inf' values as missing values
dataset = dataset.replace([np.inf, -np.inf], np.nan)

smiles = dataset["SMILES"].values[:-1]

#%%
rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]


#%%

# also briefly get logP predicitons out

for mol in rdkit_mols:
    print(Descriptors.MolLogP(mol))

# %%

# Get Morgan fingerprints

morgan_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in rdkit_mols]

morgan_fps_array = np.array(morgan_fps)

# combined wtih the smiles into a dataframe and write out

morgan_fps_df = pd.DataFrame(morgan_fps_array)

morgan_fps_df.insert(0, "SMILES", smiles)

# write out without header

morgan_fps_df.to_csv("ligand_morganfingerprints.csv", header=False, index=False)

# %%
