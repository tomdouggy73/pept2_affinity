#%%

from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt
from rdkit.Chem import rdDistGeom

dataset = pd.read_csv("ligand_smiles.dat")

# Treat all 'inf' values as missing values

dataset = dataset.replace([np.inf, -np.inf], np.nan)

smiles = dataset["SMILES"].values[:-1]

#%%
rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]

for n,mol in enumerate(rdkit_mols):
    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMolecule(mol)
    
    with Chem.SDWriter(f"mol{n}.sdf") as w:
        w.write(mol)

