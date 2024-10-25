#%%

from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory

dataset = pd.read_csv("ligand_smiles.dat")

# Treat all 'inf' values as missing values
dataset = dataset.replace([np.inf, -np.inf], np.nan)


smiles = dataset["SMILES"].values[:-1]

rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]



# %%

# Get 2D pharmacophore fingerprints


fdefName = '../regression_2dpharm_randomforest/features.fdef'
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3,trianglePruneBins=False)
sigFactory.skipFeats=['ZnBinder']
sigFactory.SetBins([(0, 2), (2, 5),(5, 8)])
sigFactory.Init()

#%%

pharm2d_fps = []
for n,mol in enumerate(rdkit_mols):
    fp = Generate.Gen2DFingerprint(mol,sigFactory)
    pharm2d_fps.append(fp)

#%%

pharm2d_fps_array = np.array(pharm2d_fps)

pharm2d_fps_df = pd.DataFrame(pharm2d_fps_array)


pharm2d_fps_df.insert(0, "SMILES", smiles)

# write out without header

pharm2d_fps_df.to_csv("ligand_2dpharm.csv", header=False, index=False)

# %%
