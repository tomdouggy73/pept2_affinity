
#%%
import numpy as np
import datasets
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# suppress deprecation warnings from RDkit

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ds = load_dataset("sagawa/ZINC-canonicalized")

combined_dataset = pd.concat([ds['train'].to_pandas(),ds['validation'].to_pandas()])

output_folder = "mf_zinc_chunks"
os.makedirs(output_folder, exist_ok=True)

# loop through each chunk

start_chunk = 0
NUM_CHUNKS = 100
for chunk in range(start_chunk, NUM_CHUNKS):
    print(f"Processing chunk {chunk}")

    # calculate index limits
    
    lower_lim = int(chunk * len(combined_dataset['smiles']) / NUM_CHUNKS)
    upper_lim = int((chunk + 1) * len(combined_dataset['smiles']) / NUM_CHUNKS)

    # Convert SMILES to RDKit mols with error handling
    
    rdkit_mols = []
    invalid_indices = []
    for i, smi in enumerate(combined_dataset['smiles'][lower_lim:upper_lim]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_indices.append(lower_lim + i)  
            
    # Store the global index of the invalid SMILES
    
        else:
            rdkit_mols.append(mol)

    # Output invalid SMILES indices
    
    if invalid_indices:
        print(f"Invalid SMILES encountered at indices: {invalid_indices}")

    # Generate Morgan fingerprints only for valid molecules
    
    morgan_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in rdkit_mols]
    morgan_fps_array = np.array(morgan_fps)

    # combined with the smiles into a dataframe and write out

    morgan_fps_df = pd.DataFrame(morgan_fps_array)

  # Prepare the valid SMILES list

    valid_smiles = [smi for i, smi in enumerate(combined_dataset['smiles'][lower_lim:upper_lim]) if lower_lim + i not in invalid_indices]
    morgan_fps_df.insert(0, "SMILES", valid_smiles)

    # Write out without header into output file
    
    output_file = os.path.join(output_folder, f"ligand_morganfingerprints_chunk{chunk}.csv")
    morgan_fps_df.to_csv(output_file, header=False, index=False)

    print(f"Chunk {chunk} saved to {output_file}")

# %%
