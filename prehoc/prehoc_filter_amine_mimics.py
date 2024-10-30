import pandas as pd
from rdkit import Chem
from tqdm import tqdm  # For progress tracking

# Load dataset
from datasets import load_dataset
ds = load_dataset("sagawa/ZINC-canonicalized")
combined_dataset = pd.concat([ds['train'].to_pandas(), ds['validation'].to_pandas()])

# Define SMARTS patterns for carboxyl and amine mimic groups
patterns = {
    'carboxyl': Chem.MolFromSmarts('C(=O)[O]'),
    'amidine': Chem.MolFromSmarts('C(=N)N'),
    'guanidine': Chem.MolFromSmarts('NC(=N)N'),
    'imidazole': Chem.MolFromSmarts('c1ncnc1'),
}

# Output file setup
output_file = "prehoc/classified_molecules_candidates.csv"
with open(output_file, 'w') as f:
    f.write('smiles,classifications\n')  # Write header

def classify_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Skip invalid molecules
    
    classifications = []
    has_carboxyl = mol.HasSubstructMatch(patterns['carboxyl'])

    # Check for at least one amine mimic group
    for name, smarts in patterns.items():
        if name != 'carboxyl' and mol.HasSubstructMatch(smarts):
            classifications.append(name)

    # Return classifications only if both a carboxyl group and an amine mimic group are found
    if has_carboxyl and classifications:
        return ', '.join(classifications)
    return None  

# Apply the SMARTS search and classification
filtered_data = []
for smiles in tqdm(combined_dataset['smiles'], desc="Classifying molecules"):
    classification = classify_smiles(smiles)
    if classification:  # Only keep candidates with both groups
        filtered_data.append({'smiles': smiles, 'classification': classification})

# Save filtered results to CSV
result_df = pd.DataFrame(filtered_data)
result_df.to_csv(output_file, index=False)
print(f"Classification complete. Results saved to {output_file}.")