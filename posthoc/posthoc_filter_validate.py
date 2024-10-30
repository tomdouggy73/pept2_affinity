import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load dataset
dataset = pd.read_csv("source_data/dataset_smiles.csv")
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# Get indices and values where "SKPT (PEPT2)" data is available
indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compound_names = dataset["Compound"][indices].values

# Define SMARTS for primary amine, amide, carboxyl, and carbonyl groups
primary_amine_smarts = Chem.MolFromSmarts('[N;H2]')
amide_smarts = Chem.MolFromSmarts('C(=O)N')
carboxyl_smarts = Chem.MolFromSmarts('[C](=O)[O]')
carbonyl_smarts = Chem.MolFromSmarts('C(=O)')

# BFS function to calculate maximum bond count between groups
def bfs_max_bonds(mol, group1_atoms, group2_atoms):
    visited = set()
    max_bond_count = 0
    for start in group1_atoms:
        queue = [(start, 0)]
        visited.add(start)
        while queue:
            current_atom, bond_count = queue.pop(0)
            if current_atom in group2_atoms:
                max_bond_count = max(max_bond_count, bond_count)
                continue
            for neighbor in mol.GetAtomWithIdx(current_atom).GetNeighbors():
                if neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    queue.append((neighbor.GetIdx(), bond_count + 1))
    return max_bond_count

# Classification function
def classify_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'invalid molecule'
    
    mol = Chem.AddHs(mol)  # Add hydrogens
    if AllChem.EmbedMolecule(mol) != 0:
        return 'failed to embed'
    
    AllChem.UFFOptimizeMolecule(mol)
    
    # Find matches for primary amine, amide, carboxyl, and carbonyl groups
    amine_matches = mol.GetSubstructMatches(primary_amine_smarts)
    amide_matches = mol.GetSubstructMatches(amide_smarts)
    carboxyl_matches = mol.GetSubstructMatches(carboxyl_smarts)
    carbonyl_matches = mol.GetSubstructMatches(carbonyl_smarts)

    # Check for at least one amine or amide and one carboxyl or carbonyl group
    if (len(amine_matches) + len(amide_matches)) < 1 or (len(carboxyl_matches) + len(carbonyl_matches)) < 1:
        return f'inhibitor, insufficient groups (amines: {len(amine_matches)}, amides: {len(amide_matches)}, carboxyls: {len(carboxyl_matches)}, carbonyls: {len(carbonyl_matches)})'
    
    max_distance = 0
    max_bond_count = 0
    
    # Create a combined list of carboxyl and carbonyl matches
    combined_matches = list(carboxyl_matches) + list(carbonyl_matches)

    # Check distances and bond counts for amine matches
    for amine in amine_matches:
        for carboxyl_or_carbonyl in combined_matches:
            amine_coord = mol.GetConformer().GetAtomPosition(amine[0])
            carbonyl_or_carboxyl_coord = mol.GetConformer().GetAtomPosition(carboxyl_or_carbonyl[0])
            distance = amine_coord.Distance(carbonyl_or_carboxyl_coord)
            max_distance = max(max_distance, distance)
            bond_count = bfs_max_bonds(mol, [amine[0]], [carboxyl_or_carbonyl[0]])
            max_bond_count = max(max_bond_count, bond_count)

    # Check distances and bond counts for amide matches
    for amide in amide_matches:
        for carboxyl_or_carbonyl in combined_matches:
            amide_coord = mol.GetConformer().GetAtomPosition(amide[0])
            carbonyl_or_carboxyl_coord = mol.GetConformer().GetAtomPosition(carboxyl_or_carbonyl[0])
            distance = amide_coord.Distance(carbonyl_or_carboxyl_coord)
            max_distance = max(max_distance, distance)
            bond_count = bfs_max_bonds(mol, [amide[0]], [carboxyl_or_carbonyl[0]])
            max_bond_count = max(max_bond_count, bond_count)

    # Output classification based on distance and bond count
    if 3.0 <= max_distance <= 10.0:
        if 3 <= max_bond_count <= 9:
            return f'substrate, distance={max_distance:.2f} Å, bonds={max_bond_count}'
        elif max_bond_count < 3:
            return f'inhibitor, distance={max_distance:.2f} Å, bonds={max_bond_count} (too few bonds)'
        else:
            return f'inhibitor, distance={max_distance:.2f} Å, bonds={max_bond_count} (too many bonds)'
    else:
        if max_distance < 3.0:
            return f'inhibitor, distance={max_distance:.2f} Å (too close)'
        return f'inhibitor, distance={max_distance:.2f} Å (too far)'

# Apply classification function to the filtered rows only
classifications = [classify_molecule(smi) for smi in smiles]

# Create a DataFrame for the classified results, aligning them with original indices
classified_df = pd.DataFrame({'classification': classifications}, index=indices)

# Merge classifications back into the full DataFrame (unclassified rows get NaN)
dataset['classification'] = classified_df['classification']

# Filter dataset to include only necessary columns and the classification
filtered_dataset = dataset.loc[indices, ["Compound", "SMILES", "SKPT (PEPT2)", "classification"]]

# Save to CSV with just the required columns for the specified rows
output_file = "posthoc/classified_molecules_validation.csv"
filtered_dataset.to_csv(output_file, index=False)

print(f"Classification complete. Results saved to {output_file}")