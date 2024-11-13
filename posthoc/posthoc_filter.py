import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

# Load the CSV file without headers
input_file = "zinc/zinc_prediction/final_top_1000.csv"
df = pd.read_csv(input_file, header=None)

# Manually assign column names (assuming the first column contains SMILES)
df.columns = ['smiles','affinity']

# Define SMARTS for primary amine and carboxyl group
primary_amine_smarts = Chem.MolFromSmarts('[N;H2]')
carboxyl_smarts = Chem.MolFromSmarts('[C](=O)[O]')

def bfs_max_bonds(mol, amine_atoms, carboxyl_atoms):
    visited = set()
    max_bond_count = 0

    for start in amine_atoms:
        queue = [(start, 0)]  # (atom index, bond count)
        visited.add(start)

        while queue:
            current_atom, bond_count = queue.pop(0)

            # Check if we reached a carboxyl atom
            if current_atom in carboxyl_atoms:
                max_bond_count = max(max_bond_count, bond_count)
                continue  # Continue to find more bonds

            # Get the neighbors (connected atoms)
            for neighbor in mol.GetAtomWithIdx(current_atom).GetNeighbors():
                if neighbor.GetIdx() not in visited:
                    visited.add(neighbor.GetIdx())
                    queue.append((neighbor.GetIdx(), bond_count + 1))

    return max_bond_count

def classify_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return 'invalid molecule'
    
    # Generate 3D conformer
    mol = Chem.AddHs(mol)  # Add hydrogens for 3D structure
    if AllChem.EmbedMolecule(mol) != 0:  # Fail to embed a 3D conformer
        return 'failed to embed'
    
    # Optimize the molecule's geometry
    AllChem.UFFOptimizeMolecule(mol)
    
    # Find amine and carboxyl groups
    amine_matches = mol.GetSubstructMatches(primary_amine_smarts)
    carboxyl_matches = mol.GetSubstructMatches(carboxyl_smarts)

    # Debug: Print found matches
    print(f"Found {len(amine_matches)} primary amine matches.")
    print(f"Found {len(carboxyl_matches)} carboxyl matches.")

    # Check the number of amine and carboxyl groups found
    if len(amine_matches) < 1 or len(carboxyl_matches) < 1:
        return f'inhibitor, no sufficient groups found (amines: {len(amine_matches)}, carboxyls: {len(carboxyl_matches)})'

    max_distance = 0
    max_bond_count = 0

    # Calculate the pairwise distances and bonds
    for amine in amine_matches:
        for carboxyl in carboxyl_matches:
            # Get 3D coordinates of the amine and carboxyl groups
            amine_coord = mol.GetConformer().GetAtomPosition(amine[0])
            carboxyl_coord = mol.GetConformer().GetAtomPosition(carboxyl[0])
            
            # Calculate the Euclidean distance between the two groups
            distance = amine_coord.Distance(carboxyl_coord)
            max_distance = max(max_distance, distance)  # Track maximum distance
            
            # Debug: Print distance for each pair
            print(f"Distance between amine and carboxyl: {distance:.2f} Å")
            
            # Count the number of bonds between the amine and carboxyl groups
            bond_count = bfs_max_bonds(mol, [amine[0]], [carboxyl[0]])
            max_bond_count = max(max_bond_count, bond_count)  # Track maximum bond count
            print(f"Number of bonds between amine and carboxyl: {bond_count}")

    # Output classification based on maximum distance and bond count
    if 5.0 <= max_distance <= 10.0:
        if 5 < max_bond_count < 9:
            return f'substrate, distance={max_distance:.2f} Å, bonds={max_bond_count}'
        elif max_bond_count < 5:
            return f'inhibitor, distance={max_distance:.2f} Å, bonds={max_bond_count} (too few bonds)'
        else:
            return f'inhibitor, distance={max_distance:.2f} Å, bonds={max_bond_count} (too many bonds)'
    else:
        if max_distance < 5.0:
            return f'inhibitor, distance={max_distance:.2f} Å (too close together)'
        return f'inhibitor, distance={max_distance:.2f} Å (too far apart)'

# Apply the classification to each molecule in the CSV
df['classification'] = df['smiles'].apply(classify_molecule)

# Save the updated dataframe to a new CSV file
output_file = "posthoc/top_1000_classified.csv"
df.to_csv(output_file, index=False)

print(f"3D distance-based classification complete. Results saved to {output_file}")