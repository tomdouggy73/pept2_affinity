import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, GraphDescriptors, Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors
import collections.abc as collections
import argparse

def flatten(x):
    """
    Flatten a nested list.

    Args:
        x (iterable): The nested list to be flattened.

    Returns:
        list: The flattened list.
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def smiles2mol(smi):
    """
    Convert a SMILES string to an RDKit molecule.

    Args:
        smi (str): The SMILES string.

    Returns:
        Chem.Mol: The RDKit molecule.
    """
    return Chem.MolFromSmiles(smi)


def smiles2mols(smiles):
    """
    Convert a list of SMILES strings to RDKit molecules.

    Args:
        smiles (list): The list of SMILES strings.

    Returns:
        numpy.ndarray: Array of RDKit molecules.
    """
    vect_smiles2mol = np.vectorize(smiles2mol)
    return vect_smiles2mol(smiles)


def addh_mol(mol):
    """
    Add hydrogens to an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        Chem.Mol: The RDKit molecule with added hydrogens.
    """
    return Chem.AddHs(mol)


def addh_mols(mols):
    """
    Add hydrogens to a list of RDKit molecules.

    Args:
        mols (list): The list of RDKit molecules.

    Returns:
        numpy.ndarray: Array of RDKit molecules with added hydrogens.
    """
    vect_addh_mol = np.vectorize(addh_mol)
    return vect_addh_mol(mols)


def gen_3d_mol(mol):
    """
    Generate 3D coordinates for an RDKit molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        Chem.Mol: The RDKit molecule with 3D coordinates.
    """
    return AllChem.EmbedMolecule(mol, useRandomCoords=True)


def gen_3d_mols(mols):
    """
    Generate 3D coordinates for a list of RDKit molecules.

    Args:
        mols (list): The list of RDKit molecules.

    Returns:
        numpy.ndarray: Array of RDKit molecules with 3D coordinates.
    """
    vect_t3d_mol = np.vectorize(gen_3d_mol)
    return vect_t3d_mol(mols)


def optimise_mol(mol):
    """
    Optimize an RDKit molecule using UFF force field.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        Chem.Mol: The optimized RDKit molecule.
    """
    return AllChem.UFFOptimizeMolecule(mol, maxIters=1000)


def optimise_mols(mols):
    """
    Optimize a list of RDKit molecules using UFF force field.

    Args:
        mols (list): The list of RDKit molecules.

    Returns:
        numpy.ndarray: Array of optimized RDKit molecules.
    """
    vec_uff = np.vectorize(optimise_mol)
    return vec_uff(mols)


def compute_descs(mol):
    """
    Calculate a set of standard RDKit molecular descriptors.

    Args:
        mol (Chem.Mol): The RDKit molecule.

    Returns:
        list: List of calculated molecular descriptors.
    """
    std_descriptors = []
    m = mol
    desc = []

    # Low dimensional descriptors
    desc.append(Chem.GraphDescriptors.BalabanJ(m))        
    desc.append(Chem.GraphDescriptors.BertzCT(m))         
    desc.append(Chem.GraphDescriptors.HallKierAlpha(m))   
    desc.append(Chem.Crippen.MolLogP(m))                  
    desc.append(Chem.Crippen.MolMR(m))                    
    desc.append(Chem.Descriptors.ExactMolWt(m))
    desc.append(Chem.Descriptors.FpDensityMorgan1(m))
    desc.append(Chem.Descriptors.MaxPartialCharge(m))
    desc.append(Chem.Descriptors.NumRadicalElectrons(m))
    desc.append(Chem.Descriptors.NumValenceElectrons(m))
    desc.append(Chem.Lipinski.FractionCSP3(m))
    desc.append(Chem.Lipinski.HeavyAtomCount(m))
    desc.append(Chem.Lipinski.NHOHCount(m))
    desc.append(Chem.Lipinski.NOCount(m))
    desc.append(Chem.MolSurf.LabuteASA(m))
    desc.append(Chem.rdMolDescriptors.CalcAsphericity(m))
    desc.append(Chem.rdMolDescriptors.CalcChi0n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi0v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi1n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi1v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi2n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi2v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi3v(m))
    desc.append(Chem.rdMolDescriptors.CalcChi4n(m))
    desc.append(Chem.rdMolDescriptors.CalcChi4v(m))
    desc.append(Chem.rdMolDescriptors.CalcCrippenDescriptors(m))
    desc.append(Chem.rdMolDescriptors.CalcEccentricity(m))
    desc.append(Chem.rdMolDescriptors.CalcFractionCSP3(m))
    desc.append(Chem.rdMolDescriptors.CalcHallKierAlpha(m))
    desc.append(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa1(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa2(m))
    desc.append(Chem.rdMolDescriptors.CalcKappa3(m))
    desc.append(Chem.rdMolDescriptors.CalcLabuteASA(m))
    desc.append(Chem.rdMolDescriptors.CalcNPR1(m))
    desc.append(Chem.rdMolDescriptors.CalcNPR2(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAmideBonds(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHBA(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHBD(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHeteroatoms(m))
    desc.append(Chem.rdMolDescriptors.CalcNumHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))
    desc.append(Chem.rdMolDescriptors.CalcNumRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedRings(m))
    desc.append(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))
    desc.append(Chem.rdMolDescriptors.CalcPBF(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI1(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI2(m))
    desc.append(Chem.rdMolDescriptors.CalcPMI3(m))
    desc.append(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))
    desc.append(Chem.rdMolDescriptors.CalcSpherocityIndex(m))
    desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
    
    # High dimensional descriptors
    desc.append(Chem.rdMolDescriptors.CalcWHIM(m))
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR2D(m)) 
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR3D(m)) 
    desc.append(Chem.rdMolDescriptors.CalcRDF(m)) 
    desc.append(Chem.rdMolDescriptors.CalcMORSE(m)) 

    desc = flatten(desc)
    std_descriptors.append(desc)

    return std_descriptors

def parse_args():

    parser = argparse.ArgumentParser(description="Script for generating standard descriptors from compound SMILES.")
    parser.add_argument("infile", help="Input file path (.CSV with smiles header)")
    parser.add_argument("outfile", help="Output file path (.CSV)")
    args = parser.parse_args()

    return args.infile, args.outfile

def load_data(inpath):

    data = pd.read_csv(inpath)
    smiles = data['SMILES']

    return smiles

def main():

    inpath, outpath = parse_args()
    smiles = load_data(inpath)

    mols = smiles2mols(smiles)
    mols_H = addh_mols(mols)
    gen_3d_mols(mols_H)
    optimise_mols(mols_H)

    std_descs = []

    for m in mols_H:
        std_desc = compute_descs(m)[0]
        std_descs.append(std_desc)
    
    std_df = pd.DataFrame(data=std_descs, index=smiles)
    std_df.to_csv(outpath, header=False)

if __name__=='__main__':
    main()
