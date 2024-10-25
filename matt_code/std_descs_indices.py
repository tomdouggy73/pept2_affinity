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

    desc_names = []

    # Low dimensional descriptors
    desc.append(Chem.GraphDescriptors.BalabanJ(m))
    if not isinstance(Chem.GraphDescriptors.BalabanJ(m), float):
        for i in range(len(Chem.GraphDescriptors.BalabanJ(m))):
            desc_names.append('BalabanJ (Value' + str(i)+" )") 
    else:
        desc_names.append('BalabanJ')
    desc.append(Chem.GraphDescriptors.BertzCT(m))         
    if not isinstance(Chem.GraphDescriptors.BertzCT(m), float):
        for i in range(len(Chem.GraphDescriptors.BertzCT(m))):
            desc_names.append('BertzCT (Value' + str(i)+" )")
    else:
        desc_names.append('BertzCT')
    desc.append(Chem.GraphDescriptors.HallKierAlpha(m))
    if not isinstance(Chem.GraphDescriptors.HallKierAlpha(m), float):
        for i in range(len(Chem.GraphDescriptors.HallKierAlpha(m))):
            desc_names.append('HallKierAlpha (Value' + str(i)+" )")
    else:
        desc_names.append('HallKierAlpha')
    desc.append(Chem.Crippen.MolLogP(m))    
    if not isinstance(Chem.Crippen.MolLogP(m), float):
        for i in range(len(Chem.Crippen.MolLogP(m))):
            desc_names.append('MolLogP (Value' + str(i)+" )")
    else:
        desc_names.append('MolLogP')              
    desc.append(Chem.Crippen.MolMR(m))    
    if not isinstance(Chem.Crippen.MolMR(m), float):
        for i in range(len(Chem.Crippen.MolMR(m))):
            desc_names.append('MolMR (Value' + str(i)+" )")
    else:
        desc_names.append('MolMR')                
    desc.append(Chem.Descriptors.ExactMolWt(m))
    if not isinstance(Chem.Descriptors.ExactMolWt(m), float):
        for i in range(len(Chem.Descriptors.ExactMolWt(m))):
            desc_names.append('ExactMolWt (Value' + str(i)+" )")
    else:
        desc_names.append('ExactMolWt')
    desc.append(Chem.Descriptors.FpDensityMorgan1(m))
    if not isinstance(Chem.Descriptors.FpDensityMorgan1(m), float):
        for i in range(len(Chem.Descriptors.FpDensityMorgan1(m))):
            desc_names.append('FpDensityMorgan1 (Value' + str(i)+" )")
    else:
        desc_names.append('FpDensityMorgan1')
    desc.append(Chem.Descriptors.MaxPartialCharge(m))
    if not isinstance(Chem.Descriptors.MaxPartialCharge(m), float):
        for i in range(len(Chem.Descriptors.MaxPartialCharge(m))):
            desc_names.append('MaxPartialCharge (Value' + str(i)+" )")
    else:
        desc_names.append('MaxPartialCharge')
    desc.append(Chem.Descriptors.NumRadicalElectrons(m))
    try:
        for i in range(len(Chem.Descriptors.NumRadicalElectrons(m))):
            desc_names.append('NumRadicalElectrons (Value' + str(i)+" )")
    except:
        desc_names.append('NumRadicalElectrons')
    desc.append(Chem.Descriptors.NumValenceElectrons(m))
    try:
        for i in range(len(Chem.Descriptors.NumValenceElectrons(m))):
            desc_names.append('NumValenceElectrons (Value' + str(i)+" )")
    except:
        desc_names.append('NumValenceElectrons')
    desc.append(Chem.Lipinski.FractionCSP3(m))
    try:
        for i in range(len(Chem.Lipinski.FractionCSP3(m))):
            desc_names.append('FractionCSP3 (Value' + str(i)+" )")
    except:
        desc_names.append('FractionCSP3')
    desc.append(Chem.Lipinski.HeavyAtomCount(m))
    try:
        for i in range(len(Chem.Lipinski.HeavyAtomCount(m))):
            desc_names.append('HeavyAtomCount (Value' + str(i)+" )")
    except:
        desc_names.append('HeavyAtomCount')
    desc.append(Chem.Lipinski.NHOHCount(m))
    try:
        for i in range(len(Chem.Lipinski.NHOHCount(m))):
            desc_names.append('NHOHCount (Value' + str(i)+" )")
    except:
        desc_names.append('NHOHCount')
    desc.append(Chem.Lipinski.NOCount(m))
    try:
        for i in range(len(Chem.Lipinski.NOCount(m))):
            desc_names.append('NOCount (Value' + str(i)+" )")
    except:
        desc_names.append('NOCount')
    desc.append(Chem.MolSurf.LabuteASA(m))
    try:
        for i in range(len(Chem.MolSurf.LabuteASA(m))):
            desc_names.append('LabuteASA (Value' + str(i)+" )")
    except:
        desc_names.append('LabuteASA')
    desc.append(Chem.rdMolDescriptors.CalcAsphericity(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcAsphericity(m))):
            desc_names.append('Asphericity (Value' + str(i)+" )")
    except:
        desc_names.append('Asphericity')
    desc.append(Chem.rdMolDescriptors.CalcChi0n(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi0n(m))):
            desc_names.append('Chi0n (Value' + str(i)+" )")
    except:
        desc_names.append('Chi0n')
    desc.append(Chem.rdMolDescriptors.CalcChi0v(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi0v(m))):
            desc_names.append('Chi0v (Value' + str(i)+" )")
    except:
        desc_names.append('Chi0v')
    desc.append(Chem.rdMolDescriptors.CalcChi1n(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi1n(m))):
            desc_names.append('Chi1n (Value' + str(i)+" )")
    except:
        desc_names.append('Chi1n')
    desc.append(Chem.rdMolDescriptors.CalcChi1v(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi1v(m))):
            desc_names.append('Chi1v (Value' + str(i)+" )")
    except:
        desc_names.append('Chi1v')
    desc.append(Chem.rdMolDescriptors.CalcChi2n(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi2n(m))):
            desc_names.append('Chi2n (Value' + str(i)+" )")
    except:
        desc_names.append('Chi2n')
    desc.append(Chem.rdMolDescriptors.CalcChi2v(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi2v(m))):
            desc_names.append('Chi2v (Value' + str(i)+" )")
    except:
        desc_names.append('Chi2v')
    desc.append(Chem.rdMolDescriptors.CalcChi3v(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi3v(m))):
            desc_names.append('Chi3v (Value' + str(i)+" )")
    except:
        desc_names.append('Chi3v')
    desc.append(Chem.rdMolDescriptors.CalcChi4n(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi4n(m))):
            desc_names.append('Chi4n (Value' + str(i)+" )")
    except:
        desc_names.append('Chi4n')
    desc.append(Chem.rdMolDescriptors.CalcChi4v(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcChi4v(m))):
            desc_names.append('Chi4v (Value' + str(i)+" )")
    except:
        desc_names.append('Chi4v')
    desc.append(Chem.rdMolDescriptors.CalcCrippenDescriptors(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcCrippenDescriptors(m))):
            desc_names.append('CrippenDescriptors (Value' + str(i)+" )")
    except:
        desc_names.append('CrippenDescriptors')
    desc.append(Chem.rdMolDescriptors.CalcEccentricity(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcEccentricity(m))):
            desc_names.append('Eccentricity (Value' + str(i)+" )")
    except:
        desc_names.append('Eccentricity')
    desc.append(Chem.rdMolDescriptors.CalcFractionCSP3(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcFractionCSP3(m))):
            desc_names.append('FractionCSP3 (Value' + str(i)+" )")
    except:
        desc_names.append('FractionCSP3')
    desc.append(Chem.rdMolDescriptors.CalcHallKierAlpha(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcHallKierAlpha(m))):
            desc_names.append('HallKierAlpha (Value' + str(i)+" )")
    except:
        desc_names.append('HallKierAlpha')
    desc.append(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcInertialShapeFactor(m))):
            desc_names.append('InertialShapeFactor (Value' + str(i)+" )")
    except:
        desc_names.append('InertialShapeFactor')
    desc.append(Chem.rdMolDescriptors.CalcKappa1(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcKappa1(m))):
            desc_names.append('Kappa1 (Value' + str(i)+" )")
    except:
        desc_names.append('Kappa1')
    desc.append(Chem.rdMolDescriptors.CalcKappa2(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcKappa2(m))):
            desc_names.append('Kappa2 (Value' + str(i)+" )")
    except:
        desc_names.append('Kappa2')
    desc.append(Chem.rdMolDescriptors.CalcKappa3(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcKappa3(m))):
            desc_names.append('Kappa3 (Value' + str(i)+" )")
    except:
        desc_names.append('Kappa3')
    desc.append(Chem.rdMolDescriptors.CalcLabuteASA(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcLabuteASA(m))):
            desc_names.append('LabuteASA (Value' + str(i)+" )")
    except:
        desc_names.append('LabuteASA')
    desc.append(Chem.rdMolDescriptors.CalcNPR1(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNPR1(m))):
            desc_names.append('NPR1 (Value' + str(i)+" )")
    except:
        desc_names.append('NPR1')
    desc.append(Chem.rdMolDescriptors.CalcNPR2(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNPR2(m))):
            desc_names.append('NPR2 (Value' + str(i)+" )")
    except:
        desc_names.append('NPR2')
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAliphaticCarbocycles(m))):
            desc_names.append('NumAliphaticCarbocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumAliphaticCarbocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAliphaticHeterocycles(m))):
            desc_names.append('NumAliphaticHeterocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumAliphaticHeterocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumAliphaticRings(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAliphaticRings(m))):
            desc_names.append('NumAliphaticRings (Value' + str(i)+" )")
    except:
        desc_names.append('NumAliphaticRings')
    desc.append(Chem.rdMolDescriptors.CalcNumAmideBonds(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAmideBonds(m))):
            desc_names.append('NumAmideBonds (Value' + str(i)+" )")
    except:
        desc_names.append('NumAmideBonds')
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAromaticCarbocycles(m))):
            desc_names.append('NumAromaticCarbocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumAromaticCarbocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAromaticHeterocycles(m))):
            desc_names.append('NumAromaticHeterocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumAromaticHeterocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumAromaticRings(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumAromaticRings(m))):
            desc_names.append('NumAromaticRings (Value' + str(i)+" )")
    except:
        desc_names.append('NumAromaticRings')
    desc.append(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(m))):
            desc_names.append('NumBridgeheadAtoms (Value' + str(i)+" )")
    except:
        desc_names.append('NumBridgeheadAtoms')
    desc.append(Chem.rdMolDescriptors.CalcNumHBA(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumHBA(m))):
            desc_names.append('NumHBA (Value' + str(i)+" )")
    except:
        desc_names.append('NumHBA')
    desc.append(Chem.rdMolDescriptors.CalcNumHBD(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumHBD(m))):
            desc_names.append('NumHBD (Value' + str(i)+" )")
    except:
        desc_names.append('NumHBD')
    desc.append(Chem.rdMolDescriptors.CalcNumHeteroatoms(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumHeteroatoms(m))):
            desc_names.append('NumHeteroatoms (Value' + str(i)+" )")
    except:
        desc_names.append('NumHeteroatoms')
    desc.append(Chem.rdMolDescriptors.CalcNumHeterocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumHeterocycles(m))):
            desc_names.append('NumHeterocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumHeterocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumLipinskiHBA(m))):
            desc_names.append('NumLipinskiHBA (Value' + str(i)+" )")
    except:
        desc_names.append('NumLipinskiHBA')
    desc.append(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumLipinskiHBD(m))):
            desc_names.append('NumLipinskiHBD (Value' + str(i)+" )")
    except:
        desc_names.append('NumLipinskiHBD')
    desc.append(Chem.rdMolDescriptors.CalcNumRings(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumRings(m))):
            desc_names.append('NumRings (Value' + str(i)+" )")
    except:
        desc_names.append('NumRings')
    desc.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumRotatableBonds(m))):
            desc_names.append('NumRotatableBonds (Value' + str(i)+" )")
    except:
        desc_names.append('NumRotatableBonds')
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumSaturatedCarbocycles(m))):
            desc_names.append('NumSaturatedCarbocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumSaturatedCarbocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumSaturatedHeterocycles(m))):
            desc_names.append('NumSaturatedHeterocycles (Value' + str(i)+" )")
    except:
        desc_names.append('NumSaturatedHeterocycles')
    desc.append(Chem.rdMolDescriptors.CalcNumSaturatedRings(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumSaturatedRings(m))):
            desc_names.append('NumSaturatedRings (Value' + str(i)+" )")
    except:
        desc_names.append('NumSaturatedRings')
    desc.append(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcNumSpiroAtoms(m))):
            desc_names.append('NumSpiroAtoms (Value' + str(i)+" )")
    except:
        desc_names.append('NumSpiroAtoms')
    desc.append(Chem.rdMolDescriptors.CalcPBF(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcPBF(m))):
            desc_names.append('PBF (Value' + str(i)+" )")
    except:
        desc_names.append('PBF')
    desc.append(Chem.rdMolDescriptors.CalcPMI1(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcPMI1(m))):
            desc_names.append('PMI1 (Value' + str(i)+" )")
    except:
        desc_names.append('PMI1')
    desc.append(Chem.rdMolDescriptors.CalcPMI2(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcPMI2(m))):
            desc_names.append('PMI2 (Value' + str(i)+" )")
    except:
        desc_names.append('PMI2')
    desc.append(Chem.rdMolDescriptors.CalcPMI3(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcPMI3(m))):
            desc_names.append('PMI3 (Value' + str(i)+" )")
    except:
        desc_names.append('PMI3')
    desc.append(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcRadiusOfGyration(m))):
            desc_names.append('RadiusOfGyration (Value' + str(i)+" )")
    except:
        desc_names.append('RadiusOfGyration')
    desc.append(Chem.rdMolDescriptors.CalcSpherocityIndex(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcSpherocityIndex(m))):
            desc_names.append('SpherocityIndex (Value' + str(i)+" )")
    except:
        desc_names.append('SpherocityIndex')
    desc.append(Chem.rdMolDescriptors.CalcTPSA(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcTPSA(m))):
            desc_names.append('TPSA (Value' + str(i)+" )")
    except:
        desc_names.append('TPSA')
    
    # High dimensional descriptors
    desc.append(Chem.rdMolDescriptors.CalcWHIM(m))
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcWHIM(m))):
            desc_names.append('WHIM (Value' + str(i)+" )")
    except:
        desc_names.append('WHIM')
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR2D(m)) 
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcAUTOCORR2D(m))):
            desc_names.append('AUTOCORR2D (Value' + str(i)+" )")
    except:
        desc_names.append('AUTOCORR2D')
    desc.append(Chem.rdMolDescriptors.CalcAUTOCORR3D(m)) 
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcAUTOCORR3D(m))):
            desc_names.append('AUTOCORR3D (Value' + str(i)+" )")
    except:
        desc_names.append('AUTOCORR3D')
    desc.append(Chem.rdMolDescriptors.CalcRDF(m)) 
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcRDF(m))):
            desc_names.append('RDF (Value' + str(i)+" )")
    except:
        desc_names.append('RDF')
    desc.append(Chem.rdMolDescriptors.CalcMORSE(m)) 
    try:
        for i in range(len(Chem.rdMolDescriptors.CalcMORSE(m))):
            desc_names.append('MORSE (Value' + str(i)+" )")
    except:
        desc_names.append('MORSE')

    desc = flatten(desc)
    std_descriptors.append(desc)
    
    with open("descriptor_names.txt", "w") as f:
        for item in desc_names:
            f.write("%s\n" % item)
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
