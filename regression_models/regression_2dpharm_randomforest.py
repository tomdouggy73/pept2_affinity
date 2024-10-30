#%%

from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt

from matt_code.ml_networks import *


# script settings
PERFORM_GRID_SEARCH = True

#%%

dataset = pd.read_csv("source_data/dataset_smiles.csv")

# Treat all 'inf' values as missing values
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# find all indices for which data exists for Caco-2 (PEPT1)

indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)

rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]

# Get descriptors

descriptors_all = pd.read_csv("source_data/dataset_2dpharm.csv", header=None)
descriptors = descriptors_all.iloc[indices]

descriptors_array = descriptors.drop(0, axis=1).values



# %%


def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):

    # Build model
    rf_reg = Regressor_RandomForest(logKi, 
                    descriptors_array, 
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    seed=seed

    )

    # Train and evaluate model using cross validation
    rf_reg.train_model_cv(
                        kfolds=3, 
                        repeats=5,
                        seed=seed
    )

    # Sort the data according to the test_indices

    test_indices = rf_reg.test_indices_list

    predicted_data = [[] for i in range(len(logKi))]

    for i in range(len(test_indices)):
        for j in range(len(test_indices[i])):
            predicted_data[test_indices[i][j]].append(rf_reg.test_pred[i][j]-3)

    predicted_data_means = [np.mean(x) for x in predicted_data]
    predicted_data_stds = [np.std(x) for x in predicted_data]

    # get MSE and PCC from the computed means

    MSE = np.mean((np.array(predicted_data_means) - logKi +3)**2)
    PCC = np.corrcoef(predicted_data_means, logKi-3)[0,1]

    return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC)

def plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC, outfile=None, show=True):

    # plot the data
    plt.errorbar(logKi-3, predicted_data_means, yerr=predicted_data_stds, fmt='x', color='black',ecolor='grey', capsize=3)

    # equal axes, square plot and 45 degree line
    plt.axis('square')

    plot_range = (-10, 0.5)

    plt.xlim(plot_range)
    plt.xticks(np.arange(-8, 2, 2))
    plt.yticks(np.arange(-8, 2, 2))
    plt.ylim(plot_range)

    # calculate kT at 310K in kcal/mol

    kT = 0.0019872043 * 310

    kcal_mol = kT * np.log(10)

    # 45 degree line

    plt.plot(plot_range, plot_range, color='grey', linestyle='dotted')

    plt.fill_between(plot_range, plot_range - kcal_mol, plot_range + kcal_mol, color='grey', alpha=0.2)

    plt.xlabel("Experimental log$_{10}$(Ki / M)")
    plt.ylabel("Predicted log$_{10}$(Ki / M)")

    plt.text(-9.5, -1.5, "MSE = %.2f\nMSE (kcal/mol) = %.2f\nPearson= %.2f" % (MSE, MSE/kcal_mol, PCC), fontsize=11)
    
    # also make a linear best fit line

    plt.plot(plot_range, np.poly1d(np.polyfit(logKi-3, predicted_data_means, 1))(plot_range), color='dimgrey', linestyle='dashed')
    
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.clf()


#%%

# perform 2D-grid search of n_estimators and max_depth
if PERFORM_GRID_SEARCH:
        
    n_estimators = [10, 50, 100, 200, 500]
    max_depth = [2, 5, 10, 20, 50]

    MSE_grid = np.zeros((len(n_estimators), len(max_depth)))
    PCC_grid = np.zeros((len(n_estimators), len(max_depth)))

    for i in range(len(n_estimators)):
        for j in range(len(max_depth)):
            MSEs, PCCs = [], []
            for replicate in range(3):
                predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(n_estimators[i], max_depth[j], 2, 1, 'sqrt', True, seed=replicate)
                MSEs.append(MSE)
                PCCs.append(PCC)
                print(f"Regression with n_estimators={n_estimators[i]}, max_depth={max_depth[j]}, rep {replicate} -> MSE={MSE}, PCC={PCC}")
            MSE_grid[i,j] = np.mean(MSEs)
            PCC_grid[i,j] = np.mean(PCCs)


    np.save("regression_outputs/regression_2dpharm_randomforest/PCC_grid_n_estimators_max_depth.npy", PCC_grid)
    np.save("regression_outputs/regression_2dpharm_randomforest/MSE_grid_n_estimators_max_depth.npy", MSE_grid)

PCC_grid_n_estimators_max_depth = np.load("regression_outputs/regression_2dpharm_randomforest/PCC_grid_n_estimators_max_depth.npy")
MSE_grid_n_estimators_max_depth = np.load("regression_outputs/regression_2dpharm_randomforest/MSE_grid_n_estimators_max_depth.npy")


#%%

# perform 2D-grid search of n_samples_split and n_samples_leaf

if PERFORM_GRID_SEARCH:
    min_samples_split = [2, 5, 10, 20, 50]
    min_samples_leaf = [1, 2, 5, 10, 20]

    MSE_grid = np.zeros((len(min_samples_split), len(min_samples_leaf)))
    PCC_grid = np.zeros((len(min_samples_split), len(min_samples_leaf)))

    for i in range(len(min_samples_split)):
        for j in range(len(min_samples_leaf)):
            MSEs, PCCs = [], []
            for replicate in range(3):
                predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(100, 10, min_samples_split[i], min_samples_leaf[j], 'sqrt', True, seed=replicate)
                MSEs.append(MSE)
                PCCs.append(PCC)
                print(f"Regression with min_samples_split={min_samples_split[i]}, min_samples_leaf={min_samples_leaf[j]}, rep {replicate} -> MSE={MSE}, PCC={PCC}")
            MSE_grid[i,j] = np.mean(MSEs)
            PCC_grid[i,j] = np.mean(PCCs)


    np.save("regression_outputs/regression_2dpharm_randomforest/PCC_grid_min_samples_split_min_samples_leaf.npy", PCC_grid)
    np.save("regression_2dpharm_randomforest/MSE_grid_min_samples_split_min_samples_leaf.npy", MSE_grid)

PCC_grid_min_samples_split_min_samples_leaf = np.load("regression_outputs/regression_2dpharm_randomforest/PCC_grid_min_samples_split_min_samples_leaf.npy")
MSE_grid_min_samples_split_min_samples_leaf = np.load("regression_outputs/regression_2dpharm_randomforest/MSE_grid_min_samples_split_min_samples_leaf.npy")


#%%
plt.clf()
plt.imshow(MSE_grid_n_estimators_max_depth, cmap='plasma', origin='lower', interpolation='nearest')
plt.colorbar(label="MSE", orientation='vertical')
plt.xticks(np.arange(len(max_depth)), max_depth)
plt.yticks(np.arange(len(n_estimators)), n_estimators)
plt.xlabel("Max depth")
plt.ylabel("Number of estimators")
plt.show()

plt.clf()

plt.imshow(MSE_grid_min_samples_split_min_samples_leaf, cmap='plasma', origin='lower', interpolation='nearest')
plt.colorbar(label="MSE", orientation='vertical')
plt.xticks(np.arange(len(min_samples_leaf)), min_samples_leaf)
plt.yticks(np.arange(len(min_samples_split)), min_samples_split)
plt.xlabel("Min samples leaf")
plt.ylabel("Min samples split")
plt.show()

 

#%%

# using the best parameters, make some predictions with a random seed.

predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(100, 10, 2, 1, 'sqrt', True, seed=42)
plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC)

feature_importances = rf_reg.model.feature_importances_

np.save("regression_outputs/regression_2dpharm_randomforest/feature_importances.npy", feature_importances)
np.save("regression_outputs/regression_2dpharm_randomforest/predicted_data_means.npy", predicted_data_means)
np.save("regression_outputs/regression_2dpharm_randomforest/predicted_data_stds.npy", predicted_data_stds)
np.save("regression_outputs/regression_2dpharm_randomforest/MSE.npy", MSE)
np.save("regression_outputs/regression_2dpharm_randomforest/PCC.npy", PCC)

feature_importances = np.load("regression_outputs/regression_2dpharm_randomforest/feature_importances.npy")
predicted_data_means = np.load("regression_outputs/regression_2dpharm_randomforest/predicted_data_means.npy")
predicted_data_stds = np.load("regression_outputs/regression_2dpharm_randomforest/predicted_data_stds.npy")
MSE = np.load("regression_outputs/regression_2dpharm_randomforest/MSE.npy")
PCC = np.load("regression_outputs/regression_2dpharm_randomforest/PCC.npy")


#%%
# plot the feature importances

plt.clf()

plt.plot(np.arange(len(feature_importances)), feature_importances)
plt.xlabel("2D-pharmacophore bit index")
plt.ylabel("Feature importance")
plt.show()
# %%



# take the top 25 features

top_features = np.argsort(feature_importances)[::-1][:25]
print(top_features)

# Interpret the top 25 bits

from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory, Utils

fdefName = 'regression_outputs/regression_2dpharm_randomforest/features.fdef'
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3,trianglePruneBins=False)
sigFactory.skipFeats=['ZnBinder']
sigFactory.SetBins([(0, 2), (2, 5),(5, 8)])
sigFactory.Init()

for i in top_features:
    print(sigFactory.GetBitDescription(i))




def visualise_pharmophore(mol, bits, feature_importances,sigFactory):
    """ Draw the molecule with the pharmacophore bits in the list highlighted. """

    fp = Generate.Gen2DFingerprint(mol,sigFactory)
    featFams = sigFactory.GetFeatFamilies()
    feats = sigFactory.GetMolFeats(mol)

    on_bits_of_interest = []
    for bit in bits:
        # Test if the bit is on
        if fp.GetBit(int(bit)):
            on_bits_of_interest.append(bit)
    
    
    # Now we follow the procedure rdkit takes for generating pharmacophore bits,
    # then check whether the bit is of interest, followed by highlighting the atom ids.
            
    dMat = Chem.GetDistanceMatrix(mol, sigFactory.includeBondOrder)
    
    minCount = sigFactory.minPointCount
    maxCount = sigFactory.maxPointCount
    nFeats = len(featFams)

    perms = []
    for count in range(minCount, maxCount + 1):
        perms.extend(Utils.GetIndexCombinations(nFeats, count))

    sig = sigFactory.GetSignature()

    atom_indices_to_highlight = dict()
    for bit in on_bits_of_interest:
        atom_indices_to_highlight[bit] = [[] for i in range(nFeats)]

    for perm in perms:
        featClasses = [0] * len(perm)
        for i in range(1, len(perm)):
            if perm[i] == perm[i - 1]:
                featClasses[i] = featClasses[i - 1]
            else:
                featClasses[i] = featClasses[i - 1] + 1
    
        matchPerms = [feats[x] for x in perm]
        matchesToMap = Utils.GetUniqueCombinations(matchPerms, featClasses)
        for i, entry in enumerate(matchesToMap):
            matchesToMap[i] = [x[1] for x in entry]
        for match in matchesToMap:
            if sigFactory.shortestPathsOnly:
                idx = Generate._ShortestPathsMatch(match, perm, sig, dMat, sigFactory)
                
                if idx in on_bits_of_interest:
                    for n, p in enumerate(perm):
                        atom_indices_to_highlight[idx][p].extend(match[n])

    # draw the molecule as many times as there are on bits of interest

    rgba_color = (0.78, 0.71, 0.24, 1.0)


    highlightAtomColor = [{x: rgba_color for x in range(len(mol.GetAtoms()))}]*len(on_bits_of_interest)
    highlightBondColor = [{x: rgba_color for x in range(len(mol.GetAtoms()))}]*len(on_bits_of_interest)

    mols_to_draw = []
    highlight_lists = []
    for i,b in enumerate(on_bits_of_interest):
        mol_copy = Chem.Mol(mol)
        to_highlight_list = []
        for j in range(nFeats):
            for atom in atom_indices_to_highlight[b][j]:
                to_highlight_list.append(atom)
        highlight_lists.append(to_highlight_list)
        mols_to_draw.append(mol_copy)
    print(highlight_lists)
    return Draw.MolsToGridImage(mols_to_draw, legends=[f"bit {x}, weight {round(feature_importances[x],3)}" for x in on_bits_of_interest], highlightAtomLists = highlight_lists, highlightAtomColors=highlightAtomColor, highlightBondColors=highlightBondColor)

#%%

visualise_pharmophore(rdkit_mols[0], top_features,feature_importances,sigFactory)
#visualise_pharmophore(rdkit_mols[147], top_features,feature_importances,sigFactory)




#%%

# flatness plot

from rdkit import DataStructs

def tanimoto_similarity(mol1, mol2):
    """ Calculate tanimoto similarity between two molecules. """
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

pairwise_similarity = np.zeros((len(rdkit_mols)-1, len(rdkit_mols)-1))

for i in range(len(rdkit_mols)-1):
    for j in range(i,len(rdkit_mols)-2):
        sim = tanimoto_similarity(rdkit_mols[i], rdkit_mols[j+1])
        pairwise_similarity[i,j] = sim
        pairwise_similarity[j,i] = sim

exp_differences = np.zeros((len(rdkit_mols), len(rdkit_mols)))
pred_differences = np.zeros((len(rdkit_mols), len(rdkit_mols)))

for i in range(len(rdkit_mols)-1):
    for j in range(i,len(rdkit_mols)-2):
        exp_differences[i,j] = abs(logKi[i]-logKi[j+1])
        exp_differences[j,i] = abs(logKi[i]-logKi[j+1])
        pred_differences[i,j] = abs(predicted_data_means[i]-predicted_data_means[j+1])
        pred_differences[j,i] = abs(predicted_data_means[i]-predicted_data_means[j+1])

plt.clf()


# compute a 1d histogram of experimental - predicted differences

hist, bins = np.histogram(exp_differences.flatten()-pred_differences.flatten(), bins=20, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# do this again sequentially, taken into account molecules in similarity brackets

similarity_brackets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for i in range(len(similarity_brackets)-1):
    indices = np.where((pairwise_similarity > similarity_brackets[i]) & (pairwise_similarity <= similarity_brackets[i+1]))
    hist,_ =  np.histogram(exp_differences[indices].flatten()-pred_differences[indices].flatten(), bins=bins, density=True)
    plt.plot(bin_centers, hist, label=f"{similarity_brackets[i]} < similarity < {similarity_brackets[i+1]}", color= plt.cm.viridis(i/len(similarity_brackets)))

plt.legend()
plt.xlabel("Experimental - predicted pairwise log$_{10}$(Ki / M) difference")
plt.ylabel("Histogram density")
plt.vlines(0, 0, 1.2, color='grey', linestyle='dotted')
plt.show()

# %%

# predict the penG ligands

descriptors_penG = pd.read_csv("penG_predictions/ligand_2dpharm.csv", header=None)

descriptors_array_penG = descriptors_penG.drop(0, axis=1).values

prediction = rf_reg.get_predictions(rf_reg.model,descriptors_array_penG) -3

# convert to kcal/mol

kT = 0.0019872043 * 298

kcal_mol = kT * np.log(10)

prediction_kcal = prediction * kcal_mol
# %%
