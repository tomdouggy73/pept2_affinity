#%%

from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from matt_code.ml_networks import * # type: ignore

# script settings
PERFORM_GRID_SEARCH = False

#%%

dataset = pd.read_csv("/biggin/b229/chri6405/pept2_affinity/source_data/dataset_smiles.csv")

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

descriptors_all = pd.read_csv("/biggin/b229/chri6405/pept2_affinity/source_data/dataset_morganfingerprints.csv", header=None)
descriptors = descriptors_all.iloc[indices]

descriptors_array = descriptors.drop(0, axis=1).values

# %%

def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=41):
    # Build model
    rf_reg = Regressor_RandomForest(
        logKi, descriptors_array, 
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        seed=seed
    )

    # Train and evaluate model using nested cross-validation
    rf_reg.train_model_nested_cv(
        outer_kfolds=5, 
        inner_kfolds=3,
        repeats=1,
        seed=seed
    )

    # Calculate average predictions and test errors
    test_indices = rf_reg.test_indices_list
    predicted_data = [[] for i in range(len(logKi))]

    for i in range(len(test_indices)):
        for j in range(len(test_indices[i])):
            predicted_data[test_indices[i][j]].append(rf_reg.test_pred[i][j] - 3)

    predicted_data_means = [np.mean(x) for x in predicted_data]
    predicted_data_stds = [np.std(x) for x in predicted_data]

    # Overall metrics across all folds
    mse_mean = np.mean((np.array(predicted_data_means) - logKi + 3) ** 2)
    pcc_mean = np.corrcoef(predicted_data_means, logKi - 3)[0, 1]

    print(f'Average Inner MSE: {rf_reg.mean_inner_scores["MSE"]:.4f}, PCC: {rf_reg.mean_inner_scores["PCC"]:.4f}')
    print(f'Average Outer MSE: {rf_reg.mean_outer_scores["MSE"]:.4f}, PCC: {rf_reg.mean_outer_scores["PCC"]:.4f}')
    print(f'Overall Mean MSE (outer predictions): {mse_mean:.4f}, PCC: {pcc_mean:.4f}')

    return {
        "predicted_data_means": predicted_data_means,
        "predicted_data_stds": predicted_data_stds,
        "rf_reg": rf_reg,
        "mean_inner_scores": rf_reg.mean_inner_scores,
        "mean_outer_scores": rf_reg.mean_outer_scores,
        "overall_mse": mse_mean,
        "overall_pcc": pcc_mean
    }

def plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC, outfile=None, show=True):

    # Set the plot size
    plt.figure(figsize=(24, 16))

    # Increase general font size for the plot
    plt.rcParams.update({'font.size': 18})  # Set a general font size for the entire plot

    # Plot predicted_data vs logKi - 3 without error bars
    plt.plot(logKi - 3, predicted_data, 'x', color='black')  # Scatter plot without error bars

    # Square plot with equal axes and a 45-degree line
    plt.axis('square')
    plot_range = (-10, 3)
    plt.xlim(-10, 1)
    plt.ylim(plot_range)
    plt.xticks(np.arange(-8, 0.1, 2), fontsize=12)  # Adjust x-tick font size
    plt.yticks(np.arange(-8, 2.1, 2), fontsize=12)  # Adjust y-tick font size

    # kT at 310K in kcal/mol
    kT = 0.0019872043 * 310
    kcal_mol = kT * np.log(10)

    # 45-degree line
    plt.plot(plot_range, plot_range, color='grey', linestyle='dotted')

    # Shaded region for ± kcal/mol around 45-degree line
    plt.fill_between(plot_range, plot_range - kcal_mol, plot_range + kcal_mol, color='grey', alpha=0.2)

    # Labels for the plot with increased font size
    plt.xlabel("Experimental log$_{10}$(Ki / M)", fontsize=16)
    plt.ylabel("Predicted log$_{10}$(Ki / M)", fontsize=16)

    # Display MSE and PCC on the plot with larger font
    plt.text(-9.5, -1.5, f"MSE = {MSE:.2f}\nMSE (kcal/mol) = {MSE/kcal_mol:.2f}\nPearson = {PCC:.2f}", fontsize=14)

    # Linear fit line for predicted_data vs logKi - 3
    plt.plot(plot_range, np.poly1d(np.polyfit(logKi - 3, predicted_data, 1))(plot_range), color='dimgrey', linestyle='dashed')

    # Save or show the plot
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


    np.save("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/PCC_grid_n_estimators_max_depth.npy", PCC_grid)
    np.save("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/MSE_grid_n_estimators_max_depth.npy", MSE_grid)

PCC_grid_n_estimators_max_depth = np.load("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/PCC_grid_n_estimators_max_depth.npy")
MSE_grid_n_estimators_max_depth = np.load("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/MSE_grid_n_estimators_max_depth.npy")


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


    np.save("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/PCC_grid_min_samples_split_min_samples_leaf.npy", PCC_grid)
    np.save("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/MSE_grid_min_samples_split_min_samples_leaf.npy", MSE_grid)

PCC_grid_min_samples_split_min_samples_leaf = np.load("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/PCC_grid_min_samples_split_min_samples_leaf.npy")
MSE_grid_min_samples_split_min_samples_leaf = np.load("/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_morganfp_randomforest/MSE_grid_min_samples_split_min_samples_leaf.npy")


#%%
""" plt.clf()
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
"""
 

#%%

# using the best parameters, make some predictions with a random seed.

predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC, test_set_occupancies = perform_regression(100, 10, 2, 1, 'sqrt', True, seed=42, quantify_test_set_occupancies=True)
plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC)

print (f'MSE = {MSE}, PCC = {PCC}')