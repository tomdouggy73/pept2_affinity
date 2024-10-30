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
descriptors_all = pd.read_csv("source_data/dataset_morganfingerprints.csv", header=None)
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

def perform_regression_beta_lactam_split(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):

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
    rf_reg.train_model_custom_split(range(0,142), range(142,184))

    # Sort the data according to the test_indices

    predicted_data = rf_reg.test_pred

    predicted_data_means = [np.mean(x) for x in predicted_data]
    predicted_data_stds = [np.std(x) for x in predicted_data]

    # get MSE and PCC from the computed means

    MSE = np.mean((np.array(predicted_data_means) - logKi[142:184] +3)**2)
    PCC = np.corrcoef(predicted_data_means, logKi[142:184]-3)[0,1]

    return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC)



def plot_regression_betalactam(predicted_data_means, predicted_data_stds, MSE, PCC, outfile=None, show=True, data_range_lower = 0, data_range_upper = 184):

    # plot the data
    plt.errorbar(logKi[data_range_lower:data_range_upper]-3, predicted_data_means, yerr=predicted_data_stds, fmt='x', color='black',ecolor='grey', capsize=3)

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

    # compute fresh MSE and PCC for this data subset

    PCC = np.corrcoef(predicted_data_means, logKi[data_range_lower:data_range_upper]-3)[0,1]
    MSE = np.mean((np.array(predicted_data_means) - logKi[data_range_lower:data_range_upper] +3)**2)

    plt.text(-9.5, -1.5, "MSE = %.2f\nMSE (kcal/mol) = %.2f\nPearson= %.2f" % (MSE, MSE/kcal_mol, PCC), fontsize=11)
    
    # also make a linear best fit line

    plt.plot(plot_range, np.poly1d(np.polyfit(logKi[data_range_lower:data_range_upper]-3, predicted_data_means, 1))(plot_range), color='dimgrey', linestyle='dashed')
    
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.clf()


def plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC, outfile=None, show=True, data_range_lower = 0, data_range_upper = 184):

    # plot the data
    plt.errorbar(logKi[data_range_lower:data_range_upper]-3, predicted_data_means[data_range_lower:data_range_upper], yerr=predicted_data_stds[data_range_lower:data_range_upper], fmt='x', color='black',ecolor='grey', capsize=3)

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

    # compute fresh MSE and PCC for this data subset

    PCC = np.corrcoef(predicted_data_means[data_range_lower:data_range_upper], logKi[data_range_lower:data_range_upper]-3)[0,1]
    MSE = np.mean((np.array(predicted_data_means[data_range_lower:data_range_upper]) - logKi[data_range_lower:data_range_upper] +3)**2)

    plt.text(-9.5, -1.5, "MSE = %.2f\nMSE (kcal/mol) = %.2f\nPearson= %.2f" % (MSE, MSE/kcal_mol, PCC), fontsize=11)
    
    # also make a linear best fit line

    plt.plot(plot_range, np.poly1d(np.polyfit(logKi[data_range_lower:data_range_upper]-3, predicted_data_means[data_range_lower:data_range_upper], 1))(plot_range), color='dimgrey', linestyle='dashed')
    
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    plt.clf()

 

#%%

# using the best parameters, make some predictions with a random seed.

predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(100, 10, 2, 1, 'sqrt', True, seed=42)
plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC, data_range_lower=142, data_range_upper=184)

predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression_beta_lactam_split(100, 10, 2, 1, 'sqrt', True, seed=42)
plot_regression_betalactam(predicted_data_means, predicted_data_stds, MSE, PCC, data_range_lower=142, data_range_upper=184)

#%%

np.save("regression_morganfp_randomforest/predicted_data_means_beta_lactams.npy", predicted_data_means)
np.save("regression_morganfp_randomforest/predicted_data_stds_beta_lactams.npy", predicted_data_stds)

# %%
