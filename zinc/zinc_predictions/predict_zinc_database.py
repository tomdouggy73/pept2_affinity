#%%

from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors, rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.ipython_useSVG=True 
import matplotlib.pyplot as plt
import os

from matt_code.ml_networks import *

#%%

dataset = pd.read_csv("dataset_smiles.csv")

# Treat all 'inf' values as missing values
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# find all indices for which data exists for SKPT (PEPT2)

indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)

rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles]

# Get descriptors

descriptors_all = pd.read_csv("dataset_morganfingerprints.csv", header=None)
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

def plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC, outfile=None, show=False):

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

# using the best parameters, make some predictions with a random seed, this constructs the model

predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(100, 10, 2, 1, 'sqrt', True, seed=42)
plot_regression(predicted_data_means, predicted_data_stds, MSE, PCC)

# %%

# initialise all predictions

all_predictions_kcal = []

# predict the zinc ligands

# Create the output directory
output_dir = 'zinc_chunk_output'
os.makedirs(output_dir, exist_ok=True)

# Number of chunks
NUM_CHUNKS = 100

# Initialize a list to hold all DataFrames for concatenation later
all_chunk_dfs = []

for chunk in range(NUM_CHUNKS):
    print(f"Predicting chunk {chunk}")

    descriptors_zinc = pd.read_csv(f"mf_zinc_chunks/ligand_morganfingerprints_chunk{chunk}.csv", header=None)
    print(f"Done loading chunk {chunk}")

    zinc_smiles_array = descriptors_zinc[0].values
    descriptors_array_zinc = descriptors_zinc.drop(0, axis=1).values

    prediction = rf_reg.get_predictions(rf_reg.model, descriptors_array_zinc) - 3

    # Convert to kcal/mol
    kT = 0.0019872043 * 298
    kcal_mol = kT * np.log(10)
    prediction_kcal = prediction * kcal_mol

    # Get top 1000 binders for the current chunk
    indices_top1000 = np.argsort(prediction_kcal)[:1000]
    mols_top1000 = [Chem.MolFromSmiles(zinc_smiles_array[i]) for i in indices_top1000]
    smiles_top1000 = [zinc_smiles_array[i] for i in indices_top1000]

    # Save top 1000 binders for this chunk
    chunk_df = pd.DataFrame({
        'SMILES': smiles_top1000,
        'Predicted_kcal': prediction_kcal[indices_top1000]
    })
    chunk_filename = os.path.join(output_dir, f'chunk{chunk}_top_1000.csv')
    chunk_df.to_csv(chunk_filename, index=False)
    print(f"Saved top 1000 for chunk {chunk} to {chunk_filename}")

    # Append the DataFrame for this chunk to the list for later concatenation
    all_chunk_dfs.append(chunk_df)

# Concatenate all the DataFrames from each chunk into a single DataFrame
combined_df = pd.concat(all_chunk_dfs, ignore_index=True)

# Filter out peptides for the final DataFrame
top_non_peptides = []

# Load all possible peptides into rdkit mols
peptide_smiles = pd.read_csv("peptides/all_peptide_smiles.dat", header=None)[0].values
peptide_smiles_canon = [Chem.CanonSmiles(p) for p in peptide_smiles]

for i in range(len(combined_df)):
    canon_smiles_mol = Chem.CanonSmiles(combined_df['SMILES'].iloc[i])
    if canon_smiles_mol not in peptide_smiles_canon:
        top_non_peptides.append((combined_df['SMILES'].iloc[i], combined_df['Predicted_kcal'].iloc[i]))

# Create a DataFrame for all top non-peptides
top_non_peptides_df = pd.DataFrame(top_non_peptides, columns=["SMILES", "Predicted_kcal"])

# Get top 1000 non-peptides from all chunks
final_top_1000 = top_non_peptides_df.nlargest(1000, "Predicted_kcal")

# Save the final output
final_output_filename = 'final_top_1000.csv'
final_top_1000.to_csv(final_output_filename, index=False)
print(f"Final top 1000 non-peptides saved to {final_output_filename}")
