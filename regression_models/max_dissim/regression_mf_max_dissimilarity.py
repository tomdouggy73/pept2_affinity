from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import sys
sys.path.append('../')
from matt_code.ml_networks import * # type: ignorecd

IPythonConsole.ipython_useSVG = True

# Main script settings
PERFORM_GRID_SEARCH = True

# Load dataset
dataset = pd.read_csv('/biggin/b229/chri6405/pept2_affinity/tanimoto_based_clusters/max_dissim/morgan_fingerprints/final_merged_data_with_fingerprints_3_clusters_max_dissim.csv')

# Data preprocessing
indices = dataset["SKPT (PEPT2)_x"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)_x"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)
clusters = dataset['Cluster'].values

# Get descriptors
descriptors = dataset.iloc[indices]
descriptors_array = descriptors.drop(descriptors.columns[0:12], axis=1).values


def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42, quantify_test_set_occupancies=False):
    
    """Perform regression using random forest and return results."""
    # Build model
    rf_reg = Regressor_RandomForest(logKi,  
                                     descriptors_array, 
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     bootstrap=bootstrap,
                                     seed=seed)

    # List to store test scores
    test_scores_list = []
    predicted_data = [[] for _ in range(len(logKi))]

    # Iterate through each unique cluster
    unique_clusters = np.unique(clusters)

    for test_cluster in unique_clusters:
        # Get indices for the current test cluster
        test_indices = np.where(clusters == test_cluster)[0]

        # Get indices for all other clusters (training set)
        training_indices = np.where(clusters != test_cluster)[0]

        # Train model with the custom split for the current cluster
        rf_reg.train_model_custom_split(training_indices, test_indices)

        # Store the test score for this iteration
        test_scores_list.append(rf_reg.test_scores)

        # Store the predicted data for this iteration at the correct indices
        predictions = rf_reg.pred_test  # Assume this is a list of predictions for the current test set
        for idx, pred in zip(test_indices, predictions):
            predicted_data[idx] = pred  # Append the prediction to the corresponding index

    # Calculate mean of the test scores across all iterations
    mean_test_scores = pd.concat(test_scores_list).mean()

    # Print only the mean test scores
    print(f'Mean Test Scores - MSE: {mean_test_scores["MSE"]:.4f}, PCC: {mean_test_scores["PCC"]:.4f}')

    # Save MSE, PCC
    MSE = mean_test_scores["MSE"]
    PCC = mean_test_scores["PCC"]

    return (predicted_data, rf_reg, MSE, PCC)

def plot_regression(predicted_data, MSE, PCC, outfile="/biggin/b229/chri6405/pept2_affinity/regression_outputs/regression_max_dissim/morgan_fingerprints/predicted_data_max_dissim.png", show=True):
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

# Call the regression model using best hyperparameters
predicted_data, rf_reg, MSE, PCC = perform_regression(
    n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
    max_features='sqrt', bootstrap=True
)

plot_regression(predicted_data, MSE, PCC)