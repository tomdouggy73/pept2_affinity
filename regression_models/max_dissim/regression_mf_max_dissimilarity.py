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
sys.path.append('../../')
from matt_code.ml_networks import *


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

    # Calculate mean of the test scores across all iterations
    mean_test_scores = pd.concat(test_scores_list).mean()

    # Print only the mean test scores
    # print(f'Mean Test Scores - MSE: {mean_test_scores["MSE"]:.4f}, PCC: {mean_test_scores["PCC"]:.4f}')

    # Sort the data according to the test_indices
    predicted_data = [[] for i in range(len(logKi))]

    # Iterate directly over test_indices
    for i, test_idx in enumerate(test_indices):
        if i < len(rf_reg.pred_test):  # Ensure we are within bounds
            predicted_data[test_idx].append(rf_reg.pred_test[i] - 3)

    # Compute means and stds while handling empty entries
    predicted_data_means = [np.mean(x) if len(x) > 0 else np.nan for x in predicted_data]
    predicted_data_stds = [np.std(x) if len(x) > 0 else np.nan for x in predicted_data]
    
    print(predicted_data)

    if quantify_test_set_occupancies:
        test_set_occupancies = [len(x) for x in predicted_data]
    
    # Get MSE and PCC from the computed means
    MSE = np.nanmean((np.array(predicted_data_means) - logKi + 3) ** 2)
    PCC = np.corrcoef(predicted_data_means, logKi - 3)[0, 1] 

    return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC)


if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv('../../tanimoto/max_dissimilarity/final_merged_data_with_fingerprints.csv')

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

    predicted_data_means, predicted_data_stds, rf_reg, mse, pcc = perform_regression(
        n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
        max_features='sqrt', bootstrap=True
    )
