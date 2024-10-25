from rdkit import Chem
import rdkit.Chem.Descriptors as Descriptors
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import numpy as np
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

IPythonConsole.ipython_useSVG = True

np.set_printoptions(threshold=1)

# Define the Regressor_RandomForest class
class Regressor_RandomForest:
    """Class for random forest regression model"""

    def __init__(self, target, descriptor, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):
        self.target = target
        self.desc = descriptor
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed
        self.model = self.build_model()
        self.test_scores = test_scores
        self.train_scores = train_scores
        self.clusters = clusters

    def build_model(self):
        model = RandomForestRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features,
                                      bootstrap=self.bootstrap,
                                      random_state=self.seed)
        return model

    def train_model_cv(self, kfolds, repeats=1, seed=42):
        """
        Train model and obtain predictions using (repeated) Group K-fold cross validation.

        Args:
            kfolds: Number of folds for cross-validation.
            repeats: Number of times to repeat kfold CV.
            seed: Random seed.
        """
        # Use GroupKFold to split based on the pre-assigned clusters
        kf = GroupKFold(n_splits=kfolds)

        # Store results
        train_scores, test_scores = [], []
        y_train_all, y_test_all, pred_train_all, pred_test_all = [], [], [], []

        # Perform Group K-Fold cross-validation, ensuring splits respect clusters
        for train_index, test_index in kf.split(self.desc, self.target, groups=self.clusters):
            x_train, x_test = self.desc[train_index], self.desc[test_index]
            y_train, y_test = self.target[train_index], self.target[test_index]

            # Build and train the model
            model = self.build_model()
            model.fit(x_train, y_train)

            # Get predictions
            pred_train = self.get_predictions(model, x_train)
            pred_test = self.get_predictions(model, x_test)

            # Store training and testing scores
            train_scores.append(self.get_scores(y_train, pred_train))
            test_scores.append(self.get_scores(y_test, pred_test))

            # Optionally, store predictions and actual values for further analysis
            y_train_all.append(y_train)
            y_test_all.append(y_test)
            pred_train_all.append(pred_train)
            pred_test_all.append(pred_test)

        # Convert the results into DataFrame or any other structure for further analysis
        self.train_scores = pd.DataFrame(train_scores, columns=["MSE", "PCC"])
        self.test_scores = pd.DataFrame(test_scores, columns=["MSE", "PCC"])

        # Optionally store predictions for post-evaluation
        self.y_train_all = np.concatenate(y_train_all)
        self.y_test_all = np.concatenate(y_test_all)
        self.pred_train_all = np.concatenate(pred_train_all)
        self.pred_test_all = np.concatenate(pred_test_all)

    def get_predictions(self, model, x):
        return model.predict(x);

    def get_scores(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        pcc = np.corrcoef(y_true.ravel(), y_pred.ravel())[0][1]
        return [mse, pcc];


# Main script settings
PERFORM_GRID_SEARCH = True

# Load dataset
dataset = pd.read_csv('merged_clusters_complete.csv');

# Data preprocessing
indices = dataset["SKPT (PEPT2)"].dropna().index
smiles = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)
clusters = dataset['Cluster'].values

# Get descriptors
descriptors = dataset.iloc[indices]
descriptors_array = descriptors.drop(descriptors.columns[0:4], axis=1).values

def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):
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
    print(f'Mean Test Scores - MSE: {mean_test_scores["MSE"]:.4f}, PCC: {mean_test_scores["PCC"]:.4f}')

    # Sort the data according to the test_indices
    predicted_data = [[] for i in range(len(logKi))]

    # Iterate directly over test_indices
    for i, test_idx in enumerate(test_indices):  # test_indices is already an array of indices
        predicted_data[test_idx].append(rf_reg.test_pred[i] - 3)

    predicted_data_means = [np.mean(x) for x in predicted_data]
    predicted_data_stds = [np.std(x) for x in predicted_data]

    # Get MSE and PCC from the computed means
    MSE = np.mean((np.array(predicted_data_means) - logKi + 3) ** 2)
    PCC = np.corrcoef(predicted_data_means, logKi - 3)[0, 1]
   
    return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC)


predicted_data_means, predicted_data_stds, rf_reg, mse, pcc = perform_regression(
    n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
    max_features='sqrt', bootstrap=True
)


    
    
#%% Perform 2D-grid search of n_estimators and max_depth
if PERFORM_GRID_SEARCH:
        
    n_estimators = [10, 20, 30, 40, 50, 60, 100, 500]
    max_depth = [2, 5, 8, 10, 12, 14, 50]

    MSE_grid = np.zeros((len(n_estimators), len(max_depth)))
    PCC_grid = np.zeros((len(n_estimators), len(max_depth)))

    for i in range(len(n_estimators)):
        for j in range(len(max_depth)):
            MSEs, PCCs = [], []
            for replicate in range(3):
                # Pass the correct parameters based on your regression function
                predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(
                    n_estimators[i], 
                    max_depth[j], 
                    2,  # min_samples_split
                    1,  # min_samples_leaf
                    'sqrt',  # max_features
                    True,  # bootstrap
                    seed=replicate  # random seed
                )
                MSEs.append(MSE)
                PCCs.append(PCC)
                print(f"Regression with n_estimators={n_estimators[i]}, max_depth={max_depth[j]}, rep {replicate} -> MSE={mse}, PCC={pcc}")
            MSE_grid[i, j] = np.mean(MSEs)
            PCC_grid[i, j] = np.mean(PCCs)


    np.save("tanimoto/hierarch_cluster/output/PCC_grid_n_estimators_max_depth.npy", PCC_grid)
    np.save("tanimoto/hierarch_cluster/output/MSE_grid_n_estimators_max_depth.npy", MSE_grid)

PCC_grid_n_estimators_max_depth = np.load("tanimoto/hierarch_cluster/output/PCC_grid_n_estimators_max_depth.npy")
MSE_grid_n_estimators_max_depth = np.load("tanimoto/hierarch_cluster/output//MSE_grid_n_estimators_max_depth.npy")


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
                print(f"Regression with min_samples_split={min_samples_split[i]}, min_samples_leaf={min_samples_leaf[j]}, rep {replicate} -> MSE={mse}, PCC={pcc}")
            MSE_grid[i,j] = np.mean(MSEs)
            PCC_grid[i,j] = np.mean(PCCs)


    np.save("tanimoto/hierarch_cluster/output/PCC_grid_min_samples_split_min_samples_leaf.npy", PCC_grid)
    np.save("tanimoto/hierarch_cluster/output/MSE_grid_min_samples_split_min_samples_leaf.npy", MSE_grid)

PCC_grid_min_samples_split_min_samples_leaf = np.load("tanimoto/hierarch_cluster/output//PCC_grid_min_samples_split_min_samples_leaf.npy")
MSE_grid_min_samples_split_min_samples_leaf = np.load("tanimoto/hierarch_cluster/output//MSE_grid_min_samples_split_min_samples_leaf.npy")


