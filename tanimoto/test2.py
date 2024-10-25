from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Define the Regressor_RandomForest class
class Regressor_RandomForest:
    """Class for random forest regression model with group K-fold cross-validation."""

    def __init__(self, target, descriptor, clusters, n_estimators, max_depth, min_samples_split, min_samples_leaf,                              max_features, bootstrap, seed=42):
        self.target = target
        self.desc = descriptor
        self.clusters = clusters  # Store clusters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed
        self.model = self.build_model()
        self.test_scores = []  # Initialize test_scores
        self.train_scores = []  # Initialize train_scores
        self.pred_train_all = [] # Initialize for training predictions
        self.pred_test_all = []

    def build_model(self):
        """Build the RandomForestRegressor model."""
        return RandomForestRegressor(n_estimators=self.n_estimators,
                                     max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     min_samples_leaf=self.min_samples_leaf,
                                     max_features=self.max_features,
                                     bootstrap=self.bootstrap,
                                     random_state=self.seed)

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

        # Perform Group K-Fold cross-validation
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

            # Ensure predictions are collected correctly
            y_train_all.append(y_train)
            y_test_all.append(y_test)
            pred_train_all.append(pred_train)
            pred_test_all.append(pred_test)

        # Concatenate results
        self.train_scores = pd.DataFrame(train_scores, columns=["MSE", "PCC"])
        self.test_scores = pd.DataFrame(test_scores, columns=["MSE", "PCC"])

        # Flatten and concatenate predictions
        self.y_train_all = np.concatenate(y_train_all)
        self.y_test_all = np.concatenate(y_test_all)
        self.pred_train_all = np.concatenate(pred_train_all)
        self.pred_test_all = np.concatenate(pred_test_all)

    def check_for_nans(self):
        if self.pred_test_all is not None and np.any(np.isnan(self.pred_test_all)):
            print("Warning: NaNs detected in predictions")
        
    def get_scores(self, y_true, y_pred):
        """Calculate MSE and PCC for true and predicted values."""
        mse = mean_squared_error(y_true, y_pred)
        pcc = np.corrcoef(y_true.ravel(), y_pred.ravel())[0][1]
        return [mse, pcc]
    
    def train_model_custom_split(self, training_indices, test_indices):
        # Extract training and test data based on indices
        x_train = self.desc[training_indices]
        y_train = self.target[training_indices]
        x_test = self.desc[test_indices]   
        y_test = self.target[test_indices]

        # Fit the model
        self.model.fit(x_train, y_train)

        # Store predictions
        self.pred_train_all = self.model.predict(x_train)
        self.pred_test_all = self.model.predict(x_test)

        # Debugging output
        print("Model trained on training data.")
        print(f"Predicted train: {self.pred_train_all[:5]}")  # Print first 5 predictions
        print(f"Predicted test: {self.pred_test_all[:5]}")    # Print first 5 predictions

        # Optionally, you can return scores directly if needed
        train_mse = mean_squared_error(y_train, self.pred_train_all)
        test_mse = mean_squared_error(y_test, self.pred_test_all)
        return train_mse, test_mse



# Load dataset
dataset = pd.read_csv('merged_clusters_complete.csv')

# Data preprocessing
indices = dataset["SKPT (PEPT2)"].dropna().index
Ki = dataset["SKPT (PEPT2)"].dropna().values
logKi = np.log10(Ki)
clusters = dataset['Cluster'].values[indices]

# Get descriptors
descriptors = dataset.iloc[indices]
descriptors_array = descriptors.drop(descriptors.columns[0:4], axis=1).values

# Function for performing regression
def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):
    """Perform regression using random forest and return results."""
    
    # Build model
    rf_reg = Regressor_RandomForest(logKi, 
                                     descriptors_array, 
                                     clusters, 
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     bootstrap=bootstrap,
                                     seed=seed)

    test_scores_list = []

    # Iterate through each unique cluster
    unique_clusters = np.unique(clusters)

    for test_cluster in unique_clusters:
        test_indices = np.where(clusters == test_cluster)[0]
        training_indices = np.where(clusters != test_cluster)[0]

        # # Train model with the custom split for the current cluster
        train_mse, test_mse = rf_reg.train_model_custom_split(training_indices, test_indices)  # Capture returned values

        print(f"Training on test_indices: {test_indices} and training_indices: {training_indices}")
        print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")

        # Store test scores
        test_scores_list.append(test_mse)  # Append test MSE to the list
        
    # After collecting scores, you can summarize or print them
    print("Test Scores: ", test_scores_list)
      
    if len(test_scores_list) > 0:
        mean_test_scores = pd.concat(test_scores_list).mean()
        print(f'Mean Test Scores - MSE: {mean_test_scores["MSE"]:.4f}, PCC: {mean_test_scores["PCC"]:.4f}')
    else:
        print("Warning: No valid test scores collected.")

    # Collect predictions for further evaluation
    predicted_data = [[] for _ in range(len(logKi))]

    for i, test_idx in enumerate(test_indices):
        predicted_data[test_idx].append(rf_reg.pred_test_all[i])

    # Ensure predicted data is numeric
    predicted_data_means = [np.mean(x) if len(x) > 0 else np.nan for x in predicted_data]
    predicted_data_stds = [np.std(x) if len(x) > 0 else np.nan for x in predicted_data]

    # Calculate metrics
    MSE = np.mean((np.array(predicted_data_means) - logKi) ** 2)
    PCC = np.corrcoef(predicted_data_means, logKi)[0, 1]

    return predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC

# Perform regression
predicted_data_means, predicted_data_stds, rf_reg, mse, pcc = perform_regression(
    n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
    max_features='sqrt', bootstrap=True
)


# Perform grid search for hyperparameters
def grid_search():
    """Perform grid search of n_estimators and max_depth."""
    n_estimators = [10, 20, 50, 100, 500]
    max_depth = [2, 5, 10, 20, 50]

    MSE_grid = np.zeros((len(n_estimators), len(max_depth)))
    PCC_grid = np.zeros((len(n_estimators), len(max_depth)))

    for i, est in enumerate(n_estimators):
        for j, depth in enumerate(max_depth):
            MSEs, PCCs = [], []
            for replicate in range(3):
                predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC = perform_regression(
                    est, depth, 2, 1, 'sqrt', True, seed=replicate)
                MSEs.append(MSE)
                PCCs.append(PCC)
            MSE_grid[i, j] = np.mean(MSEs)
            PCC_grid[i, j] = np.mean(PCCs)

    np.save("MSE_grid.npy", MSE_grid)
    np.save("PCC_grid.npy", PCC_grid)

# Optionally, perform grid search if required
grid_search()