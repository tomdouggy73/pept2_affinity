from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from vendi_score import vendi
from sklearn.ensemble import RandomForestRegressor

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')


def split_data(kf, desc, target):
    """
    Splits the descriptors and target into training and testing sets using K-fold cross validation.

    Args:
        kf: KFold object for splitting data.
        desc: Descriptors (features).
        target: Target variable (output).

    Returns:
        splits: A dictionary with training and testing sets.
        test_indices_list: A list of indices for test sets.
    """
    splits = {'xtrain': [], 'xtest': [], 'ytrain': [], 'ytest': []}
    test_indices_list = []

    for train_index, test_index in kf.split(desc):
        splits['xtrain'].append(desc[train_index])  # Ensure desc is numeric here
        splits['xtest'].append(desc[test_index])
        splits['ytrain'].append(target[train_index])
        splits['ytest'].append(target[test_index])
        test_indices_list.append(test_index)

    return splits, test_indices_list


class Regressor_RandomForest():
    """Class for random forest regression model"""

    def __init__(self, target, descriptor, smiles, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):
        self.target = target
        self.desc = descriptor
        self.smiles = smiles
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed
        self.model = self.build_model()
        self.test_scores = None
        self.train_scores = None
        self.test_indices_list = []
    
    def build_model(self):
        """
        Construct random forest model using input parameters.

        Returns:
            model: Compiled model.
        """
        model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, 
                                       min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, 
                                       max_features=self.max_features, bootstrap=self.bootstrap, 
                                       random_state=self.seed)
        return model
    
    def train_model_cv(self, kfolds, repeats=1, seed=42):
        """
        Train model and obtain predictions using (repeated) K-fold cross validation.

        Args:
            kfolds: Number of folds for cross validation.
            repeats: Number of times to repeat kfold CV.
            seed: Random seed.
        """
   
        kf = RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
        splits, test_indices_list = split_data(kf, self.desc, self.target)  # Store test indices in the class
        train_scores, test_scores, vendi_scores, all_vendi_scores, vendi_variances  = [], [], [], [], []
        y_train_all, y_test_all, pred_train_all, pred_test_all = [], [], [], []

        for k in range(kf.get_n_splits()):
            model = self.build_model()
            x_train, x_test = splits['xtrain'][k], splits['xtest'][k]
            y_train, y_test = splits['ytrain'][k], splits['ytest'][k]

            # Fit the model using the descriptor data
            model.fit(x_train, y_train)

            pred_train = self.get_predictions(model, x_train)
            pred_test = self.get_predictions(model, x_test)
            train_scores.append(self.get_scores(y_train, pred_train))
            test_scores.append(self.get_scores(y_test, pred_test))

            y_train_all.append(y_train)
            y_test_all.append(y_test)
            pred_train_all.append(pred_train)
            pred_test_all.append(pred_test)

            # Retrieve SMILES for test indices using the previously defined list
            test_indices = test_indices_list[k]
            test_smiles = [self.smiles[idx] for idx in test_indices]

            # Calculate Tanimoto matrix for the test SMILES
            fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2) for smi in test_smiles]

            print(f"Length of fps at fold {k}: {len(fps)}")

            tanimoto_matrix = np.zeros((len(fps), len(fps)))
            for i, fp1 in enumerate(fps):
                for j, fp2 in enumerate(fps):
                    tanimoto_matrix[i, j] = FingerprintSimilarity(fp1, fp2)

            
            # Convert to NumPy array
            tanimoto_matrix_array = tanimoto_matrix

            # Calculate Vendi score
            vendi_score = vendi.score_K(tanimoto_matrix_array)

            print(f'Vendi score = {vendi_score}')

            # Calculate Vendi score
            vendi_score = vendi.score_K(tanimoto_matrix_array)
            vendi_scores.append(vendi_score)

            # Collect vendi scores in all_vendi_scores for every 3 folds (repeat)
            all_vendi_scores.append(vendi_score)

            # Every 3 folds, calculate the variance of the vendi_scores for this repeat
            if (k + 1) % 3 == 0:
                # Calculate variance of vendi scores for the current repeat (3-folds)
                vendi_variance = np.var(all_vendi_scores[-3:])  # The last 3 vendi scores
                print(f"Vendi variance after repeat {((k + 1) // 3)}: {vendi_variance}")
                
                # Append the vendi_variance to the list of variances
                vendi_variances.append(vendi_variance)
                
                # Reset the list for the next repeat
                all_vendi_scores = [] 

            # Store scores for later analysis if needed
            self.vendi_scores = vendi_scores
            self.vendi_variances = vendi_variances
          
            self.train_y = y_train_all
            self.train_pred = pred_train_all
            self.test_y = y_test_all
            self.test_pred = pred_test_all
            self.test_indices_list = test_indices_list
            self.train_scores = pd.DataFrame(data=train_scores).set_axis(["MSE","PCC"], axis=1)
            self.test_scores = pd.DataFrame(data=test_scores).set_axis(["MSE","PCC"], axis=1)
            self.model = model

    def get_predictions(self, model, x):
        """
        Obtain predictions from model.

        Args:
            model: Trained model.
            x: Descriptor data.

        Returns:
            pred: List of predicted values.
        """
        pred = model.predict(x)
        return(pred)
    
    def get_scores(self, y_true, y_pred):
        """
        Compare truth and predicted targets to obtain performance metrics.

        Args:
            y_true: List of ground truth targets.
            y_pred: List of predicted targets.

        Returns:
            scores: List containings mean squared error and Pearson correlation coefficient.
        """
        mse = mean_squared_error(y_true, y_pred)
        pcc = np.corrcoef(y_true.ravel(), y_pred.ravel())[0][1]
        scores = [mse, pcc]

        return scores


# Train and Predict with Vendi

dataset = pd.read_csv("/biggin/b229/chri6405/pept2_affinity/source_data/dataset_smiles.csv")

# Treat all 'inf' values as missing values
dataset = dataset.replace([np.inf, -np.inf], np.nan)

# find all indices for which data exists for Caco-2 (PEPT1)

indices = dataset["SKPT (PEPT2)"].dropna().index
smiles_data = dataset["SMILES"][indices].values
Ki = dataset["SKPT (PEPT2)"].dropna().values
compounds_names = dataset["Compound"][indices].values
logKi = np.log10(Ki)

# Get descriptors

descriptors_all = pd.read_csv("/biggin/b229/chri6405/pept2_affinity/source_data/dataset_morganfingerprints.csv", header=None)
descriptors = descriptors_all.iloc[indices]

descriptors_array = descriptors.drop(0, axis=1).values

# %%


def perform_regression(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42, quantify_test_set_occupancies=False):

    
    # Create the regressor instance
    rf_reg = Regressor_RandomForest(
        target=logKi,
        descriptor=descriptors_array,
        smiles=smiles_data,  # Ensure you pass the smiles_data here
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

    if quantify_test_set_occupancies:
        test_set_occupancies = [len(x) for x in predicted_data]

    # get MSE and PCC from the computed means

    MSE = np.mean((np.array(predicted_data_means) - logKi +3)**2)
    PCC = np.corrcoef(predicted_data_means, logKi-3)[0,1]

    if quantify_test_set_occupancies:
        return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC, test_set_occupancies)
    else:
        return (predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC)
    
predicted_data_means, predicted_data_stds, rf_reg, MSE, PCC, vendi_score = perform_regression(100, 10, 2, 1, 'sqrt', True, seed=42, quantify_test_set_occupancies=True)