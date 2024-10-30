# Module for neural networks

import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Model, backend
from tensorflow.keras.layers import Input, Dense, GaussianNoise, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay, mean_squared_error, f1_score, make_scorer, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, RepeatedKFold, RandomizedSearchCV, LeaveOneOut
from sklearn.utils import shuffle
from scikeras.wrappers import KerasClassifier, KerasRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from tqdm import tqdm


class Classifier():
    """ Class for classification model."""

    def __init__(self, labels, descriptor, architecture, activation, optimiser, learning_rate):
        """
        Initialise classifier instance.
        
        Args:
            labels: Array of class labels.
            descriptor: Array of descriptor data.
            architecture: List of hidden layer sizes.
            activation: Activation function.
            optimiser: Optimiser.
            learning_rate: Learning rate.    
        """
        self.label = labels
        self.desc = descriptor
        self.arch = architecture
        self.actv = activation
        self.optm = optimiser
        self.rate = learning_rate

    def build_model(self):
        """
        Construct network using input architecture and parameters.

        Returns:
            model: Compiled model.
        """
        keras.backend.clear_session()
        input_layer = Input(self.desc.shape[1])
        hidden_layer = Dropout(0.1)(input_layer)
        for layer in self.arch:
            hidden_layer = Dense(layer, activation=self.actv)(hidden_layer)

        output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)
        model = Model(input_layer, output_layer)
        model_optimiser = set_optimiser(self.optm, self.rate)
        model.compile(optimizer=model_optimiser, loss='binary_crossentropy', metrics=[mcc])
        
        return model

    def train_model_cv(self, epochs, kfolds, repeats=1, seed=42, plot_history=True, verb=False, noise=False, oversample=False, class_samples='minority', clf_threshold=0.5):
        """
        Train model and obtain predictions using (repeated) K-fold cross validation.

        Args:
            epochs: Number of epochs.
            kfolds: Number of folds for cross validation.
            repeats: Number of times to repeat kfold CV.
            seed: Random seed for generating train:test splits.
            plot_history: Whether to display accuracy during training.
            verb: Whether to output information at each epoch.
            oversample: Whether to apply SMOTE to training set.
            class_samples: The number of samples in each class after oversampling.
            clf_threshold: The threshold used to determined class predictions.
        """
        skf = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
        splits = split_data(skf, self.desc, self.label)
        train_pred_all, test_pred_all = [], []
        y_train_all, y_test_all = [], []
        train_scores, test_scores = [], []

        callback = EarlyStopping(monitor='val_loss', patience=50)

        for k in range(skf.get_n_splits()):
            if noise:
                model = self.build_noisy_model()
            else:
                model = self.build_model()

            if oversample:
                sm = SMOTE(sampling_strategy=class_samples)
                x_train, y_train = sm.fit_resample(splits['xtrain'][k],splits['ytrain'][k])
            else:
                x_train, y_train = splits['xtrain'][k], splits['ytrain'][k]

            x_train, x_test, x_scaler = scale_data(x_train, splits['xtest'][k]) # Scale for each kfold split to avoid leakage
            y_test = splits['ytest'][k]
            history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=verb, callbacks=[callback])

            if k<3 and plot_history:
                self.plot_history(history)

            train_prob, train_pred = self.get_predictions(model, x_train, clf_threshold)
            test_prob, test_pred = self.get_predictions(model, x_test, clf_threshold)
            train_pred_all.append(train_pred)
            test_pred_all.append(test_pred)
            y_train_all.append(y_train)
            y_test_all.append(y_test)
            train_scores.append(self.get_scores(y_train, train_pred))
            test_scores.append(self.get_scores(y_test, test_pred))

        self.train_true = y_train_all
        self.test_true = y_test_all
        self.train_pred = train_pred_all
        self.test_pred = test_pred_all
        self.train_scores = pd.DataFrame(data=train_scores)
        self.test_scores = pd.DataFrame(data=test_scores)
        self.train_scores.set_axis(["Accuracy","Sensitivity", "Specificity", "Precision","F1-Score","MCC"], axis=1, inplace=True)
        self.test_scores.set_axis(["Accuracy","Sensitivity", "Specificity", "Precision","F1-Score","MCC"], axis=1, inplace=True)

    def train_model_loo(self, mol_ids, epochs, val_split=0.12, plot_history=True, plot_scatter=False, verb=False, early_stopping=True, oversample=False, seed=42, clf_threshold=0.5, batch_size=32):
        """ 
        Train model and obtain predictions using leave-one-out cross validation.

        Args:
            mol_ids: List containing compound IDs/names in same order as they appear in descriptor.
            epochs: Number of epochs.
            val_split: Fraction of training data to be used for validation (i.e. not trained on).
            plot_history: Whether to display accuracy during training.
            plot_scatter: Whether to display accuracy during training.
            verb: Whether to output information at each epoch.
            early_stopping: Whether to use early stopping.
            oversample: Whether to apply SMOTE to training set.
            seed: Random seed for generating train:test splits.
            clf_threshold: The threshold used to determined class predictions.
            batch_size: Batch size for training.
        """
        loo = LeaveOneOut()
        train_scores, test_scores = [], []
        self.y_train_all, self.y_test_all, self.pred_train_all, self.pred_test_all, self.prob_train_all, self.prob_test_all, self.pred_test_ids = [], [], [], [], [], [], []
        counter = 0

        callback = EarlyStopping(monitor='val_loss', patience=50)

        with tqdm(total=len(self.label)) as pbar:

            for train_index, test_index in loo.split(self.desc):

                counter += 1
                model = self.build_model()

                x_train, y_train = self.desc[train_index], self.label[train_index].reshape(-1,1)
                x_test, y_test = self.desc[test_index], self.label[test_index].reshape(-1,1)

                if oversample:
                    sm = SMOTE(sampling_strategy='minority')
                    x_train, y_train = sm.fit_resample(x_train, y_train)

                x_train, x_test, x_scaler = scale_data(x_train, x_test)
                x_train, y_train = shuffle(x_train, y_train, random_state=seed) # Shuffle training data so validation set includes Amino and Ben cmpds

                if early_stopping == True:
                    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=verb, callbacks=[callback])
                else:
                    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=verb)

                if plot_history and counter <= 3:
                    self.plot_history(history)

                prob_train, pred_train = self.get_predictions(model, x_train, clf_threshold)
                prob_test, pred_test = self.get_predictions(model, x_test, clf_threshold)

                pred_test_id = mol_ids[test_index[0]]
                self.pred_test_ids.append(pred_test_id)

                self.y_train_all.append(y_train)
                self.y_test_all.append(y_test)

                self.prob_train_all.append(prob_train)
                self.prob_test_all.append(prob_test)

                self.pred_train_all.append(pred_train)
                self.pred_test_all.append(pred_test)

                pbar.update(1)

    def get_predictions(self, model, x, threshold=0.5):
        """ 
        Obtain predictions from model (probability) and converts prediction to integer label.

        Args:
            model: Trained model.
            x: Descriptor data.
            threshold: Threshold used to determine class predictions.

        Returns:
            y: List of probabilities that x belongs to class 1 (active).
            pred: List of predicted labels (0 or 1).
        """
        y = model.predict(x)
        pred = []

        for i in range(len(y)):
            
            if y[i] < threshold:
                pred.append(0)
            
            else:
                pred.append(1)

        return(y, pred)

    def get_scores(self, y_true, y_pred, plot_cm=False):
        """ 
        Compare truth labels and predicted labels to obtain performance metrics.

        Args:
            y_true: List of ground truth labels.
            y_pred: List of predicted labels.
            plot_cm: Whether to plot confusion matrix for this set.

        Returns:
            scores: List containing accuracy, sensitivity, specificity and precision metrics.
        """
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        if plot_cm == True:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
        total=sum(sum(cm))
        tp = cm[1,1]
        tn = cm[0,0]
        fp = cm[0,1]
        fn = cm[1,0]
        accuracy = (tp+tn)/total
        specificity = tn/(tn+fp)
        sensitivity = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1_score = (2*precision*sensitivity)/(precision+sensitivity)
        scores = [accuracy, sensitivity, specificity, precision, f1_score, mcc]

        return scores

    def save_scores(self, outpath, set="test"):
        """
        Save training or test scores as pickled dataframe.

        Args:
            set: Which set scores to save (train or test)
            outpath: Path where scores will be saved.

        """
        if set.lower() == "test":
            pickle.dump(self.test_scores,open(outpath,"wb"))
        elif set.lower() == "train":
            pickle.dump(self.train_scores,open(outpath,"wb"))

    def plot_history(self, history):
        """ 
        Plot model accuracy against training epoch.

        Args:
            history: Model history.
        """
        plt.plot(history.history['loss'],color='orange') # could also use 'acc'
        
        try:
            plt.plot(history.history['val_loss'],color='forestgreen')
        except:
            print("No validation set provided")
    
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'])
        plt.show()

    def plot_cumulative_scores(self, kfolds):
        """ 
        Plot the scores against number of k-fold CV models.

        Args:
            kfolds: Number of folds used in k-fold CV.
        """
        metric = "MCC"
        cum_mean_train = self.train_scores[metric].expanding(min_periods=3).mean()
        cum_std_train = self.train_scores[metric].expanding(min_periods=3).std(ddof=0) # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.
        cum_mean_test = self.test_scores[metric].expanding(min_periods=3).mean()
        cum_std_test = self.test_scores[metric].expanding(min_periods=3).std(ddof=0) # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.

        plt.errorbar(x=range(1,len(cum_mean_train[kfolds-1::kfolds])+1), y=cum_mean_train[kfolds-1::kfolds], yerr=cum_std_train[kfolds-1::kfolds], fmt='o', ms=8, linestyle="-", color='orange', label="Training")
        plt.errorbar(x=range(1,len(cum_mean_test[kfolds-1::kfolds])+1), y=cum_mean_test[kfolds-1::kfolds], yerr=cum_std_test[kfolds-1::kfolds], fmt='o', ms=8, linestyle="-", color='steelblue', label="Test")

        plt.title("Cumulative score and error", pad=10)
        plt.ylabel("Mean {}".format(metric), labelpad=10)
        plt.xlabel("Models", labelpad=10)
        plt.legend()
        plt.show()

    def plot_scores(model, kfolds):
        """ 
        Plot the scores against number of k-fold CV models.

        Args:
            kfolds: Number of folds used in k-fold CV.
        """
        metric = "MCC"

        mean_train = np.array([sum(model.train_scores[metric][i:i+kfolds])/kfolds for i in range(0,len(model.train_scores),kfolds)])
        cum_mean_train = [mean_train[:i].mean() for i in range(1,len(mean_train)+1)]
        cum_std_train = [mean_train[:i].std(ddof=0) for i in range(1,len(mean_train)+1)] # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.

        mean_test = np.array([sum(model.test_scores[metric][i:i+kfolds])/kfolds for i in range(0,len(model.test_scores),kfolds)])
        cum_mean_test = [mean_test[:i].mean() for i in range(1,len(mean_test)+1)]
        cum_std_test = [mean_test[:i].std(ddof=0) for i in range(1,len(mean_test)+1)] # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.

        plt.errorbar(x=np.arange(1,len(cum_mean_train)+1), y=cum_mean_train, yerr=cum_std_train, fmt='o', ms=8, linestyle="-", color='orange', label="Training")
        plt.errorbar(x=np.arange(1,len(cum_mean_test)+1), y=cum_mean_test, yerr=cum_std_test, fmt='o', ms=8, linestyle="-", color='steelblue', label="Test")

        plt.ylabel(metric, labelpad=10)
        plt.xlabel("Repeats", labelpad=10)
        plt.legend()
        plt.show()

    def plot_cm(self, set="test"):
        """ 
        Plot confusion matrix for the test/training set across all repeats.

        Args:
            set: Which set to plot ("test" or "training")
        """
        if set.lower() == "training" or set.lower() == "train":
            cm = confusion_matrix(flatten(self.train_true), flatten(self.train_pred))
        else:
            cm = confusion_matrix(flatten(self.test_true), flatten(self.test_pred))

        group_names = ['TN','FP','FN','TP']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        fig = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False)
        fig.set_xlabel('\nPredicted')
        fig.set_ylabel('\nTrue')
        fig.xaxis.set_ticklabels(['Inactive','Active'])
        fig.yaxis.set_ticklabels(['Inactive','Active'])

    def plot_roc(self):
        """
        Plot ROC curve for test set across all repeats.
        """
        y_train, y_test = flatten(self.y_train_all), flatten(self.y_test_all)
        prob_train, prob_test = flatten(self.prob_train_all), flatten(self.prob_test_all)

        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prob_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prob_test)
        auc_train, auc_test = auc(fpr_train, tpr_train), auc(fpr_test, tpr_test)

        prec_test, recall_test, thresholds_test = precision_recall_curve(y_test, prob_test)
        fscore_test = (2 * prec_test * recall_test) / (prec_test + recall_test)
        ix = np.argmax(fscore_test)

        plt.plot(fpr_train, tpr_train, label='Train AUC = {:.3f}'.format(auc_train))
        plt.plot(fpr_test, tpr_test, label='Test AUC = {:.3f}'.format(auc_test))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Best threshold={:.3f}, F-Score={:.3f}'.format(thresholds_test[ix], fscore_test[ix]))
        plt.legend(loc='best')
        plt.show()

    def randomised_search_cv(self, kfolds, repeats, iterations, param_grid, seed=42):
        """
        Perform grid search to optmise hyperparameters.

        Args:
            kfolds: Number of folds for cross validation.
            repeats: Number of times to repeat kfold CV.
            epochs: Number of epochs.
            iterations: Number of search iterations.
            param_grid (dict): Grid of parameters to search.
            seed: Random seed.

        """
        def create_model(input_shape, arch, actv, optm, lr):
            """ Construct network using input architecture and parameters.

            Returns:
                model: Constructed model.
            """
            keras.backend.clear_session()
            input_layer = Input(input_shape)
            hidden_layer = input_layer
            for layer in arch:
                hidden_layer = Dense(layer, activation=actv)(hidden_layer)

            output_layer = Dense(units=1, activation='sigmoid')(hidden_layer)
            model = Model(input_layer, output_layer)
            model_optimiser = set_optimiser(optm, lr)
            model.compile(optimizer=model_optimiser, loss='binary_crossentropy', metrics=[mcc])
            return model

        model = KerasClassifier(model=create_model, verbose=0)
        cv = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
        callback = EarlyStopping(monitor='val_loss', patience=30)
        mcc_scorer = make_scorer(matthews_corrcoef)
        search = RandomizedSearchCV(model, param_grid, n_iter=iterations, scoring=mcc_scorer, n_jobs=1, cv=cv)
        search_result = search.fit(X=self.desc, y=self.label, epochs=100, callbacks=[callback])

        print("Best: {:.4f} using {}".format(search_result.best_score_, search_result.best_params_))
        means = search_result.cv_results_['mean_test_score']
        stds = search_result.cv_results_['std_test_score']
        params = search_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("{:.4f} ± {:.3f} with: {}".format(mean, stdev, param))

        return search_result

class Regressor():
    """Class for regression model"""

    def __init__(self, target, descriptor, architecture, activation, optimiser, learning_rate, l2_reg=False):
        """
        Initialise regressor instance.
        """
        self.target = target
        self.desc = descriptor
        self.arch = architecture
        self.actv = activation
        self.optm = optimiser
        self.rate = learning_rate
        self.l2_reg = l2_reg
        self.model = self.build_model() # Build model on initialisation - overwritten when trained.

    def build_model(self):
        """
        Construct network using input architecture and parameters.

        Returns:
            model: Compiled model.
        """
        keras.backend.clear_session()
        input_layer = Input(self.desc.shape[1])
        hidden_layer = input_layer

        for layer in self.arch:
            if self.l2_reg:
                hidden_layer = Dense(layer, activation=self.actv, kernel_regularizer=l2(self.l2_reg))(hidden_layer)
            else:
                hidden_layer = Dense(layer, activation=self.actv)(hidden_layer)

        output_layer = Dense(units=1, activation='linear')(hidden_layer)
        model = Model(input_layer, output_layer)
        model_optimiser = set_optimiser(self.optm, self.rate)
        model.compile(optimizer=model_optimiser, loss='mean_squared_error', metrics=['mean_squared_error'])

        return model

    def train_model(self, epochs, test_data, repeats=1, plot_history=True, plot_scatter=False, verb=False, early_stopping=True, scale_x="MinMax", batch_size=32):
        ''' 
        Trains model and obtains predictions for the test set provided.

        Args:
            epochs: Number of epochs.
            test_data: Tuple containing x_test and y_test data.
            repeats: Number of models to train. Scores are taken as mean across n repeats.
            plot_history: Whether to display accuracy during training.
            plot_scatter: Whether to display accuracy during training.
            verb: Whether to output information at each epoch.
            early_stopping: Whether to use early stopping.
            scale_x: Whether to scale descriptor data.
            batch_size: Batch size for training.

        '''
        train_scores, test_scores = [], []
        y_train_all, y_test_all, pred_train_all, pred_test_all = [], [], [], []

        callback = EarlyStopping(monitor='val_loss', patience=50)

        for i in range(repeats):

            model = self.build_model()
            if scale_x:
                x_train, x_test, x_scaler = scale_data(self.desc, test_data[0], scale_x)
            else:
                x_train, x_test = self.desc, test_data[0]

            y_train, y_test, y_scaler = scale_data(self.target.reshape(-1,1), test_data[1].reshape(-1,1))

            if early_stopping == True:
                history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=verb, callbacks=[callback])
            else:
                history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=verb)

            if plot_history:
                self.plot_history(history)

            pred_train = self.get_predictions(model, x_train, y_scaler)
            pred_test = self.get_predictions(model, x_test, y_scaler)

            train_scores.append(self.get_scores(y_train, pred_train, y_scaler))
            test_scores.append(self.get_scores(y_test, pred_test, y_scaler))

            if plot_scatter:
                if y_scaler:
                    y_train_all.append(y_scaler.inverse_transform(y_train))
                    y_test_all.append(y_scaler.inverse_transform(y_test))
                else:
                    y_train_all.append(y_train)
                    y_test_all.append(y_test)

                pred_train_all.append(pred_train)
                pred_test_all.append(pred_test)

        self.train_y = y_train_all
        self.train_pred = pred_train_all
        self.test_y = y_test_all
        self.test_pred = pred_test_all
        self.train_scores = pd.DataFrame(data=train_scores).set_axis(["MSE","PCC"], axis=1)
        self.test_scores = pd.DataFrame(data=test_scores).set_axis(["MSE","PCC"], axis=1)
        self.model = model

        if plot_scatter:
            self.plot_scatter(y_train_all,y_test_all,pred_train_all,pred_test_all)

    def train_model_cv(self, epochs, kfolds, repeats=1, seed=42, plot_history=True, plot_scatter=True, verb=False, early_stopping=True, scale_x="MinMax", batch_size=32):
        """
        Train model and obtain predictions using (repeated) K-fold cross validation.

        Args:
            epochs: Number of epochs.
            kfolds: Number of folds for cross validation.
            repeats: Number of times to repeat kfold CV.
            seed: Random seed.
            plot_history: Whether to display accuracy during training.
            plot_scatter: Whether to display accuracy during training.
            verb: Whether to output information at each epoch.
            early_stopping: Whether to use early stopping.
            scale_x: Whether to scale descriptor data.
            batch_size: Batch size for training.
        """
        kf = RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
        splits, test_indices_list = split_data(kf, self.desc, self.target)
        train_scores, test_scores = [], []
        y_train_all, y_test_all, pred_train_all, pred_test_all = [], [], [], []

        callback = EarlyStopping(monitor='val_loss', patience=50)

        for k in range(kf.get_n_splits()):

            model = self.build_model()
            
            if scale_x:
                x_train, x_test, x_scaler = scale_data(splits['xtrain'][k], splits['xtest'][k], scale_x) # Scale for each kfold split to avoid leakage
            else:
                x_train, x_test = splits['xtrain'][k], splits['xtest'][k]

            y_train, y_test, y_scaler = scale_data(splits['ytrain'][k].reshape(-1,1), splits['ytest'][k].reshape(-1,1))

            if early_stopping == True:
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=verb, callbacks=[callback])
            else:
                history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=verb)

            if k<3 and plot_history:
                self.plot_history(history)

            pred_train = self.get_predictions(model, x_train, y_scaler)
            pred_test = self.get_predictions(model, x_test, y_scaler)
            train_scores.append(self.get_scores(y_train, pred_train, y_scaler))
            test_scores.append(self.get_scores(y_test, pred_test, y_scaler))

            if plot_scatter:
                if y_scaler:
                    y_train_all.append(y_scaler.inverse_transform(y_train))
                    y_test_all.append(y_scaler.inverse_transform(y_test))
                else:
                    y_train_all.append(y_train)
                    y_test_all.append(y_test)

                pred_train_all.append(pred_train)
                pred_test_all.append(pred_test)
        
        self.train_y = y_train_all
        self.train_pred = pred_train_all
        self.test_y = y_test_all
        self.test_pred = pred_test_all
        self.test_indices_list = test_indices_list
        self.train_scores = pd.DataFrame(data=train_scores).set_axis(["MSE","PCC"], axis=1)
        self.test_scores = pd.DataFrame(data=test_scores).set_axis(["MSE","PCC"], axis=1)
        self.model = model

        if plot_scatter:
            self.plot_scatter(y_train_all,y_test_all,pred_train_all,pred_test_all)
            
            

    def train_model_loo(self, mol_ids, epochs, val_split=0.1, plot_history=True, plot_scatter=False, verb=False, early_stopping=True, seed=42, scale_x="MinMax", batch_size=32):
        """
        Train model and obtain predictions using leave-one-out cross validation.

        Args:
            mol_ids: List containing compound IDs/names in same order as they appear in descriptor.
            epochs: Number of epochs.
            val_split: Fraction of training data to be used for validation (i.e. not trained on).
            plot_history: Whether to display accuracy during training.
            plot_scatter: Whether to display accuracy during training.
            verb: Whether to output information at each epoch.
            early_stopping: Whether to use early stopping.
            seed: Random seed for generating train:test splits.
            scale_x: Whether to scale descriptor data.
            batch_size: Batch size for training.
        """
        loo = LeaveOneOut()
        train_scores, test_scores = [], []
        self.y_train_all, self.y_test_all, self.pred_train_all, self.pred_test_all, self.pred_test_ids = [], [], [], [], []
        self.all_shap_values = []
        counter = 0

        callback = EarlyStopping(monitor='val_loss', patience=50)

        with tqdm(total=len(self.target)) as pbar:

            for train_index, test_index in loo.split(self.desc):

                counter += 1
                model = self.build_model()

                if scale_x:
                    x_train, x_test, x_scaler = scale_data(self.desc[train_index], self.desc[test_index], scale_x)
                else:
                    x_train, x_test = self.desc[train_index], self.desc[test_index]

                y_train, y_test, y_scaler = scale_data(self.target[train_index].reshape(-1,1), self.target[test_index].reshape(-1,1))
                x_train, y_train = shuffle(x_train, y_train, random_state=seed) # Shuffle training data

                if early_stopping:
                    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=verb, callbacks=[callback])
                else:
                    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, verbose=verb)

                if plot_history and counter <= 3:
                    self.plot_history(history)

                pred_train = self.get_predictions(model, x_train, y_scaler)
                pred_test = self.get_predictions(model, x_test, y_scaler)

                pred_test_id = mol_ids[test_index[0]]
                self.pred_test_ids.append(pred_test_id)

                self.y_train_all.append(y_scaler.inverse_transform(y_train))
                self.y_test_all.append(y_scaler.inverse_transform(y_test))

                self.pred_train_all.append(pred_train)
                self.pred_test_all.append(pred_test)

                pbar.update(1)

    def get_predictions(self, model, x, scaler):
        """
        Obtain predictions from model and rescale.

        Args:
            model: Trained model.
            x: Descriptor data.
            scaler: Scaler used to transform the data.

        Returns:
            pred: List of predicted values.
        """
        y = model.predict(x)
        if scaler:
            pred = scaler.inverse_transform(y)
        else:
            pred = y

        return(pred)

    def get_scores(self, y_true, y_pred, scaler=None):
        """
        Compare truth and predicted targets to obtain performance metrics.

        Args:
            y_true: List of ground truth targets (scaled).
            y_pred: List of predicted targets.
            scaler: Scaler used to transform the data.

        Returns:
            scores: List containings mean squared error, Pearson correlation coefficient and accuracy.
        """
        if scaler is not None:
            y_true = scaler.inverse_transform(y_true)

        mse = mean_squared_error(y_true, y_pred)
        
        try:
            pcc = np.corrcoef(y_true.ravel(), y_pred.ravel())[0][1]
        except:
            pcc = np.corrcoef(y_true, y_pred)[0][1]
        
        scores = [mse, pcc]

        return scores

    def plot_history(self, history):
        """ 
        Plot model accuracy against training epoch.

        Args:
            history: Model history.
        """
        fig = plt.figure(facecolor='w', edgecolor='k', figsize=(4,4))
        plt.plot(history.history['mean_squared_error'],color='steelblue') # could also use 'acc'

        try:
            plt.plot(history.history['val_mean_squared_error'],color='darkorange')
        except:
            print("No validation set provided")

        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'])
        plt.show()

    def plot_scatter(self, train_true, test_true, train_pred, test_pred):
        """ 
        Plot scatter plot of true vs predicted values.

        Args:
            train_true: Target values for training set.
            test_true: Target values for test set.
            train_pred: Predicted values for training set.
            test_pred: Predicted values for test set.
        """
        fig = plt.figure(facecolor='w', edgecolor='k', figsize=(4,4))

        for i in range(len(train_true)):
            plt.scatter(x=flatten(train_true), y=flatten(train_pred), color='steelblue', alpha=0.8, s=20)
            plt.scatter(x=flatten(test_true), y=flatten(test_pred), color='firebrick', alpha=0.8, s=20)

        plt.tick_params(axis='both', which='major')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()

    def plot_scores(self, kfolds, metric):
        """ 
        Plot the scores against number of k-fold CV selfs.

        Args:
            kfolds: Number of folds used in k-fold CV.
            metric: Metric to plot ("MSE" or "PCC")
        """

        mean_train = np.array([sum(self.train_scores[metric][i:i+kfolds])/kfolds for i in range(0,len(self.train_scores),kfolds)])
        cum_mean_train = [mean_train[:i].mean() for i in range(1,len(mean_train)+1)]
        cum_std_train = [mean_train[:i].std(ddof=0) for i in range(1,len(mean_train)+1)] # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.

        mean_test = np.array([sum(self.test_scores[metric][i:i+kfolds])/kfolds for i in range(0,len(self.test_scores),kfolds)])
        cum_mean_test = [mean_test[:i].mean() for i in range(1,len(mean_test)+1)]
        cum_std_test = [mean_test[:i].std(ddof=0) for i in range(1,len(mean_test)+1)] # ddof=0 means dividing by n to calculate std. Default ddof=1 is dividing by n-1.

        plt.errorbar(x=np.arange(1,len(cum_mean_train)+1), y=cum_mean_train, yerr=cum_std_train, fmt='o', ms=8, linestyle="-", color='orange', label="Training")
        plt.errorbar(x=np.arange(1,len(cum_mean_test)+1), y=cum_mean_test, yerr=cum_std_test, fmt='o', ms=8, linestyle="-", color='steelblue', label="Test")

        plt.ylabel(metric, labelpad=10)
        plt.xlabel("Repeats", labelpad=10)
        plt.legend()
        plt.show()

    def randomised_search_cv(self, kfolds, repeats, iterations, param_grid, seed=42):
        """
        Perform grid search to optmise hyperparameters.

        Args:
            kfolds: Number of folds for cross validation.
            repeats: Number of times to repeat kfold CV.
            epochs: Number of epochs.
            iterations: Number of search iterations.
            param_grid (dict): Grid of parameters to search.
            seed: Random seed.

        """
        def create_model(input_shape, arch, actv, optm, lr):
            """ 
            Construct network using input architecture and parameters.

            Returns:
                model: Constructed model.
            """
            keras.backend.clear_session()
            input_layer = Input(input_shape)
            hidden_layer = input_layer
            for layer in arch:
                hidden_layer = Dense(layer, activation=actv)(hidden_layer)

            output_layer = Dense(units=1, activation='linear')(hidden_layer)
            model = Model(input_layer, output_layer)
            model_optimiser = set_optimiser(optm, lr)
            model.compile(optimizer=model_optimiser, loss='mean_squared_error', metrics=['mean_squared_error'])
            return model

        model = KerasRegressor(model=create_model, verbose=0, validation_split=0.1)
        cv = RepeatedKFold(n_splits=kfolds, n_repeats=repeats, random_state=seed)
        callback = EarlyStopping(monitor='val_loss', patience=50)
        mse_scorer = make_scorer(mean_squared_error)
        search = RandomizedSearchCV(model, param_grid, n_iter=iterations, scoring='neg_mean_squared_error', n_jobs=1, cv=cv)

        desc, target = shuffle(self.desc, self.target, random_state=seed) # Shuffle so validation set includes Amino and Ben cmpds
        search_result = search.fit(X=desc, y=target, epochs=200, callbacks=[callback])

        print("Best result:\n {:.4f} using {}\n".format(search_result.best_score_, search_result.best_params_))
        means = search_result.cv_results_['mean_test_score']
        stds = search_result.cv_results_['std_test_score']
        params = search_result.cv_results_['params']

        print("All results:")
        for mean, stdev, param in zip(means, stds, params):
            print("{:.4f} ± {:.3f} with: {}".format(mean, stdev, param))

        return search_result.best_score_, search_result.best_params_

    def save_model(self, filename):
        """
        Save model to file.

        Args:
            filename: Name of file to save model to.
        """
        self.model.save(filename)





from sklearn.ensemble import RandomForestRegressor


class Regressor_RandomForest():
    """Class for random forest regression model"""

    def __init__(self, target, descriptor, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, seed=42):
        """
        Initialise regressor instance.
        """
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
        self.test_scores = None
        self.train_scores = None
        
    
    def build_model(self):
        """
        Construct random forest model using input parameters.

        Returns:
            model: Compiled model.
        """
        model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, bootstrap=self.bootstrap, random_state=self.seed)
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
        splits, test_indices_list = split_data(kf, self.desc, self.target)
        train_scores, test_scores = [], []
        y_train_all, y_test_all, pred_train_all, pred_test_all = [], [], [], []

        for k in range(kf.get_n_splits()):

            model = self.build_model()
            x_train, x_test = splits['xtrain'][k], splits['xtest'][k]
            y_train, y_test = splits['ytrain'][k], splits['ytest'][k]

            model.fit(x_train, y_train)

            pred_train = self.get_predictions(model, x_train)
            pred_test = self.get_predictions(model, x_test)
            train_scores.append(self.get_scores(y_train, pred_train))
            test_scores.append(self.get_scores(y_test, pred_test))

            y_train_all.append(y_train)
            y_test_all.append(y_test)

            pred_train_all.append(pred_train)
            pred_test_all.append(pred_test)
        
        self.train_y = y_train_all
        self.train_pred = pred_train_all
        self.test_y = y_test_all
        self.test_pred = pred_test_all
        self.test_indices_list = test_indices_list
        self.train_scores = pd.DataFrame(data=train_scores).set_axis(["MSE","PCC"], axis=1)
        self.test_scores = pd.DataFrame(data=test_scores).set_axis(["MSE","PCC"], axis=1)
        self.model = model

    def train_model_custom_split(self, training_indices, test_indices):
        # Extract training and test sets
        x_train, x_test = self.desc[training_indices], self.desc[test_indices]
        y_train, y_test = self.target[training_indices], self.target[test_indices]

        # Proceed with model training if no issues found
        model = self.build_model()
        model.fit(x_train, y_train)

        # Get predictions for training and test sets
        pred_train = self.get_predictions(model, x_train)
        pred_test = self.get_predictions(model, x_test)

        # Store the actual and predicted values
        self.y_train = y_train
        self.pred_train = pred_train
        self.y_test = y_test
        self.pred_test = pred_test
        self.model = model
        
        # Calculate scores and store them as attributes of the instance
        self.train_scores = pd.DataFrame(data=[self.get_scores(y_train, pred_train)], columns=["MSE", "PCC"])
        self.test_scores = pd.DataFrame(data=[self.get_scores(y_test, pred_test)], columns=["MSE", "PCC"])   

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

    def get_feature_importances(self):
        """
        Get feature importances from model.

        Returns:
            feature_importances: List of feature importances.
        """
        feature_importances = self.model.feature_importances_
        return feature_importances


def split_data(kfold, desc, target, ids=None):
    """ 
    Create train and test splits.

    Args:
        kfold: sklearn KFold object.
        desc: Descriptor vector.
        target: Target values or labels.
        ids: Target IDs or names.

    Returns:
        splits: Dictionary containing data splits for each fold
        test_indices_list: list of indices of test set.
    """
    splits = {'xtrain':[], 'xtest':[], 'ytrain':[], 'ytest':[], 'id_train':[], 'id_test':[]}

    test_indices_list = []

    for train_index, test_index in kfold.split(X=desc, y=target):
        test_indices_list.append(test_index)
        splits['xtrain'].append(desc[train_index,:])
        splits['xtest'].append(desc[test_index,:])
        splits['ytrain'].append(target[train_index])
        splits['ytest'].append(target[test_index])
        if ids is not None:
            splits['id_train'].append(ids[train_index])
            splits['id_test'].append(ids[test_index])

    return (splits, test_indices_list)

def set_optimiser(optm, rate):
    """ 
    Set optimiser for backpropagation.

    Args:
        optm: Optimiser.
        rate: Learning rate.
    """
    if optm == 'SGD':
        output = optimizers.SGD(learning_rate=rate)
    elif optm == 'RMSprop' :
        output = optimizers.RMSprop(learning_rate=rate)
    elif optm == 'Adam' :
        output = optimizers.Adam(learning_rate=rate)
    elif optm == 'Adadelta' :
        output = optimizers.Adadelta(learning_rate=rate)
    elif optm == 'Adagrad' :
        output = optimizers.Adagrad(learning_rate=rate)
    elif optm == 'Adamax' :
        output = optimizers.Adagrad(learning_rate=rate)
    else:
        print("Error: Keras optimizer not provided")
        return

    return output

def scale_data(train,  test=[], scaler_type="MinMax", range=(0,1)):
    """ 
    Scale data between given range using MinMaxScaler. 
    Scaler is fit to the training data and then used to rescale both training and test/prediction space.

    Args:
        range: Output range for rescaled data.
        train: Training set.
        test: Test set.

    """
    if scaler_type == "MinMax":
        scaler = MinMaxScaler(feature_range=range)
    elif scaler_type == "Standard":
        scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train)
    if len(test) > 0:
        test_scaled = scaler.transform(test)
        return train_scaled, test_scaled, scaler
    else:
        return train_scaled, scaler

def mcc(y_true, y_pred):
    """ 
    Matthews Correlation Coefficient for keras.fit() method.
    """
    K = keras.backend
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def f1(y_true, y_pred):
    """ 
    F1-score function for keras.fit() method.
    """
    K = keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def flatten(nested_list):
    """
    Flatten nested list.

    Args:
        nested_list: Nested list.

    Returns:
        flat_list: Flattened list.
    """
    flat_list = list(np.concatenate(nested_list). flat)
    return(flat_list)
