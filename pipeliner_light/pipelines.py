"""
this file create a class call ClassicPipe.
the goal of this class is to store and load all the data from the previously trained models into dictionnaries.
"""

import json
import os

import joblib
import pandas as pd

from .algos import scalers_dict

pd.options.mode.chained_assignment = None
import numpy as np

class ClassicPipe:
#------------------------------------------------------------------------------------------------------------
    #initial build of the class, this part sets all the initial values
    def __init__(self, features, estimator='lgbm', classification=False, scaler='standard',
                 loader=None, y_endpoint=None, feature_selection=None, load_from=None):

        assert scaler in scalers_dict.keys(), 'Your scaler is not supported. Supported scalers: standard, minmax '
       
        self.scaler = scaler
        self.feature_selection = feature_selection
        self.features = features  #descriptors list
        self.load_from = load_from
        self.classification = classification
        self.y_endpoint = y_endpoint
        self.features_importances = []
        self.n_estimators = []

        if loader:
            #if there is a loader, load data
            self.fitted_scalers = loader['fitted_scalers']
            self.fitted_selectors = loader['fitted_selectors']
            self.fitted_estimators = loader['fitted_estimators']
            self.fitted_estimators_perf = loader['fitted_estimators_perf']
            self.optimal_params = loader['optimal_params']
            self.estimator = estimator
        else:
            #initiating list for data
            self.fitted_scalers = []
            self.fitted_selectors = []
            self.fitted_estimators = []
            self.fitted_estimators_perf = {"train": [], "val": []}
            self.optimal_params = {}
            self.estimator = estimator
            #what type of LGBM(Regressor or Classifier)
            if self.classification:
                self.estimator += '_class'
            else:
                self.estimator += '_reg'
#------------------------------------------------------------------------------------------------------------
 
#------------------------------------------------------------------------------------------------------------
    #the goal of this part is to run each model and then calculate the mean value and return it into an array
    def predict_vector(self, feature_vector):
        predictions_list = []
        feature_vector = feature_vector.reshape(1, -1)

        for scaler, selector, estimator in zip(self.fitted_scalers, self.fitted_selectors, self.fitted_estimators):
            feature_vector_scaled = scaler.transform(feature_vector)
            if self.feature_selection is not None:
                feature_vector_scaled = feature_vector_scaled[:, selector]
            predictions_list.append(estimator.predict(feature_vector_scaled))

        #calculate the mean value of the predictions
        mean_prediction = np.array(predictions_list).mean(axis=0)[0]

        return mean_prediction
#------------------------------------------------------------------------------------------------------------
 
    @classmethod
    def load(cls, model_folder):
        #initialisation of dictonnaries
        loader = {}
        fitted_estimators_perf = {"train": [], "val": []}

        #init lists
        fitted_estimators = []
        fitted_selectors = []
        fitted_scalers = []

        #gets the names of each row and puts it into a list 
        perf_df = pd.read_csv(os.path.join(model_folder, 'performance.csv'))

        #removing space after the values
        perf_df.columns = perf_df.columns.str.strip()
        perf_df['set'] = perf_df['set'].astype(str).str.strip()

        metrics = list(perf_df.columns.values)[1:]

        #in the performance.csv file, there is a column call set
        #this code verifiy this column which contain either train or val
        #those are 2 dictionnaries which are compose of other dictionnaries who have for keys the values in metrics
        #(adjusted_r2 ,r2 ,p_corr ,k_tau ,rmse ,mae) each key is map to all the value of is column in the perfomance.csv file
        for i, row in perf_df.iterrows():
            try:
                fitted_estimators_perf[row['set']].append({metric: row[metric] for metric in metrics})
                #if there is no set in file, skip
            except KeyError:
                pass

#------------------------------------------------------------------------------------------------------------
        #loading data from predicted_training_set
        cv_predicted_df = pd.read_csv(os.path.join(model_folder, 'cv_predicted_training_set.csv'))

        #get the initial params(features(type->DESC),scaler,estimator,y_endpoint,feature_selection,classification)
        with open(os.path.join(model_folder, 'args.json')) as args_in:
            args_dict = json.load(args_in)

       #pass through all the models and retreive the data in scaler.sav, selector.npy and estimator.sav
        for i in range(0, 10000):
            try:
                scaler = joblib.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'scaler.sav'))
                estimator = joblib.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'estimator.sav'))

                #if in the args.json file feature_selection != none then load the selector
                if args_dict['feature_selection'] is not None:
                    selector = np.load(os.path.join(model_folder, 'model_{!s}'.format(i), 'selector.npy'))
                
                #make add the data to a list for each model
                    fitted_selectors.append(selector)
                fitted_scalers.append(scaler) #either a StandardScaler or a MinMaxScaler
                fitted_estimators.append(estimator) #either a LGBMRegressor or LGBMClassifier

            #if we cant find anymore file we get out of the loop
            except FileNotFoundError as error:
                print('Have read {!s} models'.format(i))
                break
#------------------------------------------------------------------------------------------------------------

        #adding to a loader(dictionnary) the scalers,selectors,estimators,estimators_perf and predicted_df values
        loader['fitted_scalers'] = fitted_scalers
        if args_dict['feature_selection'] is not None:
            loader['fitted_selectors'] = fitted_selectors
        loader['fitted_estimators'] = fitted_estimators
        loader['fitted_estimators_perf'] = fitted_estimators_perf
        loader['cv_predicted_df'] = cv_predicted_df

        #get the optimal params and add it into the loader
        with open(os.path.join(model_folder, 'params.json')) as params_in:
            params_dict = json.load(params_in)
        loader['optimal_params'] = params_dict

        #adding loader data into args_dict and returning it
        args_dict['loader'] = loader
        return cls(**args_dict)
#------------------------------------------------------------------------------------------------------------
 