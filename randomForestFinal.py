import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import random
# Sklearn models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from prettytable import PrettyTable
import matplotlib.pyplot as plt

import pickle
import os.path

from dataloader2 import Dataloader2

class RandomForest():

    def __init__(self, data_path='BikeSharing-Dataset/hour.csv'):

        # Make results reproducible
        random.seed(100)
        
        # Load data form bike sharing csv
        self.data = {}
        dataloader2 = Dataloader2(data_path)
        self.data['all'] = dataloader2.getAllData()

        # Define feature and target variables
        self.features= ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
        self.target = ['cnt']

        # Convert pandas frame into samples and labels
        self.samples, self.labels = {}, {}
        self.samples['all'] = self.data['all'][self.features].values
        self.labels['all'] = self.data['all'][self.target].values.ravel()

        # Define model 
        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=4,
           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
           oob_score=False, random_state=100, verbose=0, warm_start=False)


    def GridSearchCVparameter(self, iter=100):


        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True]
        # Create the random grid
        grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        self.model = GridSearchCV(estimator = rf, param_distributions = grid, n_iter =iter, cv = 5, random_state=0)

        return self.model.get_params()

    def train(self, samples, labels):

       #training of model
        assert(len(samples) == len(labels))
        self.model.fit(samples, labels)

    def test(self, samples, labels):

        # Testing of model
        if self.model is None:
            print("Please load or train a model before!")
            return

        assert (len(samples) == len(labels))
        pred = self.model.predict(samples)
        mse = mean_squared_error(labels, pred)
        rmse = np.sqrt(mean_squared_error(labels, pred))
        mae = mean_absolute_error(labels, pred)
        score = self.model.score(samples, labels)    
        rmsle = np.sqrt(mean_squared_log_error(labels, pred))

        return { 'mse':mse, 'rmse': rmse, 'mae': mae, 'score': score, 'rmsle': rmsle}

    def kFoldCrossvalidation(self): 

        table = PrettyTable()
        table.field_names = ["Model", "Mean Squared Error", "Root Mean Squared Error", "Mean Absolute Error", 'RMSLE', "R² score"]

        kf = KFold(n_splits=10, shuffle=True, random_state=100)

        result = []
        split = 1
        
        for train, test in kf.split(self.samples['all']):

            samples, labels = {}, {}
            samples['train'], labels['train'] = self.selData(train)
            samples['test'], labels['test'] = self.selData(test)

            self.train(samples['train'], labels['train'])
            result.append(self.test(samples['test'], labels['test']))    
            table.add_row([type(self.model).__name__, format(result[-1]['mse'], '.3f'), format(result[-1]['rmse'], '.3f'), format(result[-1]['mae'], '.3f'), format(result[-1]['rmsle'], '.3f'), format(result[-1]['score'], '.3f')])
        
        mse = np.mean([item['mse'] for item in result])
        rmse = np.mean([item['rmse'] for item in result])
        mae = np.mean([item['mae'] for item in result])
        score = np.mean([item['score'] for item in result])
        rmsle = np.mean([item['rmsle'] for item in result])

        table.add_row([type(self.model).__name__, format(mse, '.3f'), format(rmse, '.3f'), format(mae, '.3f'), format(rmsle, '.3f'), format(score, '.3f')])
        print(table)

    def selData(self, index_list):

        samples = [self.samples['all'] [i] for i in index_list]
        labels = [self.labels['all'] [i] for i in index_list]

        return samples, labels

    
if __name__ == "__main__":

    model = RandomForest()
    model.kFoldCrossvalidation()

