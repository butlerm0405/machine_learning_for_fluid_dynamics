#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from pipelinehelper import PipelineHelper
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('winequality-red.csv', delimiter = ";")
df.shape
X = df.values[:,0:-1]
y = df.values[:,-1]
print(df.head())

# Base Random Forest Model
rfc = RandomForestClassifier()
scores = cross_val_score(rfc, X, y, cv=5)
print("Base Random Forest Model Accuracy")
print(scores)
print("Average Random Forest Model Accuracy")
print(scores.mean())


# Report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")

#Pipeline
pipe = Pipeline([
    # Data Scaling
    ('scaler', PipelineHelper([
        ('minmax', MinMaxScaler()),
        ('max', MaxAbsScaler()),
        ('robust', RobustScaler()),
    ])),
    # Feature Selection 
    ('features', PipelineHelper([
        ('var', VarianceThreshold()),
        ('kbest', SelectKBest()),
    ])),
    # Classifiers
    ('classifier', PipelineHelper([
        ('rf', RandomForestClassifier()),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC()),
        ('ada', AdaBoostClassifier())
    ])),
])

#Parameters
params = {
    # Data Scaling
    'scaler__selected_model': pipe.named_steps['scaler'].generate({
        'minmax__copy': [True],
        'max__copy': [True],
        'robust__quantile_range': [(25.0,75.0), (15.0,75.0), (25.0,85.0)],
    }),
    # Feature Selection
    'features__selected_model': pipe.named_steps['features'].generate({
        'var__threshold': [0, 0.5, 0.9],
        'kbest__k': [3, 9, 11],
    }),
    # Hyperparmeters
    'classifier__selected_model': pipe.named_steps['classifier'].generate({
        'rf__n_estimators': [100, 500],
        'rf__max_features': [1.0, 'sqrt'],
        'knn__n_neighbors': [1,2,3,4,5,6,7,8,9,10],
        'knn__weights': ['uniform', 'distance'],
        'svm__kernel' : ['linear', 'rbf'],
        'svm__C' : [0.1, 1, 10, 100],
        'ada__n_estimators' : [10, 50, 100],
        'ada__learning_rate' : [0.01, 1, 5, 10],
    })
}

n_iter_search = 1320
random_search = GridSearchCV(
    pipe, 
    params,
    cv = 3,
    scoring='accuracy',  
    #n_iter=n_iter_search,
    verbose=1,)

start = time()
random_search.fit(X, y)
end = time()
print(f"RandomizedSearchCV took {(end-start):.2f} seconds for {n_iter_search} candidates.")
report(random_search.cv_results_)
#print(grid.best_params_)
#print(grid.best_score_)

