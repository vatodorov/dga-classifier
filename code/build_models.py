########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################

import pandas as pd
import sys
import gc
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import mleap.sklearn.preprocessing.data
import mleap.sklearn.pipeline
from mleap.sklearn.preprocessing.data import FeatureExtractor
from mleap.sklearn.ensemble import forest
import pickle
from scipy import stats
import statsmodels.api as statslr
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, f1_score, classification_report, \
    accuracy_score, precision_score, recall_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline

import dga_classifier.data as data
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.cross_validation import train_test_split




# Import the modeling utilities
sys.path.insert(0, '/Users/valentint/Documents/GitRepos/modelingpipeline/utility_functions')
import modeling_utilities as utils

# Environment variables
shap.initjs()
output_path = '/Users/valentint/Documents/GitRepos/dga-classifier/results/'
seed_value = 7894
sample_size = 0.1
target = 'dga_domain'
test_size = 0.70


# Read in the data
df = pd.read_csv('/Users/valentint/Documents/GitRepos/dga-classifier/data/project-data.csv', low_memory=False)


print(df.head(10))














# Cleanup of the data
# Drop variables I don't need
drop_vars = ['home_ownership', 'application_type', 'verification_status',
             'grade', 'loan_status', 'addr_state', 'purpose', 'delinq_2yrs',
             'emp_length']

df.drop(drop_vars, axis=1, inplace=True)

# Drop rows with NAs
df.dropna(inplace=True)

# Garbage collection
gc.disable()
gc.collect()



# Downsample the data
analysis_set = df.sample(frac=sample_size,
                         replace=False,
                         weights=None,
                         random_state=seed_value)

# Create samples for training and testing
x_train, x_test, y_train, y_test = train_test_split(analysis_set.drop(target, axis=1),
                                                    analysis_set[target],
                                                    test_size=test_size,
                                                    random_state=seed_value)



### LSTM model

def eg_build_lstm_model(max_features, maxlen):
    """
    Create the LSTM model

    """

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

    return model




### CNN model






### Evaluate LSTM & CNN models

# Predict on training set and check the accuracy
utils.modelAccuracyStats(model, 'Random Forest', x_train, y_train, x_test, y_test)

# Overfit test
probability = model.predict_proba(x_train)[:, 1]
y_train_ks = pd.DataFrame({'target': y_train,
                           'probability': probability})

probability = model.predict_proba(x_test)[:, 1]
y_test_ks = pd.DataFrame({'target': y_test,
                          'probability': probability})

utils.compare_train_test(x_train, y_train_ks, x_test, y_test_ks, bins=30)







### Features extraction - Ratios





### Build a Random Forest model using the ratios

# Grid search for finetuning parameters for RF
cv_params = {'n_estimators': [100, 150, 200],
             'max_depth': [7, 10, 15],
             'class_weight': [None, 'balanced']}

rf_params = {'n_jobs': -1,
             'random_state': seed_value,
             'verbose': 0,
             'criterion': 'gini'}

utils.model_gridsearch_cv(x_train, y_train,
                          estimation_method=RandomForestClassifier,
                          model_params=rf_params, cv_params=cv_params,
                          evaluation_objective='accuracy', number_cv_folds=3, verbose=True)

# Estimate a random forest model
# TO DO: Add a GridSearchCV for parameters
rf_params = {'n_estimators': 2,
             'max_depth': 8,
             'n_jobs': -1,
             'random_state': seed_value,
             'verbose': 0,
             'criterion': 'gini',
             'class_weight': 'balanced'}

model = RandomForestClassifier(**rf_params)
model.fit(x_train, y_train)

# Predict on training set and check the accuracy
utils.modelAccuracyStats(model, 'Random Forest', x_train, y_train, x_test, y_test)

# Overfit test
probability = model.predict_proba(x_train)[:, 1]
y_train_ks = pd.DataFrame({'target': y_train,
                           'probability': probability})

probability = model.predict_proba(x_test)[:, 1]
y_test_ks = pd.DataFrame({'target': y_test,
                          'probability': probability})

utils.compare_train_test(x_train, y_train_ks, x_test, y_test_ks, bins=30)

# SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# Summarize the effects of all the features
shap.summary_plot(shap_values, x_train, plot_type='bar', max_display=20)

# Visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0, :], x_train.iloc[0, :], link='identity')

# Create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot('ratio_prior_veh_incentive_msrp', shap_values, x_train)




### Build an XGB model using the ratios
