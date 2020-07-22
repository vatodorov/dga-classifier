import pandas as pd
import numpy as np
import sys
import gc
import shap
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, f1_score, classification_report, \
    accuracy_score, precision_score, recall_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.pipeline import Pipeline


# Import the modeling utilities
sys.path.insert(0, '/Users/valentint/Documents/GitRepos/GitHub/modelingpipeline/utility_functions')
import modeling_utilities as utils

# Parameters
shap.initjs()
output_path = '/Users/valentint/Documents/GitRepos/GitHub/ml-projects/data/'
seed_value = 7894
sample_size = 0.1
target = 'dga_domain'
test_size = 0.33

# TO DO:
    # Use Decision tree to determine how to create the groups for the categorical varaibles
    # Create categories from the state variables ('addr_state') - for now, I just drop it

# Read in the data
df = pd.read_csv('/Users/valentint/Documents/Data/LendingClub/loan.csv', low_memory=False)

keep_vars = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'grade',
             'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status', 'purpose',
             'addr_state',
             'dti', 'delinq_2yrs', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
             'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'collections_12_mths_ex_med',
             'application_type']

df = df[keep_vars]
print(df.head(10))

# Explore the intersection between 'loan_status' and 'delinq_2yrs' for defining the target
delinq_cross_tab = pd.crosstab([df['loan_status'], df['delinq_2yrs']], columns='count')
delinq_cross_tab.to_csv(output_path + 'cross_tab.csv')


# Create the target variable
def create_target(row):
    if row['loan_status'] == 'Charged Off' or \
            row['loan_status'] == 'Default' or \
            row['loan_status'] == 'Does not meet the credit policy. Status:Charged Off' or \
            row['loan_status'] == 'Late (31-120 days)':
        return 1
    elif row['delinq_2yrs'] > 0:
        return 1
    else:
        return 0


df['bad_loan'] = df.apply(lambda x: create_target(x), axis=1)

# Features engineering
df['int_rate'] = df['int_rate'] / 100
df['dti'] = df['dti'] / 100
df['revol_util'] = df['revol_util'] / 100

df['emp_length'].fillna('0 years', inplace=True)
df['emp_length_int'] = df['emp_length'].str.extract(r'([year])?(\d+)')[1]
df['emp_length_int'] = df['emp_length_int'].apply(lambda x: int(x))
df['emp_length_int'].value_counts().sort_values().plot(kind='barh', figsize=(14, 8))

df['home_ownership'].value_counts().sort_values().plot(kind='barh', figsize=(14, 8))
df['rent'] = np.where(df['home_ownership'] == 'RENT', 1, 0)

df['application_type'].value_counts().sort_values().plot(kind='barh', figsize=(14, 8))
df['application_type_individual'] = np.where(df['application_type'] == 'INDIVIDUAL', 1, 0)

df['term'] = df['term'].str.extract(r'[a-z]?(\d+)', expand=False)
df['term'] = df['term'].apply(lambda x: int(x))

# Regardless of the verification status, the default rate is the same
table = pd.crosstab(df['verification_status'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

# Loan grade
# Create an ordinal variable from the 'grade' variable
table = pd.crosstab(df['grade'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


def grade_categories(row):
    if row['grade'] == 'A':
        return 1
    elif row['grade'] == 'B':
        return 2
    elif row['grade'] == 'C':
        return 3
    elif row['grade'] == 'D':
        return 4
    elif row['grade'] == 'E':
        return 5
    elif row['grade'] == 'F':
        return 6
    else:
        return 7


df['grade_categorical'] = df.apply(lambda x: grade_categories(x),
                                   axis=1)

# Borrower location
table = pd.crosstab(df['addr_state'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

# Loan purpose
table = pd.crosstab(df['purpose'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

# Keep only rows when home_ownership is in 'MORTGAGE', 'OWN', 'RENT'
table = pd.crosstab(df['home_ownership'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

df = df[(df['home_ownership'] == 'MORTGAGE') |
        (df['home_ownership'] == 'OWN') |
        (df['home_ownership'] == 'RENT')]

# Create categories from the loan purpose on the application
table = pd.crosstab(df['purpose'], df['bad_loan'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


def purpose_categories(row):
    if row['purpose'] == 'car' or \
            row['purpose'] == 'credit_card':
        return 1
    elif row['purpose'] == 'medical' or \
            row['purpose'] == 'other' or \
            row['purpose'] == 'home_improvement' or \
            row['purpose'] == 'renewable_energy' or \
            row['purpose'] == 'vacation' or \
            row['purpose'] == 'debt_consolidation' or \
            row['purpose'] == 'wedding' or \
            row['purpose'] == 'major_purchase':
        return 2
    else:
        return 3


df['purpose_categories'] = df.apply(lambda x: purpose_categories(x), axis=1)

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




# Create an MLEAP bundle
model = RandomForestClassifier(**rf_params)

# Assemble features in a vector
features_list = list(x_train.columns)

feature_assembler = FeatureExtractor(input_scalars=features_list,
                                     output_vector='input_features',
                                     output_vector_items=['f_' + x for x in features_list])

# Assemble a pipeline with features and initialize
model.mlinit(input_features='input_features',
             prediction_column='prediction_python',
             feature_names=['f_' + x for x in features_list])

model_pipeline = Pipeline([
    (feature_assembler.name, feature_assembler),
    (model.name, model)])

model_pipeline.mlinit()

# Train the pipeline
model_pipeline.fit(x_train, y_train)