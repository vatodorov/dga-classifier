########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Modeling utilities
#
########################################################################

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import six

# Imports various modeling tools
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, f1_score, classification_report, \
    accuracy_score, precision_score, recall_score, brier_score_loss

"""
This is a collection of utility functions that are useful for features engineering, development, and validation of predictive models.
Also contains various graphing utilities.

Features engineering:
    set_min_value : Sets the minimum value for a feature
    cap_outliers : Caps outliers to a user-defined percentile
    createLog : Creates new features that are the log of existing ones
    createSqrt : Creates new features that are the square root of existing ones
    createSquare : Creates new features that are the square of existing ones
    createInverse : Creates new features that are the inverse of existing ones
    createDummy : Creates dummy features that take the value 1 if the value of an input feature is larger than 0, otherwise take the value of 0

Model development:
    model_gridsearch_cv : Performs cross validation for Sklearn models - usually used for Random Forest and xgb.XGBClassifier
    downsample_data : Downsamples data for analysis
    correlation_pearson : Calculates Pearson correlation between two series

Model validation:
    evaluate_xgb_accuracy : Quick accuracy evaluation of the core (not Sklearn API) XGB models
    divergence : Calculates the divergence metric between the predicted probabilities for the positive and negative timelines
    brier_gain : Calculates the Brier Gain for each model. Brier Gain is the opposite of Brier Loss (1 - Brier Loss)
    ksTest : Kolmogorov-Smirnov test to determine the optimal probability cutoff in a binary classification model
    modelAccuracyStats : Prints accuracy statistics for each model. A nifty tool for quicking checking overfit between test and train
    liftTable : Creates a lift table for model validation
    confusionMatrix : Calculates the confusion matrix for an estimated model. It returns TN, FP, FN, TP in a list
    confusion_matrix_calculations : Creates the expanded confusion matrix we use in presentations and assessment of model performance on validation set
    plot_roc : Creates a plot of Receiver Operating Curves for each estimated model
    compare_train_test : Creates test/train overfiting plots, and calculates KS for a classifier (this method was modified from the git repo with the CV2 analysis)

Graphing utilities:
    render_mpl_table : Creates a pretty table from the Data frame lift table
    plot_multiple_graphs : (# TODO: Add X and Y axes labels) Creates subplots of variables. Allows users to chose between 'line', 'scatter' and 'barplot' charts
    plot_histogram : Plots histograms
    plot_scatter : Creates scatter plots

Miscellaneous:
    check_feature_type : Infers the type of variables or provide a list for categorical, continous, and binary
    calculate_ci : Calculates confidence intervals
    randomForest_save_trees : Saves the tree structures from an estimated Random Forest model
    checkpoint : Prints a checkpoint in the execution of a big script block
    num_events : Given a dataframe, finds the number events per index (id and order)
"""

# Collection of parameters that control plot formats - KEEP ON TOP
axis_font = {'fontname': 'DejaVu Sans', 'size': '14'}
title_font = {'fontname': 'DejaVu Sans', 'size': '14'}
xticks_font = {'fontname': 'DejaVu Sans', 'size': '12'}
yticks_font = {'fontname': 'DejaVu Sans', 'size': '12'}
legend_font = {'fontsize': '10'}


def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=7,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Creates a pretty table from the Data frame lift table
    Source: https://stackoverflow.com/questions/26678467/export-a-pandas-dataframe-as-a-table-image/39358722

    :param data:
    :param col_width:
    :param row_height:
    :param font_size:
    :param header_color:
    :param row_colors:
    :param edge_color:
    :param bbox:
    :param header_columns:
    :param ax:
    :param kwargs:
    :return:
    """

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    return ax

def plot_roc(models_labels, models_fpr, models_tpr, models_auc):
    """
    Creates a plot of Receiver Operating Curves for each estimated model.

    Parameters
    -------------
       models_labels - Name of models to show in the graph
       models_fpr - A list of False Positive rates calculated for each model. The calculated FPR is an array
       models_tpr - A list of True Positive rates calculated for each model. The calculated TPR is an array
       models_roc - A list with the AUC values for each model

    Returns
    -------------
       A plot of the Receiver Operating Curves for each model.
    """

    plt.figure(figsize=(12, 7))

    for i in range(0, len(models_labels)):
        plt.plot(models_fpr[i], models_tpr[i], label=models_labels[i] + ' (AUC = %0.2f)' % models_auc[i])
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive', **axis_font)
    plt.ylabel('True Positive', **axis_font)
    plt.title('ROC curve', **title_font)
    plt.xticks(**xticks_font)
    plt.yticks(**yticks_font)
    plt.legend(loc='lower right', **legend_font)
    plt.show()

    return


def brier_gain(models, y_true, y_prediction):
    """
    Calculates the Brier Gain for each model. Brier Gain is the opposite of Brier Loss (1 - Brier Loss).

    Parameters
    -------------
        models - A list of fitted models. If value is None, the function expects a vector of predicted values
        y_true - A vector of true outcome. In most cases this is, the 'label' variable
        y_prediction - A list of predicted values, or a data frame/array from which predictions can be calculated

    Returns
    -------------
        A vector of Brier Gain values for each model, or score provided in the input.
    """

    brier_gain_models = []

    # Loop through all the models - Calculate predictions on the fly, calculate Brier Loss and Brier Gain
    for i in range(0, len(models)):
        if models[i] == None:
            y_predicted = y_prediction[i]
        else:
            y_predicted = models[i].predict_proba(y_prediction[i])[:, 1]
        brier_gain = round((1 - brier_score_loss(y_true[i], y_predicted)), 2)
        brier_gain_models.append(brier_gain)

    return brier_gain_models


def correlation_pearson(x_series, y_series):
    """
    Calculates Pearson correlation between two series

    Parameters
    -------------
        x_series - First of two series to calculate correlation
        y_series - Second of two series to calculate correlation

    Returns
    -------------
        Correlation coefficient between both series.
    """

    pearson_correlation = sp.stats.pearsonr(x_series, y_series)[0]

    return pearson_correlation


def plot_histogram(data_series, xlabel, ylabel, title, lower_bound=0,
                   step=0.05, figsize=(8, 4), edgecolor='black', capstyle='round', align='mid'):
    """
    Function for creating histogram plots.

    Returns
    -------------
        A histogram plot.
    """

    lower_bound = lower_bound
    upper_bound = max(data_series)
    step = step
    x_bins = np.arange(lower_bound, upper_bound + step, step)

    plt.figure(figsize=figsize)
    plt.hist(x=data_series, bins=x_bins, align=align,
             edgecolor=edgecolor,
             capstyle=capstyle)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.title(title, **title_font)
    plt.xticks(np.arange(min(data_series), upper_bound + step, step))
    plt.xticks(**xticks_font)
    plt.yticks(**yticks_font)
    plt.show()

    return


def plot_scatter(x_series, y_series, xlabel, ylabel, title, figsize=(10, 6)):
    """
    Function to create scatter plots.

    Returns
    -------------
        A scatter plot.
    """

    plt.figure(figsize=figsize)
    plt.scatter(x_series, y_series, marker='.')
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    plt.title(title, **title_font)
    plt.show()

    return


def checkpoint(message):
    print(message)


def calculate_ci(variable, confidence_level=0.975):
    """
    Function to calculate confidence intervals.

    Parameters
    -------------
        variable - a variable for which the CI to be calculated
        confidence_level - This is used to calculate the z score for the CI

    Returns
    -------------
        variable_mean - the mean of a variable
        lower_bound_ci - lower bound of confidence interval
        upper_bound_ci - upper bound of confidence interval
    """

    z_score = sp.stats.norm.ppf(confidence_level)
    variable_std = np.std(variable)
    variable_mean = np.mean(variable)

    lower_bound_ci = variable_mean - ((z_score * variable_std) / np.sqrt(len(variable)))
    upper_bound_ci = variable_mean + ((z_score * variable_std) / np.sqrt(len(variable)))

    return variable_mean, lower_bound_ci, upper_bound_ci


def confusionMatrix(model, x_test, y_true):
    """
    Calculates the confusion matrix for a model.
    It accepts either (1) an estimated model with a test file and the true outcome, or (2) a pre-computed score, and the true outcome.

    Parameters
    -------------
        model : If a model is available, a trained predictive model object, otherwise empty
        _x_validate : Validation set as a Pandas dataframe
        _y_validate : Validation set with the actual label as a Pandas series

    Returns
    -------------
        The confusin matrix as an array with four elements which are
            the model's True Negative, False Positive, False Negative, True Positive.

    """

    if model == None:
        _confusion_matrix = confusion_matrix(y_true, x_test).ravel()
    else:
        _confusion_matrix = confusion_matrix(y_true, model.predict(x_test)).ravel()

    return _confusion_matrix


def ksTest(model_estimated_probs, y_test_actual_outcome):
    """
    A two sample Kolmogorov-Smirnov (KS) test to determine the optimal probability cutoff
        in a binary classification model.
    For more on the test see this https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html

    Parameters
    -------------
        model_estimated_probs : A vector with the predicted probabilities from the test dataset
        y_test_actual_outcome : A vector with the actual outcomes from the test dataset

    Returns
    -------------
        probability_cutoff : The probability threshold. Above that threshold, the predicted outcome is 1, below it's 0
        p_value_kstest : The p-value of the KS test

    """

    probability_cutoff = stats.ks_2samp(model_estimated_probs, y_test_actual_outcome)[0]
    p_value_kstest = stats.ks_2samp(model_estimated_probs, y_test_actual_outcome)[1]

    return probability_cutoff, p_value_kstest


def modelAccuracyStats(model, modeling_method, _x_train, _y_train, _x_validate, _y_validate):
    """
    Function to print accuracy statistics for each model.

    Parameters
    -------------
        model : A trained predictive model object
        modeling_method : Characted string with the name of the modeling method (e.g. 'Random Forest')
        _x_train : Training set as a Pandas dataframe
        _y_train : Training set with the actual label as a Pandas series
        _x_validate : Validation set as a Pandas dataframe
        _y_validate : Validation set with the actual label as a Pandas series

    Returns
    -------------
        Prints the False Negative and False Positive for the validation dataset.
        Prints the Accuracy and Recall of the training and validation sets.
    """

    tn, fp, fn, tp = confusionMatrix(model, _x_validate, _y_validate)
    print('Print stats for method: %s' % modeling_method)
    print('\nThe number of False Negative in the validation set is %s ' % str(fn))
    print('The number of False Positive in the validation set is %s ' % str(fp))
    print('\nAccuracy on training set is {:.5f}'.format(model.score(_x_train, _y_train)))
    print('Recall (tp/(tp+fn)) of training set is {:.5f}'.format(recall_score(_y_train, model.predict(_x_train))))
    print('\nAccuracy on validation set is {:.5f}'.format(model.score(_x_validate, _y_validate)))
    print(
        'Recall (tp/(tp+fn)) of validation set is {:.5f}'.format(recall_score(_y_validate, model.predict(_x_validate))))


def liftTable(df, target, score, number_bins=10):
    """
    Creates a lift table for the evaluation of a predictive model.
    For more detail of the purpose of a gains table see this page:
        https://www.listendata.com/2014/08/excel-template-gain-and-lift-charts.html

    Parameters
    -------------
        df : A dataframe that contains the true label. This is the test or validation datasets
        target : The name of the target variable as a string
        score : The name of the variable that contains the predicted probability
        number_bins : The number of bins to group the observations in. The default is 10

    Returns
    -------------
    Pandas dataframe : A dataframe with the following columns:
        decile : A decile is group of 10% of the records in the scored file.
                 The top decile contains the observations with the highest probability for the outcome
        min_score : The minimum predicted probability for the observations in the decile
        max_score : The maximum predicted probability for the observations in the decile
        sum_positive : Count of True Positive cases.
                       The repurchase or not outcome in each decile is observed with hindsight
        sum_negative : Count of True Positive non-cases
        total_records : Total count of observations in a decile
        cummulative_positive : Cumulative count of cases
        pct_positive_of_total : Percent cases by decile of total cases in the dataset
        pct_positive : Percent cases of the total in a decile
        gain : Cumulative percent total
        lift : Measures how much better models can identify the cases vs. a random pick
        ks_score : The predicted probability threshold to separate positive from negative

    Examples
    -------------
        # Estimate a Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        model_rf = RandomForestClassifier(n_estimators = 10,
                                          max_depth = 5,
                                          n_jobs = -1,
                                          random_state = 7894)
        model_rf.fit(x_train, y_train)

        # Predict probabilities for the testing dataset
        probs = model_rf.predict_proba(x_validate)[:, 1]

        # Create a dataframe with the predicted probabilities (scored_probs) and the actual outcome (label_y_test)
        scored_data = pd.DataFrame({'label_y_test': y_validate.values, 'scored_probs': probs})

        # Create a lift table
        ltable = liftTable(scored_data, 'label_y_test', 'scored_probs', 10)
    """

    target = target
    score = score

    conversion_rate_validation = float(len(df[df[target] == 1])) / len(df)

    # Group the data into n equal sized groups
    # The grouping is done by the predicted probability
    df['negative'] = 1 - df[target]

    df.sort_values(score, ascending=False, inplace=True)
    df['idx'] = range(1, len(df) + 1)
    df['bins'] = pd.cut(df['idx'], bins=number_bins, right=True, retbins=False, precision=3)

    # Obtain summary information for each group
    aggregated = df.groupby('bins')
    aggregated = df.groupby('bins', as_index=False)

    lift_table = pd.DataFrame(np.vstack((aggregated.min()[score].map('{:,.4f}'.format))), columns=['min_score'])
    lift_table.sort_values('min_score', ascending=False, inplace=True)
    lift_table['max_score'] = aggregated.max()[score].map('{:,.4f}'.format)
    lift_table['sum_positive'] = aggregated.sum()[target]
    lift_table['sum_negative'] = aggregated.sum().negative
    lift_table['total_records'] = lift_table.sum_positive + lift_table.sum_negative

    # Cumulative positive
    lift_table['cumulative_positive'] = lift_table.cumsum().sum_positive
    lift_table['pct_positive_of_total'] = lift_table.sum_positive / lift_table.sum().sum_positive

    # Calculate odds ratio, positive rate and KS stats
    lift_table['pct_positive'] = lift_table.sum_positive / lift_table.total_records

    # Calculate gains and lift
    deciles_count = len(df.bins.unique())
    lift_table['decile'] = np.arange(1, (deciles_count + 1))
    lift_table['gain'] = (lift_table.cumsum().pct_positive_of_total * 100).round(2)
    lift_table['cumulative_lift'] = lift_table.gain / (lift_table.decile * (100 / number_bins))
    lift_table['baseline_lift'] = lift_table.pct_positive / conversion_rate_validation

    # Calculate optimal KS score
    positive_cum = (lift_table.sum_positive / df[target].sum()).cumsum()
    negative_cum = (lift_table.sum_negative / df.negative.sum()).cumsum()
    lift_table['ks_score'] = np.round((positive_cum - negative_cum), 4) * 100

    lift_table.drop('decile', axis=1, inplace=True)

    return lift_table


def confusion_matrix_calculations(df):
    """
    Creates the full confusion matrix we use in presentations and assessment of model performance on validation set
    It expects a dataframe with already calculated TN, FP, FN, TP
    """

    total_records_test = sum(df.iloc[0])

    df['accuracy'] = (df.true_negative + df.true_positive) / total_records_test
    df['precision'] = df.true_positive / (df.true_positive + df.false_positive)
    df['recall'] = df.true_positive / (df.true_positive + df.false_negative)
    df['1-FPR'] = 1 - (df.false_positive / total_records_test)
    df['1-FNR'] = 1 - (df.false_negative / total_records_test)
    df['f1_score'] = 2 * ((df.precision * df.recall) / (df.precision + df.recall))
    df['total_records_test'] = total_records_test
    df['percent_positive'] = (df.false_negative + df.true_positive) / total_records_test

    return df


def compare_train_test(x_train, y_train, x_test, y_test, save_loc, fold_num, bins):
    """
    Creates test/train overfitting plots for classifier output

    Parameters
    -------------
        X_train - Dataframe used for training with the predictors
        y_train - Dataframe with the target for training
        X_test - Dataframe used for testing
        y_test - Dataframe with the predicted outcome
        bins - Number of bins for the distribution

    Returns
    -------------
        A plot of the CDFs of the predicted probability.
    """

    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize=(15, 10))

    decisions = []
    for X, y in ((x_train, y_train), (x_test, y_test)):
        d1 = y['probability'][y['target'] == 1]
        d2 = y['probability'][y['target'] == 0]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='+ (train)'
    )

    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='- (train)'
    )

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='+ (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='- (test)')

    # get the KS score
    ks = stats.ks_2samp(decisions[0], decisions[2])

    plt.xlabel('Classifier Output')
    plt.ylabel('Normalized Units')
    plt.title('KS Test Statistic Plot')

    plt.plot([], [], ' ', label='KS Statistic (p-value) :' + str(round(ks[0], 2)) + '(' + str(round(ks[1], 2)) + ')')
    plt.legend(loc='best')

    plt.savefig('{}/fold{}-overfit.pdf'.format(save_loc, fold_num))
    #plt.show()
