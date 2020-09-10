########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc, f1_score, classification_report, \
    accuracy_score, precision_score, recall_score
import sys

# Import the modeling utilities
sys.path.insert(0, './code/dga_classifier')
import utilities as utils


axis_font = {'fontname': 'DejaVu Sans', 'size': '14'}
title_font = {'fontname': 'DejaVu Sans', 'size': '16'}


def read_data(data_loc, analysis_run_date, model_results):
    """
    Reads the file with model results

    :param data_loc str: Location of the file with all the domains. In a pickle format
    :param analysis_run_date str: The date when the analysis was ran. Format is YYYY-MM-DD
    :param model_results str: Name of the results file
    :return data DataFrame: Analysis file in Pandas dataframe format
    """

    results = '{}/{}/{}'.format(data_loc, analysis_run_date, model_results)

    print('Starting to read the file with the analysis results...')
    with open(results, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    print('Successfully read the file {}'.format(results))

    return data


# Plot the validation and training loss by model
def linear_plot(data, _xlabel, _ylabel, _title, epochs, _ylim_max, fold_num, axis_font, title_font, save_loc=None):
    """
    Creates linear graphs

    :param data:
    :param _xlabel:
    :param _ylabel:
    :param _title:
    :param epochs:
    :param _ylim_max:
    :param fold_num:
    :param save_loc:
    :param axis_font:
    :param title_font:
    """

    map_names = {
        'loss': 'Training Loss',
        'val_loss': 'Validation Loss'
    }

    plt.figure(figsize=(10, 7))
    for k, v in data.items():
        plt.plot(v, label=map_names[k])

    plt.legend(loc='best')
    plt.xlim([0, epochs])
    plt.ylim([0, _ylim_max])
    plt.xlabel(_xlabel, **axis_font)
    plt.ylabel(_ylabel, **axis_font)
    plt.title(_title, **title_font)

    # Save the validation graphs in PDF
    if save_loc:
        plt.savefig('{}/{}-Train-Validation-Loss.pdf'.format(save_loc, fold_num))


def plot_roc(data, y_actual_label, y_predicted_label, save_loc, axis_font, title_font):
    """
    Plot the ROC curve for all models

    :param data:
    :param y_actual_label:
    :param y_predicted_label:
    :param save_loc:
    :param axis_font:
    :param title_font:
    """

    plt.figure(figsize=(12, 7))

    # Plot the ROC curve
    for k, v in data.items():
        y_actual = v[y_actual_label]
        y_predicted = v[y_predicted_label]

        roc = roc_auc_score(y_actual, y_predicted)
        fpr, tpr, thresholds = roc_curve(y_actual, y_predicted)
        plt.plot(fpr, tpr, label='{} (AUC = {})'.format(k.split('_')[0], round(roc, 3)))

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive', **axis_font)
    plt.ylabel('True Positive', **axis_font)
    plt.title('ROC curve', **title_font)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right')

    # Show in output
    # plt.show()

    # Save the validation graphs in PDF
    if save_loc:
        plt.savefig('{}/ROC_Curve_All_Models.pdf'.format(save_loc))


# Loop through the model history for all folds
def analyze_data(data_loc, analysis_date, analysis_results_file, y_actual_label, y_predicted_label, score_cutoff,
                 run_model_history=False, run_roc_curve=False, run_lift_table=False, run_cm_table=False):
    """
    Runs analyses of the data:
        run_model_history: Produces analysis of the train and validation data
        run_roc_curve:
        run_lift_table:
        run_cm_table:

    :param data_loc:
    :param analysis_date:
    :param analysis_results_file:
    :param y_actual_label:
    :param y_predicted_label:
    :param score_cutoff:
    :param run_model_history:
    :param run_roc_curve:
    :param run_lift_table:
    :param run_cm_table:
    """

    # Read in the data with the outputs from the models
    data = read_data(
        data_loc,
        analysis_date,
        analysis_results_file
    )

    # Placeholder to collect the confusion matrices for all models
    cm_list = []

    # Loop through the analysis data for all folds
    for k, v in data.items():
        fold_num = k.split('_')[0]

        # 1) Plot the training and validation loss history for each fold and epoch
        if run_model_history:
            dataplot = v['model_history']
            epochs = len(dataplot['loss'])

            # Plot the validation and training losses
            linear_plot(
                data=dataplot,
                _xlabel='Number of epochs',
                _ylabel='Loss',
                _title='Training and validation loss',
                epochs=epochs,
                _ylim_max=.3,
                fold_num=fold_num,
                save_loc='{}/{}'.format(data_loc, analysis_date),
                axis_font=axis_font,
                title_font=title_font
            )

        # 2) Create a lift table for each model separately
        if run_lift_table:
            scored_data = pd.DataFrame(
                {
                    'y_label': v[y_actual_label].tolist(),
                    'y_probs': [item for sub in v[y_predicted_label].tolist() for item in sub]
                }
            )

            lift_table = utils.liftTable(
                df=scored_data,
                target='y_label',
                score='y_probs',
                number_bins=10
            )

            # Output the Lift table metrics to a CSV
            lift_table.to_csv('{}/{}/{}_Lift_Table.csv'.format(data_loc, analysis_date, fold_num))

        # 3) Model evaluation metrics for all models combined
        # This creates a nice summary table for all models for all epochs
        if run_cm_table:
            cm_test_holdout = confusion_matrix(v[y_actual_label], v[y_predicted_label] > score_cutoff)
            cm_list.append(np.concatenate(cm_test_holdout).ravel().tolist())

    if cm_list:
        cm_df = pd.DataFrame(
            cm_list,
            columns=['true_positive', 'false_positive', 'false_negative', 'true_negative']
        )

        cm_table = utils.confusion_matrix_calculations(cm_df)
        cm_table.to_csv('{}/{}/Confusion_Matrix.csv'.format(data_loc, analysis_date))

    # 4) Plot the ROC curve with all models on it
    if run_roc_curve:
        plot_roc(
            data=data,
            y_actual_label=y_actual_label,
            y_predicted_label=y_predicted_label,
            save_loc='{}/{}'.format(data_loc, analysis_date),
            axis_font=axis_font,
            title_font=title_font
        )


# ============================================================================================================= #


analyze_data(
    data_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/results',
    analysis_date='2020-08-30',
    analysis_results_file='analysis_results.pkl',
    y_actual_label='y_test_holdout',
    y_predicted_label='probs_test_holdout',
    run_model_history=False,
    run_roc_curve=False,
    run_lift_table=False,
    score_cutoff=0.5,
    run_cm_table=True
)




# 4) Come up with a maximum number of epochs, and re-run the model
#   -> 25 epochs are too many, it takes about 8 hrs to run the model with 10 folds, 25 epochs and 200K in a training set

