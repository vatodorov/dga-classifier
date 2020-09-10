########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Estimate an LSTM model - this is the model from Endgame
#
########################################################################

import sys
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd


# Import the modeling utilities
sys.path.insert(0, './code/dga_classifier')
import utilities as utils


def read_data(analysis_file_loc):
    """
    Reads the DGA and non-DGA files for the analysis

    :param analysis_file_loc str: Location of the file with all the domains. In a pickle format
    :return data DataFrame: Analysis file in Pandas dataframe format
    """

    print('Starting to read the analysis file from {}, and build a sample for analysis...'.format(analysis_file_loc))
    with open(analysis_file_loc, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    print('Successfully read the file {}'.format(analysis_file_loc))

    return data


def compile_model(max_features, maxlen):
    """
    From Endgame: Build LSTM model

    :param maxlen int:
    :param max_features int:
    :return model:
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


def run_model(data, model_results_path, overwrite_model_results, target, target_dga, domains, epochs, nfolds, batch_size, train_size):
    """
    Estimates and evaluates an LSTM model
    Most of the code here is from the Endgame model

    :param data DataFrame:
    :param model_results_path str:
    :param overwrite_model_results bool:
    :param target str:
    :param target_dga str:
    :param domains str:
    :param epochs int:
    :param nfolds int:
    :param batch_size int:
    :param train_size int:
    :return:
    """

    # ============================ #

    # JUST FOR TESTING - MAKE SURE THIS IS COMMENTED OUT WHEN RUNNING ON AWS
    data = data.sample(n=2000, replace=False, random_state=1234)

    # ============================ #

    # Maps characters from domains to integers
    # TODO:
    #   - Need to add foreign chars - Cirilic/Russian, Spanish, French, Chinese(?)
    #   - Try using ASCII representation instead of this mapping
    valid_chars = {
        '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
        'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19,
        'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28,
        's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, '.': 37,
        '-': 38
    }

    # Convert labels to 0-1
    labels = [x for x in data[target]]
    y = np.array([1 if x == target_dga else 0 for x in labels])

    # Generate a dictionary of valid characters
    X = [x for x in data[domains]]
    max_features = len(valid_chars) + 1
    maxlen = np.max([len(xi) for xi in X])

    # TODO: Should I change the maxlen to the max allowed lenght of a domain - max allowed is 253 chars??
    # source: https://stackoverflow.com/questions/14402407/maximum-length-of-a-domain-name-without-the-http-www-com-parts#:~:text=A%20full%20domain%20name%20is,(including%20the%20separators).%22&text=The%20full%20domain%20name%20may%20not%20exceed%20a%20total%20length,its%20external%20dotted%2Dlabel%20specification.

    # Convert characters to int and pad
    X = sequence.pad_sequences([[valid_chars[y] for y in xi] for xi in X], maxlen=maxlen)

    # Print stats for X and y
    print('The sizes of y and X are {} and {}, respectively\n'.format(y.shape, X.shape))

    print('Compiling the model...')
    model = compile_model(max_features, maxlen)

    # Estimate model
    final_data = {}

    for fold in range(nfolds):
        print('Running estimate for fold {}/{}'.format(fold+1, nfolds))

        # Create data samples
        # Every time we loop through here, a new set of records is selected.
        #   That way we can estimate the model on different regions of the data.
        #   This validates the performance of the model on the different sets
        print('Create analysis data samples')
        X_train, X_test_holdout, y_train, y_test_holdout, _, labels_test_holdout = train_test_split(X, y, labels, test_size=(1 - train_size))
        X_test, X_holdout, y_test, y_holdout = train_test_split(X_test_holdout, y_test_holdout, test_size=0.5)

        print('The modeling samples are: \n'
              '  -> Training: {} \n'
              '  -> Testing: {} \n'
              '  -> Holdout: {} \n'.format(len(X_train), len(X_test), len(X_holdout)))

        print('Start training...')

        # I am being conservative here and do only a single epoch - so as not to overfit
        #   At the same time, the estimate is done
        #   We could probably do 5 epochs, and it should be ok
        model_estimate = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test)
        )

        model.summary()

        # Score the test and holdout samples
        probs_holdout = model.predict(X_holdout)
        probs_test = model.predict(X_test)
        probs_test_holdout = model.predict(X_test_holdout)

        # Print ROC AUC
        auc_holdout = sklearn.metrics.roc_auc_score(y_holdout, probs_holdout)
        auc_test = sklearn.metrics.roc_auc_score(y_test, probs_test)
        auc_test_holdout = sklearn.metrics.roc_auc_score(y_test_holdout, probs_test_holdout)
        print('ROC AUC Score: \n'
              ' -> Holdout sample: {} \n'
              ' -> Test sample: {} \n'
              ' -> Combined holdout & test: {} \n'.format(auc_holdout, auc_test, auc_test_holdout))
        print('=================================================================')

        # Calculate confusion matrix
        #   For now, use a cutoff of 0.5
        #   The cutoff score should be fine tuned, and more precisely calcuated, but for a rough analysis use 0.5
        cm_test_holdout = sklearn.metrics.confusion_matrix(y_test_holdout, probs_test_holdout > .5)

        # Collect evaluation results and save them
        # Output the predictions on the testing and validation data
        final_data['fold{}_data'.format(fold)] = {
            'y_test_holdout': y_test_holdout,
            'labels_test_holdout': labels_test_holdout,
            'probs_test_holdout': probs_test_holdout,
            'cm_test_holdout': cm_test_holdout,
            'model_history': model_estimate.history
        }

        # Overfit test - Calculate KS and save the KS stats plot for each fold
        y_train_ks = pd.DataFrame({
            'target': y_train,
            'probability': [item for sub in model.predict(X_train).tolist() for item in sub]
        })
        y_test_ks = pd.DataFrame({
            'target': y_test,
            'probability': [item for sub in model.predict(X_test).tolist() for item in sub]
        })

        if overwrite_model_results:
            # Save the model estimate
            model.save('{}/model_fold{}.h5'.format(model_results_path, fold))

            # Save the model KS stat
            utils.compare_train_test(X_train, y_train_ks, X_test, y_test_ks, model_results_path, fold, bins=30)

    # Store the results as a pickle
    if overwrite_model_results:
        f = open('{}/analysis_results.pkl'.format(model_results_path), 'wb')
        pickle.dump(final_data, f)
        f.close()

    return final_data


# ============================================================================================================= #


model_results = run_model(
    data=read_data(analysis_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/analytical_sample.pkl'), # '/root/dga-classifier/data/analytical_sample.pkl'),
    model_results_path='/Users/valentint/Documents/GitRepos/dga-classifier/data/results/', # '/root/dga-classifier/data/results/2020-08-30',
    overwrite_model_results=True,
    epochs=3,
    nfolds=3,
    batch_size=128,
    train_size=0.7,
    target='dga_domain',
    target_dga='dga',
    domains='domain'
)

