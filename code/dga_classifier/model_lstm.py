
import pickle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential


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


def build_model(max_features, maxlen):
    """
    From Endgame: Build LSTM model

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


# =================================================================================== #
# Run the model

max_epoch = 25
nfolds = 10
batch_size = 128
data = read_data(analysis_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/dga-combined-data.pkl')
target = 'dga_domain'
target_dga = 'dga'
domains = 'domain'


# ========

# Convert labels to 0-1
labels = [x for x in data[target]]
y = np.array([1 if x == target_dga else 0 for x in labels])

# Generate a dictionary of valid characters
X = [x for x in data[domains]]
valid_chars = {xi:idx+1 for idx, xi in enumerate(set(''.join(X)))}
max_features = len(valid_chars) + 1
maxlen = np.max([len(xi) for xi in X])

# Convert characters to int and pad
# (VT) Instead of this conversion, I could try ASCII representation
X = sequence.pad_sequences([[valid_chars[y] for y in xi] for xi in X], maxlen=maxlen)

# Print stats for X and y
print('The sizes of y and X are {} and {}, respectively'.format(y.shape, X.shape))



final_data = []
for fold in range(nfolds):
    print("fold %u/%u" % (fold+1, nfolds))
    X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, test_size=0.2)

    print('Build model...')
    model: Sequential = build_model(max_features, maxlen)

    print("Train...")
    X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.5)
    best_iter = -1
    best_auc = 0.0
    out_data = {}

    for ep in range(max_epoch):
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1)

        t_probs = model.predict_proba(X_holdout)
        t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

        print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))

        if t_auc > best_auc:
            best_auc = t_auc
            best_iter = ep

            probs = model.predict_proba(X_test)

            out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                        'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

            print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
        else:
            # No longer improving...break and calc statistics
            if (ep-best_iter) > 2:
                break

    final_data.append(out_data)



# ============================================================================================================= #

run_model(
    read_data(analysis_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/analytical_sample.pkl'),
    max_epoch=25,
    nfolds=10,
    batch_size=128
)


