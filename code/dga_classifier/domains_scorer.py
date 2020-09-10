########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import pandas as pd


def sanitize_data(data):
    """
    Prepare the domains for scoring

    :param data str:
    :param maxlen int:
    :return:
    """

    # Lowercase letters
    # Remove empty spaces
    data = [x.replace(' ', '').lower() for x in data]

    # Remove domains with invalid characters
    clean_data = []
    invalid_chars = "~`!@#$%^&*()_+={}[]|:;<>,?'/\\"
    for x in data:
        if any(n in x for n in [x for x in invalid_chars]):
            print('The domain {} is not valid and will not be scored'.format(x))
            continue
        clean_data.append(x)

    return clean_data


def convert_data(data, maxlen=100):
    """
    Convert the characters to integers for model scoring

    :param data:
    :param maxlen:
    :return:
    """

    valid_chars = {
        '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
        'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19,
        'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28,
        's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, '.': 37,
        '-': 38
    }

    data = sequence.pad_sequences([[valid_chars[y] for y in xi] for xi in data], maxlen=maxlen)

    return data


def model(data_loc, analysis_date, model_name):
    """
    Loads the estimated model

    :param data_loc:
    :param analysis_date:
    :param model_name:
    :return:
    """

    return load_model('{}/{}/{}.h5'.format(data_loc, analysis_date, model_name), compile=False)


def score_domains(data_loc, analysis_date, model_name, data):
    """
    Scores the domains, and returns the probability that a domain is a DGA

    :param data_loc:
    :param analysis_date:
    :param model_name:
    :param data:
    :return:
    """

    # TODO: Need to use Keras multithreading in production:
    #   https://blog.victormeunier.com/posts/keras_multithread/
    #   https://medium.com/swlh/deep-learning-in-production-a-flask-approach-7a8e839c25b9

    # Load the model
    oracle = model(data_loc, analysis_date, model_name)

    # Only score the domains that have been sanitized
    sanitized_data = sanitize_data(data)
    data = convert_data(sanitized_data)
    scored_domains = oracle.predict(data)

    # Combine the original domains with the scores
    scored_domains_pd = pd.DataFrame(
        {
            'sanitized_domains': sanitized_data,
            'predicted_probs': scored_domains.tolist()
        }
    )

    return scored_domains, scored_domains_pd


# ============================================================================ #


data = ['asdfads?.com', "asfdasdf'.com", 'asdfds\.com' , 'cnn.com',
        'bLAh.com', 'asdf asdf asdf.com', '$%#sdaf.com',
        'IUQERFSODP9IFJAPOSDFJHGOSURIJFAEWRWERGWEA.COM', 'firstenergyresourcegroup.com']

scored, scored_domains_pd = score_domains(
    data_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/results',
    analysis_date='2020-08-30',
    model_name='model_fold0',
    data=data
)

print(scored_domains_pd)

