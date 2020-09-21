########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from keras.preprocessing import sequence
from tensorflow.keras.models import load_model


class DGAScorer(object):
    """
    Implements the scoring of the domains
    """

    def __init__(self):
        self.valid_chars = {
            '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
            'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19,
            'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28,
            's': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, '.': 37,
            '-': 38
        }
        self.maxlen = 100

    def sanitize_data(self, data):
        """
        Prepare the domains for scoring

        :param data str:
        :return:
        """

        # Lowercase letters
        # Remove empty spaces
        data = [x.replace(' ', '').lower() for x in data]

        # Remove domains with invalid characters
        clean_data = []
        invalid_chars = "~`!@#$%^&*()_+={}[]|:;<>,?'/\\"
        for x in data:

            # !! IMPORTANT !!:
            #   For now, all the domains with invalid characters are dropped, but need to change later
            #   Need to probably create a JSON with a key that identifies which domain to score or not
            if any(n in x for n in [x for x in invalid_chars]):
                print('The domain {} is not valid and will not be scored'.format(x))
                continue
            clean_data.append(x)

        return clean_data

    def convert_data(self, data):
        """
        Convert the characters to integers for model scoring

        :param data:
        :return:
        """

        data = sequence.pad_sequences([[self.valid_chars[y] for y in xi] for xi in data], maxlen=self.maxlen)

        return data

    def model(self, data_loc, analysis_date, model_name):
        """
        Loads the estimated model

        :param data_loc:
        :param analysis_date:
        :param model_name:
        :return:
        """

        return load_model('{}/{}/{}.h5'.format(data_loc, analysis_date, model_name), compile=False)

    def category_mapper(self, score, cutoff):
        """
        Maps the DGA likelihood score to a category
        :param score:
        :return:
        """

        return ('Very Likely' if score >= cutoff else 'Unlikely')

    def score_domains(self, data_loc, analysis_date, model_name, data, cutoff):
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
        oracle = self.model(data_loc, analysis_date, model_name)

        # Only score the domains that have been sanitized
        sanitized_data = self.sanitize_data(data)
        tokenized_domains = self.convert_data(sanitized_data)
        scored_domains = [x for subl in oracle.predict(tokenized_domains).tolist() for x in subl]

        # Combine the original domains with the scores and output a JSON
        json_output = []
        for i in range(0, len(data)):
            json_output.append(
                {
                    'orig_domain': data[i],
                    'clean_domain': sanitized_data[i],
                    'dga_score': scored_domains[i],
                    'dga_cat': self.category_mapper(scored_domains[i], cutoff)
                }
            )

        return json_output
