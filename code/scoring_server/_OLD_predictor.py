########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

import tensorflow as tf
import joblib

class DGAScorer(object):

    def __init__(self, data_loc, analysis_date, model_name):
        self.model = joblib.load('{}/{}/{}.h5'.format(data_loc, analysis_date, model_name))
        self.tokenizer = joblib.load(<Path-to-tokenizer>)
        self.model._make_predict_function()
        self.session = tf.backend.get_session()
        self.graph = tf.get_default_graph()

    def predict(self):
        with self.session.as_default():
            with self.graph.as_default():
                t = self.tokenizer.text_to_sequences(df)
                pred = self.model.predict(t)

        return pred