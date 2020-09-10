########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

# TODO: Need to use Keras multithreading in production:
#   https://medium.com/swlh/deep-learning-in-production-a-flask-approach-7a8e839c25b9
#   https://www.linode.com/docs/applications/big-data/how-to-move-machine-learning-model-to-production/


# =====

from math import expm1
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow import keras


app = Flask(__name__)
model = keras.models.load_model("assets/price_prediction_model.h5")
transformer = joblib.load("assets/data_transformer.joblib")


@app.route("/", methods=["POST"])
def index():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(transformer.transform(df))
    predicted_price = expm1(prediction.flatten()[0])
    return jsonify({"price": str(predicted_price)})


