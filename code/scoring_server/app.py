########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from flask import Flask, jsonify, request
import yaml
from .dga_classifier.scoring_app import DGAScorer


# Initialize the Flask app
app = Flask(__name__)

# Get the vars from config file
with open('/opt/dga-classifier/config.yaml') as conf:
    confvars = yaml.load(conf, Loader=yaml.FullLoader)

MODEL_LOCATION = confvars.get('model_location')
ANALYSIS_DATE = confvars.get('analysis_date')
MODEL_NAME = confvars.get('model_name')
DGA_SCORE_CUTOFF = confvars.get('dga_score_cutoff')
app_port = confvars.get('app_port')
app_host = confvars.get('app_host')
app_threaded = False


# Define routes
@app.route('/', methods=['GET'])
@app.route('/api/domains/score', methods=['POST'])
def app_predict():
    """
    Runs the prediction app

    :return:
    """

    # This is a landing page for now
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains.'

    if request.method == 'POST':
        # Parse the payload provided by the user
        # This allows us to pass
        data = request.get_json()
        domains = data.get('domain', '')
        if domains:
            domains = domains.split(',')

        # Score the domains
        # !! IMPORTANT !!:
        #   In order to avoid a delay in the scoring of the first domain need to make one API call
        scored_domains = dga_scorer.score_domains(
            data_loc=MODEL_LOCATION,
            analysis_date=ANALYSIS_DATE,
            model_name=MODEL_NAME,
            data=domains,
            cutoff=DGA_SCORE_CUTOFF
        )

        return jsonify({'domain_score': scored_domains}), 200

if __name__ == 'main':
    # Initialize the scoring model
    dga_scorer = DGAScorer()

    # Start the app
    app.run(host=app_host, port=app_port, debug=True, threaded=app_threaded)
