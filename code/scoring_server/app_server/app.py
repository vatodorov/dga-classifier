########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from flask import Flask, jsonify, request
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import yaml

# Import the scoring algorithm
from .dga_classifier.scoring_app import DGAScorer


# Initialize the Flask app
app = Flask(__name__)
auth = HTTPBasicAuth()

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


# TODO:
#   - Add a line to deny all requests that are not POST or GET

# Dictionary with username and password
users = {
    'user1': generate_password_hash('2NJs!&JrvZ3vFVHsFdA')
}


# Enable auth verification
@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username


# Serve a page when someone visits the link
@app.route('/', methods=['GET'])
def get_web():
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains'


# Define routes
@app.route('/api/domains/score', methods=['POST'])
@auth.login_required
def app_score():
    """
    Runs the prediction app

    :return:
    """

    if auth.current_user():

        # If the user if authenticated, provide teh response
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

            return jsonify({'data': scored_domains}),\
                   201,\
                   {'Content-Type': 'application/json'}
        else:
            return jsonify({'data': 'You are not authorized to use this endpoint'}),\
                   401,\
                   {'Content-Type': 'application/json'}


if __name__ == 'main':
    # Initialize the scoring model
    dga_scorer = DGAScorer()

    # Start the app
    app.run(host=app_host, port=app_port, debug=True, threaded=app_threaded)
