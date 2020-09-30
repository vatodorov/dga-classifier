#!/Users/valentint/Virtualenv/py36/bin/python
########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import yaml

# Import the scoring algorithm
from .dga_classifier.scoring_app import DGAScorer


app = Flask(__name__)
auth = HTTPBasicAuth()

# Get the vars from config file
with open('/opt/mc_dga_classifier/config.yaml', 'r') as conf:
    try:
        confvars = yaml.safe_load(conf)['data']
    except yaml.YAMLError as err:
        print(err)

model_location = confvars.get('model_location')
analysis_date = confvars.get('analysis_date')
model_name = confvars.get('model_name')
dga_score_cutoff = confvars.get('dga_score_cutoff')
app_port = confvars.get('app_port')
app_host = confvars.get('app_host')
app_threaded = False if 'single' in confvars.get('app_threads_mode').lower() else True

# Dictionary with username and password
users = {
    'user1': generate_password_hash('2NJs!&JrvZ3vFVHsFdA')
}


# Initialize the scoring model
dga_scorer = DGAScorer()

# Enable auth verification
@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/', methods=['GET'])
def get_web():
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains'

@app.route('/api/predict', methods=['POST'])
@auth.login_required
def app_score(source='api'):
    if auth.current_user():
        if request.method == 'POST':
            # Parse the payload provided by the user
            # This allows us to pass
            data = request.get_json()
            domains = data.get('domain', '')
            if domains:
                domains = domains.split(',')

            # Score the domain
            scored_domains = dga_scorer.score_domains(
                data_loc=model_location,
                analysis_date=analysis_date,
                model_name=model_name,
                data=domains,
                cutoff=dga_score_cutoff,
                source=source
            )

            return jsonify({'data': scored_domains}),\
                   201,\
                   {'Content-Type': 'application/json'}
    else:
        return jsonify({'data': 'You are not authorized to use this endpoint'}),\
               401,\
               {'Content-Type': 'application/json'}

if __name__ == '__main__':
    # Start the app
    app.run(host=app_host, port=app_port, debug=True, threaded=app_threaded)
