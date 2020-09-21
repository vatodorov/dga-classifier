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

# Import the scoring algorithm
from dga_classifier.scoring_app import DGAScorer

app = Flask(__name__)
auth = HTTPBasicAuth()


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


@app.route('/', methods=['GET'])
def get_web():
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains'

@app.route('/api/predict', methods=['POST'])
@auth.login_required
def app_score():
    if auth.current_user():
        if request.method == 'POST':
            # Parse the payload provided by the user
            # This allows us to pass
            data = request.get_json()
            domains = data.get('domain', '')
            if domains:
                domains = domains.split(',')

            # Score the domain
            # !! IMPORTANT !!:
            #   In order to avoid a delay in the scoring of the first domain need to make one API call
            scored_domains = dga_scorer.score_domains(
                data_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/results',
                analysis_date='2020-08-30',
                model_name='model_fold0',
                data=domains,
                cutoff=.5
            )

            return jsonify({'data': scored_domains}), 201, {'Content-Type': 'application/json'}
        else:
            return jsonify({'data': 'You are not authorized to use this endpoint'}), 401, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    # Initialize the scoring model
    dga_scorer = DGAScorer()
    app.run(debug=True, threaded=True)