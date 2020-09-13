#!/Users/valentint/Virtualenv/py36/bin/python
from flask import Flask
from flask import request, jsonify, abort
from scoring_app import DGAScorer

app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/api/predict', methods=['POST'])
def create_task():

    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains'

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

        return jsonify({'domain_score': scored_domains}), 200

if __name__ == '__main__':
    # Initialize the scoring model
    dga_scorer = DGAScorer()
    app.run(debug=True, threaded=True)