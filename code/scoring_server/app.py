########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Analyze the model results
#
########################################################################

from flask import Flask, jsonify, request
import yaml
from .scoring_app import DGAScorer


app = Flask(__name__)

# Get the vars from config file
with open('/opt/dga-classifier/config.yaml') as conf:
    confvars = yaml.load(conf, Loader=yaml.FullLoader)

MODEL_LOCATION = confvars.get('model_location')
ANALYSIS_DATE = confvars.get('analysis_date')
MODEL_NAME = confvars.get('model_name')
app_port = confvars.get('app_port')
app_host = confvars.get('app_host')
app_threaded = False

# Initialize the scoring model
DGAScorer = DGAScorer()

# Define routes
@app.route('/', methods=['GET'])
@app.route('/api/domains/score', methods=['POST'])
def predict(cutoff=0.5):
    """
    Calls the domains predictor

    :param cutoff float:
    :return:
    """

    # This is a landing page for now
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains.'

    if request.method == 'POST':

        # Parse the payload provided by the user
        data = request.get_json()
        domain_lst = list(data.get('domain'))

        # We can also pass a list of domains, but for now just passing a single domain
        # In the future, need to be able to handle a list
        domains_list = False
        if len(domain_lst) > 1:
            domains_list = True

        # Score the domain
        domain_score = DGAScorer.score_domains(
            data_loc=MODEL_LOCATION,
            analysis_date=ANALYSIS_DATE,
            model_name=MODEL_NAME,
            list_of_domains=domains_list,
            data=domain_lst
        )

        # Map the calculated score to a cutoff and create a category
        map_scores = 'Likely a DGA' if domain_score >= cutoff else 'Unlikely a DGA'

        # Currently the return handles only single domains
        #   In the future, it may need to be able to handle a list of domains
        #   In that case, I'll need to build a list of JSONs for the value of the 'data' key
        return jsonify(
            {
                'data': [
                    {
                        'domain': ','.join(domain_lst),
                        'dga_likelihood': domain_score,
                        'dga_likelihood_cat': map_scores
                    }
                ]
            }
        ), 200

if __name__ == 'main':
    app.run(host=app_host, port=app_port, debug=True, threaded=app_threaded)
