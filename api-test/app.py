#!/Users/valentint/Virtualenv/py36/bin/python
from flask import Flask
from flask import request, jsonify, abort
from random import randint

app = Flask(__name__)


@app.route('/', methods=['GET'])
@app.route('/api/predict', methods=['POST'])
def create_task(cutoff=50):
    if request.method == 'GET':
        return 'Welcome to Manticore. We offer an automated classification of domains'

    if request.method == 'POST':
        data = request.get_json()
        extracted_domain = data.get('domain')

        domain_score = randint(1, 100)
        map_scores = 'Very Likely' if domain_score >= cutoff else 'Unlikely'

        return jsonify(
            {
                'data': [
                    {
                        'domain': extracted_domain,
                        'dga_likelihood': domain_score,
                        'dga_likelihood_cat': map_scores
                    }
                ]
            }
        ), 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)