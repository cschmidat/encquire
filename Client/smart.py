import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import argparse
import numpy as np
import pandas as pd
import sys
import os
import json
import requests
from flask import Flask, request, render_template

from fhir_util import load_fhir_data, load_mnist_data, client_argument_parser
import pyhe_client

hostname = ""
def xtest_from_json(jsonatt):
    testdf = pd.read_json(jsonatt, typ='series', convert_dates=False)
    testdf = testdf[["age","bmi"]]
    return testdf

def get_fhir_record(url, patid, token):
    payload = {'id': patid, 'token': token}
    with requests.session() as session:
        response = session.post(url, json=payload)
    return response.json()

def test_network(FLAGS, xtest):
    (x_train, y_train, x_test, y_test) = load_fhir_data(
        0, 1)
    x_test = xtest.to_numpy().astype("float32")
    data = x_test.flatten("C")
    print(data)
    print(FLAGS)
    client = pyhe_client.HESealClient(
        hostname,
        FLAGS.port,
        1,
        {FLAGS.tensor_name: ("encrypt", data)},
    )
    results = np.array(client.get_results()).reshape(FLAGS.batch_size,2)
    print(results)

    

    y_pred = np.argmax(results,1)
    print("y_pred", y_pred)
    return y_pred


app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_ret():
    url = request.args.get('server')
    patid = request.args.get('id')
    token = request.args.get('token')
    jsonatt = get_fhir_record(url, patid, token)
    xtest = xtest_from_json(jsonatt)
    print(xtest)
    data = test_network(FLAGS, xtest.to_frame())[0]
    outcomes = ["survive", "die"]
    outcome = outcomes[data]
    return(render_template('smart_template.html', outcome=outcome, patid=patid))
if __name__ == "__main__":
    FLAGS, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    app.run(host='0.0.0.0', port=8502, debug=True)
