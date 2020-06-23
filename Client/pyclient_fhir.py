import time
import argparse
import numpy as np
import sys
import os

from fhir_util import load_fhir_data, client_argument_parser
import pyhe_client


def test_network(FLAGS):
    (x_train, y_train, x_test, y_test) = load_fhir_data(
        FLAGS.start_batch, FLAGS.batch_size)
    data = x_test.flatten("C")

    client = pyhe_client.HESealClient(
        FLAGS.hostname,
        FLAGS.port,
        FLAGS.batch_size,
        {FLAGS.tensor_name: (FLAGS.encrypt_data_str, data)},
    )
    results = np.array(client.get_results()).reshape(FLAGS.batch_size,2)

    y_pred = np.argmax(results,1)
    print("y_pred", y_pred)


if __name__ == "__main__":
    FLAGS, unparsed = client_argument_parser().parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)

    test_network(FLAGS)
