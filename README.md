# EncQui.re
## Inference on encrypted data
In this project, I use homomorphic encryption to provide inference on encrypted health care records. This repo isn't finished yet, but here's how it works: We use the Intel nGraph HE transformer and Tensorflow to build a protobuf file of our model. This can be served used `pyclient_fhir` in the cloud. Then the `EHR` app serves a mockup of an EHR system using streamlit, which can call the `smart` Flask app. This SMART app gets the data from the FHIR server, does the encryption, communicates with the `pyclient_fhir` server, and finally prints the result.
## Setup
Build [Intel nGraph HE](https://github.com/IntelAI/he-transformer) according to the instructions in that repository.

Then train a model and create a `model.pb` protobuf file. To serve it, invoke

```
python test.py --backend=HE_SEAL --model_file=model.pb --enable_client=true --encryption_parameters=parameters.json
```
where `parameters.json` contains the appropriate cryptographic parameters.

On the client, simply run the appropriate files as web services.
