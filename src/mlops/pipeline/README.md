# Azure ML Service Pipeline
This script contains the script that:
* creates a Workspace
* creates the AML compute cluster
* registers and environment based on a Docker image
* constructs and registers a Pipeline
* executes the pipeline to train a Model

The script uses the `azureml-sdk` package to interact with the Azure ML Service.

## Setup
It is recommended that a virtual environment is used when running this script.

To install the requirements, run the following command:
```
pip install -r requirements.txt
```

## Running the Script
```
python src/mlops/pipeline/training.py
```