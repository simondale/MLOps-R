# MLOps-R with Azure ML Service
This repository demonstrates how to use the Azure ML Service to train 
and host a model written in R, using Python to bootstrap the CI/CD 
pipeline.

## Setup
It is recommended that a virtual environment is used when running this script.

To install the requirements, run the following command:
```
pip install -r requirements.txt
```

### Environment Variables
The scripts in this repository use the following environment variables:

|Variable Name|Command Line Variable|Meaning|
|-|-|-|
|WORKSPACE_NAME|--workspce-name|The name of the Azure ML workspace
|SUBSCRIPTION_ID|--subscription|The Azure subscription containing the Azure ML Service
|RESOURCE_GROUP|--resource-group|The Azure resource group containing the Azure ML Service
|LOCATION|--location|The location in which to create any resources required by this pipeline
|COMPUTE_NAME|--compute-name|The name of the compute cluster used by Azure ML Service
|VM_SIZE|--vm-size|The size of the VMs that are provisioned
|ENVIRONMENT_NAME|--environment-name|The name of the environment utilised by the pipeline
|EXPERIMENT_NAME|--experiment-name|The name of the experiement to execute
|MODEL_NAME|--model-name|The name of the model to create

## Creating the Pipeline
The first task is to create the pipeline by following the instructions [here](src/mlops/pipeline/README.md)

## Hosting the Model
Once the model is trained it can be hosted by following the instructions [here](src/mlops/hosting/README.md)

## Next Steps
The next steps are:
* create a CI/CD pipeline that will automate 
