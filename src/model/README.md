# Model Training Pipeline for Azure ML Service

These files implement a model training pipeline that can be executed by Azure ML Service.

This dictory includes the following files:
* requirements.txt
* train.py
* model.r

## requirements.txt
The Azure ML pipeline creates an environment based upon the dependencies in this file and 
ensures they are loaded into the base Docker container.

The pipeline uses the `mcr.microsoft.com/mlops/python:latest` container which includes both
Python and R runtimes.

## train.py
The Azure ML pipeline executes a PythonScriptStep to train the model (the RScript step is 
currently in preview and may change). The Python script uses the `rpy2` library to call 
R functions from within a Python application.

The first task for this script is to install the R packages used by the pipeline, currently
these are:
* optparser
* caret
* e1071

Once the model is trained the binary model is saved to Azure ML Service and the model version
is registered.

## model.r
The model is implemented using R and this script implements a function (`train_model(args)`)
that is used by the Python script as a `SignatureTranslatedAnonymousPackage` by `rpy2`.

The model is saved to the path specified so that it can be saved by the training script.

### optparser
This library provides command line parsing allowing the Python script to pass parameters
to the R script and to enable future compatibility with the RScript step.

### caret
This library provides a standard interface to many models and algorithms.

### e1071
This library is used by `caret` and the `rpart` method of training training a model.