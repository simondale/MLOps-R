# Serving the Model

These files implement a Python server for Azure ML Service that will host the model as an 
API endpoint.

This dictory includes the following files:
* requirements.txt
* server.py

## requirements.txt
The Azure ML pipeline creates an environment based upon the dependencies in this file and 
ensures they are loaded into the base Docker container.

The pipeline uses the `mcr.microsoft.com/mlops/python:latest` container which includes both
Python and R runtimes.

## server.py

The server implements two methods:
* init
* run

These are called by the Azure ML Service host and are detailed below:

### init

The `init()` method is called when the container starts and is responsible for loading the 
model and any other setup required. Since this is an R model, the packages that is requires
are also loaded using the commands:

```
R -q -e 'install.packages("caret", quiet=TRUE, repo="http://cran.us.r-project.org")
R -q -e 'install.packages("e1071", quiet=TRUE, repo="http://cran.us.r-project.org")
```

These packages are detailed in the model [README.md](../model/README.md)

### run

The `run()` method is called when requests are made to the service.

For this model, the `@rawhttp` decorator has been used so that all requests present the full
`AMLRequest` object and return an `AMLResponse` object.

This adds the overhead of handling message dispatching, but does mean the response headers can
be controlled and allows the `Access-Contol-Allow-Origin` header to be set to control access
to the service.

This method will use the R model loaded in the `init()` method to score the data being presented.
So that this is correctly marshalled, the data is loaded using the Pandas DataFrame and the Python
representation of the Iris Dataset. The columns are then mapped to match the R representation and 
to match the model's expectations.
