from azureml.core.model import Model
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from rpy2.robjects import r, pandas2ri
from rpy2.rinterface_lib import openrlib

import pandas as pd
import subprocess
import os


def init():
    global model

    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split("/")[-2]
    )

    if not os.path.exists("packages_installed"):
        subprocess.check_call(["R", "-q", "-e", 'install.packages("caret", quiet=TRUE, repos="http://cran.us.r-project.org")'])
        subprocess.check_call(["R", "-q", "-e", 'install.packages("e1071", quiet=TRUE, repos="http://cran.us.r-project.org")'])    
        with open("packages_installed", "w") as f:
            f.writelines([1])

    with openrlib.rlock:
        pandas2ri.activate()
        model = r.readRDS(model_path)


@rawhttp
def run(request: AMLRequest) -> AMLResponse:
    if request.method == "GET":
        return AMLResponse("ok", 200)
    if request.method == "POST":        
        body = request.get_data(False)
        pdf = pd.read_json(body, orient="split")

        python_columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
        r_columns = [
            "Sepal.Length",
            "Sepal.Width",
            "Petal.Length",
            "Petal.Width",
        ]
        pdf = pdf.loc[:, python_columns]
        pdf.columns = r_columns

        with openrlib.rlock:
            rdf = pandas2ri.py2rpy(pdf)            
            predictions = r.predict(model, rdf, method="class")

        pdf.columns = python_columns
        pdf["predicted_species"] = predictions.to_numpy()

        response = AMLResponse(pdf.to_json(orient="split"), 200)
        response.headers.add_header("Access-Contol-Allow-Origin", "*")
        return response
    else:
        return AMLResponse("bad request", 400)
