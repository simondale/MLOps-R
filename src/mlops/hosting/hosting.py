from flask import Flask, Response, request
from flask_restful import Api
from rpy2.robjects import r, pandas2ri
from azureml.core import Workspace

import pandas as pd
import os

pandas2ri.activate()


ws = Workspace.create(
    name=os.environ.get("WORKSPACE_NAME"),
    subscription_id=os.environ.get("SUBSCRIPTION_ID"),
    resource_group=os.environ.get("RESOURCE_GROUP"),
    location=os.environ.get("LOCATION"),
    exist_ok=True,
)

model_version = ws.models.get(os.environ.get("MODEL_NAME"))
model_path = model_version.download(exist_ok=True)
model = r.readRDS(model_path)

app = Flask(__name__)
api = Api(app)


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    body = request.get_data()
    pdf = pd.read_json(body, orient="split")

    # put model code here
    python_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    r_columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
    pdf = pdf.loc[:, python_columns]
    pdf.columns = r_columns

    rdf = pandas2ri.py2rpy(pdf)
    predictions = r.predict(model, rdf, method="class")

    pdf.columns = python_columns
    pdf["predicted_species"] = predictions.to_numpy()

    response = Response(pdf.to_json(orient="split"), status=200)
    response.headers.add_header("Access-Contol-Allow-Origin", "*")
    return response
