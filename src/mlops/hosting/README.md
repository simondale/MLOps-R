# Hosting the Model
Once a model has been trained it can be hosted as an API to serve results.

## Setup
It is recommended to create a virtual environment before running these commands.

Run the following command to ensure dependencies are installed:
```
pip install -r requirements.txt
```

## Hosting a Flask App with gunicorn
The following command will start a server and load the model:
```
gunicorn src.mlops.hosting.hosting:app
```

## Calling the application with curl
The following command will request predictions for two records:
```
curl -vv \
  -H "Content-Type: application/json" \
  -d '{"columns":["sepal_length","sepal_width","petal_length","petal_width"],"index":[0,1],"data":[[5.1,3.5,1.4,0.2],[2.9,1.0,5.4,3.2]]}' \
  http://127.0.0.1:8000/predict
```