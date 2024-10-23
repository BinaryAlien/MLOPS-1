from flask import Flask, request
from transformers import pipeline
import joblib


app = Flask(__name__)

regression = joblib.load("regression.joblib")
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")


@app.route("/predict", methods=["POST"])
def predict():
    sample = [
        float(request.json["size"]),
        float(request.json["nb_rooms"]),
        int(request.json["garden"]),
    ]
    output = regression.predict([sample])
    return {"price": output[0]}


@app.route("/camembert", methods=["POST"])
def camembert():
    return camembert_fill_mask(request.json["input"])
