from flask import Flask, request
import joblib


app = Flask(__name__)
model = joblib.load("regression.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    sample = [
        float(request.json["size"]),
        float(request.json["nb_rooms"]),
        int(request.json["garden"]),
    ]
    output = model.predict([sample])
    return {"price": output[0]}
