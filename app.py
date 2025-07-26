from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

model = joblib.load("nifty_lr_model.pkl")
raw_features = joblib.load("features.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        values = [float(data[feature]) for feature in raw_features]
        prediction = float(model.predict([values])[0])
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({"features": raw_features})
@app.route("/weights", methods=["GET"])
def get_weights():
    coefficients = model.coef_
    intercept = model.intercept_
    return jsonify({"weights": coefficients.tolist(), "intercept": intercept})
if __name__ == "__main__":
    app.run(debug=True)
