"""
Optional real-time API. Run:  python -m src.predict_api
Send a POST to http://127.0.0.1:5000/predict with a JSON flow.
Unlike a 3-feature toy API, this sends the SAME features the model was trained on.
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from src.config import MODEL_PATH
from src.detect import explain, severity
from src.preprocess import engineer_features

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    df = engineer_features(pd.DataFrame([payload]))
    model = get_model()
    # Add any training columns the caller omitted (pipeline imputes them)
    for col in model.named_steps["prep"].feature_names_in_:
        if col not in df.columns:
            df[col] = None
    df = df[model.named_steps["prep"].feature_names_in_]
    prob = float(model.predict_proba(df)[0, 1])
    row = df.iloc[0].to_dict()
    return jsonify({
        "threat_detected": prob >= 0.5,
        "attack_probability": round(prob, 4),
        "severity": severity(prob) if prob >= 0.5 else "NONE",
        "reason": explain(row) if prob >= 0.5 else "traffic looks normal",
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
