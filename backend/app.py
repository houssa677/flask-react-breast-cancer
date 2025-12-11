from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import logging
import os

from model import predict_cancer as model_predict_cancer, load_and_train_model, generate_correlation_images

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = app.logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CORR_DIR = os.path.join(BASE_DIR, "static", "corr")

model = None
feature_names = None


def init_model_and_corr():
    global model, feature_names

    # Load or train the model


    if os.path.exists(MODEL_PATH):
        app.logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            model, feature_names = pickle.load(f)
    else:
        app.logger.info("model.pkl introuvable → entraînement du modèle et sauvegarde...")
        model, feature_names = load_and_train_model()
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((model, feature_names), f)

    # Generate the correlation matrices
    if not os.path.exists(CORR_DIR) or not all(
        os.path.exists(os.path.join(CORR_DIR, fname))
        for fname in ["corr_global.png", "corr_mean.png", "corr_error.png", "corr_worst.png"]
    ):
        app.logger.info("Génération des matrices de corrélation...")
        generate_correlation_images()
    else:
        app.logger.info("Matrices de corrélation déjà présentes.")


init_model_and_corr()


@app.route("/")
def home():
    return jsonify({"message": "Backend Breast Cancer Flask is running ✅"})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or feature_names is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    data = request.get_json()
    logger.info(f"Données reçues pour la prédiction: {data}")

    features_list = [data.get(feature) for feature in feature_names]

    if None in features_list:
        return jsonify({"error": "Certaines caractéristiques sont manquantes"}), 400

    if not all(isinstance(f, (int, float)) for f in features_list):
        return jsonify({"error": "Toutes les caractéristiques doivent être numériques"}), 400

    probability, prediction = model_predict_cancer(model, features_list, feature_names)
    result = "Malignant" if prediction == 1 else "Benign"

    return jsonify({
        "probability": float(probability),
        "diagnosis": result
    })


@app.route("/correlation_global")
def correlation_global():
    return send_from_directory(CORR_DIR, "corr_global.png")


@app.route("/correlation_mean")
def correlation_mean():
    return send_from_directory(CORR_DIR, "corr_mean.png")


@app.route("/correlation_error")
def correlation_error():
    return send_from_directory(CORR_DIR, "corr_error.png")


@app.route("/correlation_worst")
def correlation_worst():
    return send_from_directory(CORR_DIR, "corr_worst.png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)




