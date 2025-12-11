import os
import numpy as np
import pandas as pd
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# Matplotlib in server mode (no graphical window)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Absolute path to the backend (directory containing this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load the Breast Cancer dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df, data.feature_names, data.target


def load_and_train_model():
    """Load the data and train a logistic regression model."""
    df, feature_names, y = load_data()

    # Keep the DataFrame to avoid scikit-learn warnings
    X = df

    clf = LogisticRegression(max_iter=500, solver="liblinear")
    clf.fit(X, y)
    return clf, feature_names


def predict_cancer(model, features, feature_names):
    """Perform a cancer prediction from the input features."""
    feature_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(feature_df)[0]
    probability = model.predict_proba(feature_df)[0][1]
    return probability, prediction


def generate_correlation_images():
    """Generate and save correlation matrices for different feature groups."""
    df, feature_names, _ = load_data()

    # Split into mean / se / worst feature groups
    df_mean = df.iloc[:, :10]
    df_se = df.iloc[:, 10:20]
    df_worst = df.iloc[:, 20:]

    # Absolute output directory: backend/static/corr
    output_dir = os.path.join(BASE_DIR, "static", "corr")
    os.makedirs(output_dir, exist_ok=True)

    def plot_corr(df_part, title, filename, annot=False):
        plt.figure(figsize=(10, 8))
        corr_matrix = df_part.corr()

        sns.heatmap(
            corr_matrix,
            annot=annot,        # annot=False for the global matrix, True for the sub-blocks
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8} if annot else None,
        )

        plt.title(f"Correlation Matrix - {title}", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        path = os.path.join(output_dir, filename)
        print("Sauvegarde image :", path)
        plt.savefig(path, dpi=120)
        plt.close()

    # Global matrix: no annotations (30x30)
    plot_corr(df, "Global", "corr_global.png", annot=False)

    # Mean / Error / Worst: 10x10, keep annot=True
    plot_corr(df_mean, "Mean", "corr_mean.png", annot=True)
    plot_corr(df_se, "Error", "corr_error.png", annot=True)
    plot_corr(df_worst, "Worst", "corr_worst.png", annot=True)


if __name__ == "__main__":
    # Train the model
    model, feature_names = load_and_train_model()

    # Generate correlation matrix images
    generate_correlation_images()

    # Example prediction
    data = load_breast_cancer()
    X = data.data
    sample_features = X[0].tolist()
    probability, prediction = predict_cancer(model, sample_features, feature_names)
    result = "Bénin" if prediction == 0 else "Malin"
    print(f"Probabilité de cancer malin (exemple): {probability:.4f}")
    print(f"Prédiction (exemple): {result}")

    # Save the trained model
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, feature_names), f)
    print("Modèle sauvegardé :", model_path)


