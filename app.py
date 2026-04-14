"""ECG Classification REST API — Gradio + Keras
Deploy on HuggingFace Spaces or run locally: python app.py
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import gradio as gr

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

model = keras.saving.load_model("ecg_cnn_final.keras")
params = np.load("normalisation_params.npz")
MEAN = params["mean"].squeeze()
STD = params["std"].squeeze()


def predict_ecg(file):
    ecg = np.load(file.name).astype(np.float32)
    if ecg.shape != (1000, 12):
        return {"error": f"Expected (1000, 12), got {ecg.shape}"}
    ecg = (ecg - MEAN) / STD
    probs = model.predict(ecg[np.newaxis], verbose=0).squeeze()
    return {sc: float(f"{p:.4f}") for sc, p in zip(SUPERCLASSES, probs)}


demo = gr.Interface(
    fn=predict_ecg,
    inputs=gr.File(label="Upload ECG (.npy, shape 1000x12)"),
    outputs=gr.JSON(label="Predicted Probabilities"),
    title="ECG Cardiovascular Disease Classifier",
    description="Upload a 12-lead ECG (100 Hz, 10s) as .npy. Predicts NORM, MI, STTC, CD, HYP.",
)

if __name__ == "__main__":
    demo.launch()
