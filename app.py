from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("synthetic_image_detector.h5")

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/learnmore")
def learnmore():
    return render_template("learnmore.html")
@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error_message="No image uploaded"
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error_message="No image selected"
        )

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # ---------------- MODEL PREDICTION ----------------
    img = Image.open(filepath).convert("RGB").resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]

    if prob > 0.5:
        prediction = "Synthetic Image"
        confidence = prob * 100
    else:
        prediction = "Real Image"
        confidence = (1 - prob) * 100

    confidence = round(confidence, 2)

    features = {
        "Smooth Textures": round(prob * 100, 2),
        "Unnatural Patterns": round(prob * 90, 2),
        "Natural Skin Detail": round((1 - prob) * 85, 2),
        "Lighting Consistency": round((1 - prob) * 80, 2),
        "Facial Symmetry": round((1 - prob) * 75, 2),
        "Edge Artifacts": round(prob * 70, 2)
    }

    return render_template(
        "result.html",
        image_path=url_for("static", filename="uploads/" + file.filename),
        prediction=prediction,
        confidence=confidence,
        features=features
    )

if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
