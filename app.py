import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

app = Flask(__name__)

MODEL_PATH = "model/MobileNetV2_best.keras"

model = load_model(MODEL_PATH)

class_names = [
    "Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        # Open and prepare image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image).astype("float32")

        # MobileNetV2 preprocessing
        img_array = preprocess_input(img_array)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(predictions)

        predicted_class = class_names[predicted_index]

        confidence = float(
            predictions[predicted_index] * 100
        )

        probabilities = {
            class_names[i]: round(
                float(predictions[i] * 100),
                2
            )
            for i in range(len(class_names))
        }

        return jsonify({
            "disease": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
