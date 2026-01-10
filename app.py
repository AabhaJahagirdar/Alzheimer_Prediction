from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# ==============================
# MODEL CONFIG
# ==============================
MODEL_PATH = "saved_models/custom_cnn_final.h5"

CLASSES = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

last_prediction = None  # 🧠 chatbot memory

# ==============================
# LOAD MODEL
# ==============================
model = None
IMG_SIZE = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    IMG_SIZE = model.input_shape[1]
    print(f"✅ Model loaded | Input size: {IMG_SIZE}x{IMG_SIZE}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ==============================
# IMAGE PREPROCESSING
# ==============================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# ROUTES
# ==============================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    global last_prediction

    if model is None:
        return "Model not loaded."

    file = request.files.get("my_image")
    if not file:
        return "No image uploaded"

    img = Image.open(file)
    img_array = preprocess_image(img)

    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    prediction = CLASSES[class_index]
    last_prediction = prediction  # 🧠 save for chatbot

    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    img_path = os.path.join(upload_dir, file.filename)
    img.save(img_path)

    return render_template(
        "index.html",
        prediction=prediction,
        img_path=img_path
    )

# ==============================
# CHATBOT API
# ==============================
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").lower()

    if "alzheimer" in user_msg:
        reply = "Alzheimer’s disease is a neurological condition that affects memory, thinking, and behavior over time."

    elif "mri" in user_msg:
        reply = "MRI scans help analyze brain structure changes associated with Alzheimer’s."

    elif "prediction" in user_msg or "result" in user_msg:
        if last_prediction:
            reply = f"The last MRI analysis result was: {last_prediction}."
        else:
            reply = "Please upload an MRI image first to get a prediction."

    elif "prevent" in user_msg:
        reply = "Healthy lifestyle, mental activity, exercise, and early diagnosis can help reduce risk."

    else:
        reply = "I can help explain Alzheimer’s, MRI scans, or your prediction results."

    return jsonify({"reply": reply})

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
