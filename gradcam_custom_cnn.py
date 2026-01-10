import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "saved_models/custom_cnn.h5"
IMAGE_PATH = "sample_images/test_image.jpg"
OUTPUT_DIR = "gradcam_outputs"

IMG_SIZE = 160
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
model = load_model(MODEL_PATH)
model.summary()

# ==============================
# FIND LAST CONV LAYER
# ==============================
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    raise ValueError("❌ No Conv2D layer found in the model")

print(f"✅ Using last conv layer: {last_conv_layer}")

# ==============================
# LOAD & PREPROCESS IMAGE
# ==============================
img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# ==============================
# PREDICTION
# ==============================
preds = model.predict(img_array)
pred_class = np.argmax(preds[0])
confidence = preds[0][pred_class]

print(f"\n🧠 Prediction: {CLASS_NAMES[pred_class]}")
print(f"📊 Confidence: {confidence:.4f}")

# ==============================
# GRAD-CAM MODEL
# ==============================
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer).output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, pred_class]

# ==============================
# GRADIENT COMPUTATION
# ==============================
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

# ==============================
# NORMALIZE HEATMAP (SAFE)
# ==============================
heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

# ==============================
# CREATE HEATMAP IMAGE
# ==============================
heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))  # ✅ FIXED
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ==============================
# OVERLAY ON ORIGINAL IMAGE
# ==============================
original_img = cv2.imread(IMAGE_PATH)
original_img = cv2.resize(original_img, (IMG_SIZE, IMG_SIZE))

superimposed_img = cv2.addWeighted(
    original_img, 0.6,
    heatmap, 0.4,
    0
)

# ==============================
# SAVE OUTPUT
# ==============================
output_path = os.path.join(
    OUTPUT_DIR,
    f"gradcam_{CLASS_NAMES[pred_class]}.png"
)

cv2.imwrite(output_path, superimposed_img)
print(f"\n✅ Grad-CAM saved at: {output_path}")

# ==============================
# DISPLAY RESULT
# ==============================
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title(f"Grad-CAM → {CLASS_NAMES[pred_class]} ({confidence:.2%})")
plt.axis("off")
plt.show()
