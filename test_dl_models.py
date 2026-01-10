# =====================================================
# test_dl_models.py
# Auto Evaluate ALL DL Models + Save Confusion Matrices
# =====================================================

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ================= GPU SAFETY (Mac / CUDA safe) =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ================= CONFIG =================
MODEL_DIR = "saved_models"
TEST_DIR = "dataset/test"
CONF_MATRIX_DIR = "confusion_matrices"
BATCH_SIZE = 32

os.makedirs(CONF_MATRIX_DIR, exist_ok=True)

# ================= MODEL INPUT SIZE MAP =================
MODEL_INPUT_SIZES = {
    "vgg": (224, 224),
    "resnet": (224, 224),
    "inception": (299, 299),
    "densenet": (224, 224),
    "efficientnet": (224, 224),
    "custom": (160, 160),
    "basic": (160, 160)
}

# ================= AUTO INPUT SIZE =================
def get_input_size(model_name):
    for key in MODEL_INPUT_SIZES:
        if key in model_name.lower():
            return MODEL_INPUT_SIZES[key]
    return (224, 224)

# ================= LOAD TEST DATA =================
def load_test_data(img_size):
    datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
        TEST_DIR,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

# ================= CONFUSION MATRIX PLOT =================
def save_confusion_matrix(cm, class_names, model_name):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    save_path = os.path.join(CONF_MATRIX_DIR, f"{model_name}_cm.png")
    plt.savefig(save_path)
    plt.close()

    print(f"🖼 Confusion matrix saved → {save_path}")

# ================= MODEL EVALUATION =================
def evaluate_model(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n==============================")
    print(f" Evaluating: {model_name}")
    print(f"==============================")

    img_size = get_input_size(model_name)
    test_data = load_test_data(img_size)

    model = load_model(model_path)

    y_true = test_data.classes
    y_prob = model.predict(test_data, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    class_names = list(test_data.class_indices.keys())

    # Metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, model_name)

    print("✔ Accuracy:", report["accuracy"])
    print("✔ ROC-AUC:", roc_auc)

    return {
        "Model": model_name,
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
        "ROC-AUC": roc_auc
    }

# ================= MAIN =================
if __name__ == "__main__":
    tf.keras.backend.clear_session()

    results = []

    print("\n🔍 Searching models in:", MODEL_DIR)

    for file in os.listdir(MODEL_DIR):
        if file.endswith((".h5", ".keras")):
            path = os.path.join(MODEL_DIR, file)
            try:
                results.append(evaluate_model(path))
            except Exception as e:
                print(f"❌ Failed {file}: {e}")

    # ================= RESULTS TABLE =================
    df = pd.DataFrame(results)
    df = df.sort_values(by="Accuracy", ascending=False)

    print("\n================ FINAL RESULTS ================")
    print(df)

    df.to_csv("model_evaluation_results.csv", index=False)
    print("\n✅ Metrics saved as model_evaluation_results.csv")
    print("✅ Confusion matrices saved in /confusion_matrices/")
