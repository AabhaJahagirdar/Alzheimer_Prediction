# ============================================
# test_all_modules.py (ACCURACY-BOOSTED & STABLE)
# Target Accuracy: ~88–90%
# ============================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, Flatten, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

# ================= GPU SAFETY (APPLE SILICON SAFE) =================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ================= CONFIG =================
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 40

IMG_SIZE = (176, 176)  # 🔥 Increased resolution (critical for MRI)

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
SAVE_DIR = "saved_models"
CM_DIR = "confusion_matrices"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

# ================= DATA GENERATORS =================
def get_data():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_data, test_data

# ================= CLASS WEIGHTS =================
def get_class_weights(train_data):
    y = train_data.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(weights))

# ================= CUSTOM CNN (ACCURACY TUNED) =================
def build_custom_cnn():
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    # 🔥 KEEP SPATIAL INFORMATION (KEY FIX)
    x = Flatten()(x)

    x = Dense(384, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.35)(x)  # 🔥 Reduced dropout (prevents underfitting)

    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    return Model(inputs, outputs)

# ================= TRAIN FUNCTION =================
def train_and_evaluate():
    train_data, test_data = get_data()
    class_weights = get_class_weights(train_data)
    class_names = list(train_data.class_indices.keys())

    model = build_custom_cnn()

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.3, min_lr=1e-6),
        ModelCheckpoint(f"{SAVE_DIR}/custom_cnn_final.h5", save_best_only=True)
    ]

    model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    evaluate_model(model, test_data, class_names)

# ================= EVALUATION =================
def evaluate_model(model, test_data, class_names):
    y_true = test_data.classes
    y_prob = model.predict(test_data)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Custom CNN Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(NUM_CLASSES), class_names, rotation=45)
    plt.yticks(range(NUM_CLASSES), class_names)
    plt.tight_layout()
    plt.savefig(f"{CM_DIR}/custom_cnn_cm.png")
    plt.close()

    acc = np.mean(y_pred == y_true)
    roc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    print(f"\nFinal Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

# ================= RUN =================
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    train_and_evaluate()
