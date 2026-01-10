import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight

# ---------------------------
# Enable Mixed Precision
# ---------------------------
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = {
    "densenet": 128,
    "resnet": 128,
    "vgg16": 128,
    "inception": 128,
    "efficientnet": 224
}
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Argument Parser
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    choices=["densenet", "resnet", "vgg16", "inception", "efficientnet", "all"],
                    help="Choose the model to train")
args = parser.parse_args()
model_name = args.model

# ---------------------------
# Data Generators
# ---------------------------
train_aug = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10
)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes

# ---------------------------
# Compute Class Weights
# ---------------------------
labels = train_gen.classes
cw = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
cw = dict(enumerate(cw))
print("Class weights:", cw)

# ---------------------------
# Model Builder
# ---------------------------
def build_model(name):
    input_shape = (IMG_SIZE[name], IMG_SIZE[name], 3)
    if name == "densenet":
        base = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "resnet":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "vgg16":
        base = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "inception":
        base = InceptionV3(include_top=False, weights="imagenet", input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        raise ValueError("Unknown model name")

    base.trainable = False  # Freeze backbone for initial training

    x = base.output
    if name == "vgg16":
        x = Flatten()(x)
    else:
        x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    out = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---------------------------
# Callbacks
# ---------------------------
def get_callbacks(name):
    return [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        ModelCheckpoint(filepath=f"{SAVE_DIR}/{name}_best.h5", monitor="val_accuracy", save_best_only=True)
    ]

# ---------------------------
# Training function
# ---------------------------
def train_model(name):
    print(f"\n================ Training {name.upper()} ================\n")
    model = build_model(name)
    model.summary()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=cw,
        callbacks=get_callbacks(name)
    )
    model.save(f"{SAVE_DIR}/{name}.h5")
    print(f"\n✔ {name} training complete! Model saved at {SAVE_DIR}/{name}.h5\n")

# ---------------------------
# Run
# ---------------------------
if model_name == "all":
    for m in ["densenet", "resnet", "vgg16", "inception", "efficientnet"]:
        train_model(m)
else:
    train_model(model_name)
