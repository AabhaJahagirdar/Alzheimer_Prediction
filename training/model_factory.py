from tensorflow.keras.applications import (
    DenseNet121, ResNet50, InceptionV3, EfficientNetB0
)
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

NUM_CLASSES = 4

def build_model(name, input_shape=(224,224,3)):
    if name == "densenet":
        base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "resnet":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "inception":
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Invalid model name")

    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(base.input, output)
    return model, base
