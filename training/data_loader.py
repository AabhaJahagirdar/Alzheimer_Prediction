from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_generators(data_dir):
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )

    val_test_aug = ImageDataGenerator(rescale=1./255)

    train = train_aug.flow_from_directory(
        f"{data_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val = val_test_aug.flow_from_directory(
        f"{data_dir}/val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test = val_test_aug.flow_from_directory(
        f"{data_dir}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train, val, test
