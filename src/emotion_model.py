import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = "datasets/new data"

IMG_HEIGHT = 48
IMG_WIDTH = 48

BATCH_SIZE = 32
SEED = 123

# ==================================================
# LOAD DATASET
# ==================================================

train_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

val_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nClasses détectées :")
print(class_names)

# ==================================================
# COMPTER LES IMAGES
# ==================================================

print("\nNombre d'images par classe :")

data_dir = Path(f"{DATA_DIR}/train")

for folder in sorted(data_dir.iterdir()):
    if folder.is_dir():
        count = len(list(folder.glob("*")))
        print(f"{folder.name}: {count}")

# ==================================================
# NORMALISATION
# ==================================================

normalization = Rescaling(1.0 / 255)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==================================================
# DATA AUGMENTATION
# ==================================================

data_augmentation = Sequential([

    RandomFlip("horizontal"),

    RandomRotation(
        factor=0.1
    ),

    RandomZoom(
        height_factor=0.1,
        width_factor=0.1
    )
])

# ==================================================
# CNN MODEL
# ==================================================

model = Sequential([

    data_augmentation,

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(48, 48, 1)
    ),

    BatchNormalization(),

    MaxPooling2D(),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(),

    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D(),

    Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(
        0.5
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ==================================================
# COMPILE MODEL
# ==================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# ==================================================
# MODEL SUMMARY
# ==================================================

print("\nArchitecture du modèle :\n")

model.summary()

# ==================================================
# CALLBACKS
# ==================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=2,

        verbose=1

    )
]

# ==================================================
# TRAINING
# ==================================================

print("\nDébut de l'entraînement...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=60,

    callbacks=callbacks
)

# ==================================================
# SAVE MODEL
# ==================================================

model.save(
    "models/emotion_model.keras"
)

print("\nModèle sauvegardé.")

# ==================================================
# RESULTS
# ==================================================

print("\nRésultats finaux :")

print(
    f"Train Accuracy : "
    f"{history.history['accuracy'][-1]:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{history.history['val_accuracy'][-1]:.4f}"
)