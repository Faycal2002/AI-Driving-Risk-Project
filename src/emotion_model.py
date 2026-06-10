import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# LOAD DATASET
# ==================================================

DATA_DIR = "datasets/new data"

train_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=(48, 48),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=(48, 48),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

print("\nClasses found:")
print(train_ds.class_names)

# ==================================================
# SHOW CLASS COUNTS
# ==================================================

data_dir = Path(f"{DATA_DIR}/train")

print("\nClass counts:")

for d in sorted(data_dir.iterdir()):
    if d.is_dir():
        count = sum(
            1 for _ in d.glob("**/*")
            if _.is_file()
        )

        print(d.name, count)

# ==================================================
# SHOW SAMPLE IMAGES
# ==================================================

for images, labels in train_ds.take(1):

    plt.figure(figsize=(8, 8))

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        plt.title(
            f"Class {labels[i].numpy()}"
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==================================================
# DATA PIPELINE
# ==================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255
)

train_ds = train_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

val_ds = val_ds.map(
    lambda x, y: (
        normalization_layer(x),
        y
    )
)

# ==================================================
# DATA AUGMENTATION
# ==================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    )
])

# ==================================================
# BUILD CNN MODEL
# ==================================================

model = Sequential([

    data_augmentation,

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(48, 48, 3)
    ),

    MaxPooling2D(),

    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    MaxPooling2D(),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        7,
        activation="softmax"
    )
])

# ==================================================
# COMPILE MODEL
# ==================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Architecture:\n")

model.summary()

# ==================================================
# TRAIN MODEL
# ==================================================

print("\nStarting training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=15,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        )
    ]
)

# ==================================================
# FINAL RESULTS
# ==================================================

print("\nTraining completed.")

print(
    f"Final Training Accuracy: "
    f"{history.history['accuracy'][-1]:.4f}"
)

print(
    f"Final Validation Accuracy: "
    f"{history.history['val_accuracy'][-1]:.4f}"
)

model.save(
    "models/emotion_model.keras"
)

print("Emotion model saved!")