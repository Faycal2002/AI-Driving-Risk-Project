import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# ==================================================
# LOAD DATASET
# ==================================================

DATA_DIR = "datasets/new data"

train_ds = image_dataset_from_directory(
    f"{DATA_DIR}/train",
    image_size=(48, 48),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    f"{DATA_DIR}/test",
    image_size=(48, 48),
    batch_size=32
)

print("\nClasses found:")
print(train_ds.class_names)

# ==================================================
# SHOW SAMPLE IMAGES
# ==================================================

for images, labels in train_ds.take(1):

    plt.figure(figsize=(8, 8))

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(f"Class {labels[i].numpy()}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()

# ==================================================
# BUILD CNN MODEL
# ==================================================

model = Sequential([

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
    epochs=10
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
model.save("models/emotion_model.keras")

print("Emotion model saved!")