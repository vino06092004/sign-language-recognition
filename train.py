import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 12

# 1. Resolve dataset directory path automatically
BASE_DIR = os.path.join("dataset", "asl_alphabet_train")
if os.path.exists(os.path.join(BASE_DIR, "asl_alphabet_train")):
    DATASET_DIR = os.path.join(BASE_DIR, "asl_alphabet_train")
elif os.path.exists(BASE_DIR):
    DATASET_DIR = BASE_DIR
else:
    DATASET_DIR = "dataset"

print(f"Loading dataset from: {DATASET_DIR}...")

MODEL_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "sign_language_model.keras")
CLASSES_SAVE_PATH = os.path.join(MODEL_DIR, "classes.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Load Dataset (80% Train, 20% Validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
print(f"Loaded {len(class_names)} classes: {class_names}")

# Optimize pipeline speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Augmentation for robust recognition
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.1)
], name="data_augmentation")

# 4. Build CNN Architecture
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Callbacks & Training
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

print("\nStarting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# 6. Save Classes and Best Model
with open(CLASSES_SAVE_PATH, "w") as f:
    f.write("\n".join(class_names))

# Ensure the final model is saved if early stopping did not trigger a checkpoint save
if not os.path.exists(MODEL_SAVE_PATH):
    model.save(MODEL_SAVE_PATH)

print(f"\nTraining complete!")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Classes saved to: {CLASSES_SAVE_PATH}")