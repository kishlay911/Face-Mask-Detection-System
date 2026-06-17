import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ----------------------------
# PATHS
# ----------------------------
DATASET_PATH = r"train\data"
with_mask_path = os.path.join(DATASET_PATH, "with_mask")
without_mask_path = os.path.join(DATASET_PATH, "without_mask")

print("[INFO] Loading images...")

data = []
labels = []

# -------------------------------------------------
# Load all images from the two folders
# -------------------------------------------------
for category, label in zip(["with_mask", "without_mask"], ["with_mask", "without_mask"]):
    folder = os.path.join(DATASET_PATH, category)

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

# Convert to numpy array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# -------------------------
# Label Binarization
# -------------------------
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# --------------------------
# Train Test Split
# --------------------------
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# --------------------------
# Model Building
# --------------------------
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
print("[INFO] Compiling model...")
opt = Adam(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
print("[INFO] Training model...")
H = model.fit(
    trainX, trainY,
    batch_size=32,
    validation_data=(testX, testY),
    epochs=10
)

# Save model
print("[INFO] Saving mask detector model...")
model.save("mask_detector.h5")
print("[INFO] Training completed successfully!")
