import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import numpy as np

# --- Hand-pose detector ---
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)
# --------------------------

project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, "classifier_dataset")
images_dir = os.path.join(dataset_dir, "images")
features_dir = os.path.join(dataset_dir, "features")

images_train_dir = os.path.join(images_dir, "train")
images_val_dir = os.path.join(images_dir, "val")
features_train_dir = os.path.join(features_dir, "train")
features_val_dir = os.path.join(features_dir, "val")

os.makedirs(features_train_dir, exist_ok=True)
os.makedirs(features_val_dir, exist_ok=True)


def extract_and_save(image_dir, feature_dir):
    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, file_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            print(f"Skip (no hand): {image_path}")
            continue

        features = []

        hand_landmarks = result.hand_landmarks[0]
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y, lm.z])

        features = np.array(features, dtype=np.float32)

        stem = os.path.splitext(file_name)[0]
        save_path = os.path.join(feature_dir, stem + ".npy")

        np.save(save_path, features)

        print(f"Saved: {save_path}")


extract_and_save(images_train_dir, features_train_dir)
extract_and_save(images_val_dir, features_val_dir)