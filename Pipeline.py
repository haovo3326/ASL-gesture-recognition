import cv2
import torch
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Classifier import Classifier


# =========================
# Constants
# =========================
MODEL_PATH = "classifier_train/train4/classifier.pth"
LANDMARKER_PATH = "hand_landmarker.task"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.6

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G",
    "H", "I", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =========================
# Load classifier
# =========================
classifier = Classifier(
    in_features=Classifier.KEYPOINTS_FLATTEN,
    out_features=Classifier.ALPHABETS
).to(DEVICE)

classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()


# =========================
# Load MediaPipe hand detector
# =========================
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)


# =========================
# Feature extraction
# =========================
def landmarks_to_feature_vector(hand_landmarks):
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    features = []
    for lm in hand_landmarks:
        features.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    return np.array(features, dtype=np.float32)


# =========================
# Predict class
# =========================
def predict_from_landmarks(hand_landmarks):
    features = landmarks_to_feature_vector(hand_landmarks)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    is_valid = confidence >= THRESHOLD
    return pred_idx, confidence, is_valid


# =========================
# Webcam loop
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    h, w, _ = frame.shape
    label_text = "No hand detected"
    text_color = (0, 255, 0)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        pts = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], (255, 0, 0), 2)

        pred_idx, confidence, is_valid = predict_from_landmarks(hand_landmarks)

        if is_valid:
            label_text = f"{CLASS_NAMES[pred_idx]} ({confidence:.2f})"
            text_color = (0, 255, 0)
        else:
            label_text = f"? ({confidence:.2f})"
            text_color = (0, 0, 255)

        x0 = min(p[0] for p in pts)
        y0 = min(p[1] for p in pts)

        cv2.putText(
            frame,
            label_text,
            (x0, max(y0 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            4
        )

    else:
        cv2.putText(
            frame,
            label_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            4
        )

    cv2.imshow("Webcam Hand Detector + Classifier", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()