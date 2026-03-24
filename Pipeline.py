import cv2
import torch
import mediapipe as mp
import numpy as np
from collections import deque, Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from Classifier import Classifier

# =========================
# Constants
# =========================
MODEL_PATH = "classifier_train/train1/classifier.pth" # Đường dẫn model sau khi train
LANDMARKER_PATH = "hand_landmarker.task"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tự tin tối thiểu để hiển thị chữ

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
# Feature extraction & Normalization
# =========================
def landmarks_to_feature_vector(hand_landmarks):
    """
    Chuẩn hóa tọa độ (Normalize) theo vị trí cổ tay (wrist)
    Giúp model nhận diện đúng dù tay ở bất kỳ góc nào trên camera
    """
    wrist = hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        
    arr = np.array(features, dtype=np.float32)
    scale = np.max(np.abs(arr)) + 1e-6
    return arr / scale

# =========================
# Predict class
# =========================
def predict_from_landmarks(hand_landmarks, classifier_model):
    features = landmarks_to_feature_vector(hand_landmarks)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = classifier_model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence

# =========================
# Main Execution
# =========================
def main():
    # Load PyTorch Classifier
    classifier = Classifier(
        in_features=Classifier.KEYPOINTS_FLATTEN,
        out_features=Classifier.ALPHABETS
    ).to(DEVICE)
    
    # Dùng weights_only=True để bảo mật khi load model
    classifier.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    classifier.eval()

    # Load MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Khởi tạo Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Bộ đệm để làm mượt kết quả dự đoán (tránh bị giật chữ)
    prediction_buffer = deque(maxlen=10)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            h, w, _ = frame.shape
            label_text = "No hand detected"

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                pts = []
                
                # Vẽ điểm
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    pts.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                # Vẽ đường nối
                for i, j in HAND_CONNECTIONS:
                    cv2.line(frame, pts[i], pts[j], (255, 0, 0), 2)

                # Dự đoán chữ cái
                pred_idx, confidence = predict_from_landmarks(hand_landmarks, classifier)
                
                # Thêm vào buffer để lấy kết quả ổn định nhất trong 10 frame gần nhất
                prediction_buffer.append(pred_idx)
                stable_pred = Counter(prediction_buffer).most_common(1)[0][0]

                # Kiểm tra ngưỡng tự tin
                if confidence >= CONFIDENCE_THRESHOLD:
                    label_text = f"{CLASS_NAMES[stable_pred]} ({confidence:.2f})"
                else:
                    label_text = f"? ({confidence:.2f})"

                # Hiển thị chữ phía trên bàn tay
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts)
                cv2.putText(
                    frame,
                    label_text,
                    (x0, max(y0 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    4
                )
            else:
                # Nếu không thấy tay, làm trống buffer
                prediction_buffer.clear()
                cv2.putText(
                    frame,
                    label_text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    4
                )

            cv2.imshow("ASL Gesture Pipeline", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()