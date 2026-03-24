import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# --- Hand-pose detector ---
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(options)
# --------------------------

# # --- Load image ---
# project_dir = os.path.dirname(os.path.abspath(__file__))
# img_path = os.path.join(project_dir, "classifier_dataset", "images", "train", "IMG2701.jpeg")
#
# img = cv2.imread(img_path)
# if img is None:
#     raise FileNotFoundError("Image not found")
#
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# mp_image = mp.Image(
#     image_format=mp.ImageFormat.SRGB,
#     data=rgb
# )
#
# # --- Detect ---
# result = landmarker.detect(mp_image)
#
# h, w, _ = rgb.shape
#
# if result.hand_landmarks:
#     for hand_landmarks in result.hand_landmarks:
#         pts = []
#
#         # draw points
#         for lm in hand_landmarks:
#             x = int(lm.x * w)
#             y = int(lm.y * h)
#             pts.append((x, y))
#
#             cv2.circle(rgb, (x, y), 4, (0, 255, 0), -1)
#
#         # draw connections
#         for i, j in HAND_CONNECTIONS:
#             cv2.line(rgb, pts[i], pts[j], (255, 0, 0), 2)
#
# # --- Show ---
# cv2.imshow("Hand Landmarks", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand-pose inference ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    # ---------------------------

    h, w, _ = frame.shape
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pts = []

            # collect pixel coordinates
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                pts.append((x, y))
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # draw connections
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, pts[i], pts[j], (255, 0, 0), 2)

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()