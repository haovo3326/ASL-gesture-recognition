from Classifier import Classifier
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np

# =========================
# Class labels
# =========================
chars = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# =========================
# Paths
# =========================
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, "classifier_dataset")

images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")
features_dir = os.path.join(dataset_dir, "features")

images_train_dir = os.path.join(images_dir, "train")
images_val_dir = os.path.join(images_dir, "val")
labels_train_dir = os.path.join(labels_dir, "train")
labels_val_dir = os.path.join(labels_dir, "val")
features_train_dir = os.path.join(features_dir, "train")
features_val_dir = os.path.join(features_dir, "val")

classifier_train_dir = os.path.join(project_dir, "classifier_train")
os.makedirs(classifier_train_dir, exist_ok=True)

# =========================
# Classifier
# =========================
MODEL_PATH = "classifier_train/train2/classifier.pth"
LANDMARKER_PATH = "hand_landmarker.task"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Classifier(
    in_features=Classifier.KEYPOINTS_FLATTEN,
    out_features=len(chars)
)

classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier = classifier.to(DEVICE)

print("Using device:", DEVICE)
print("Model device:", next(classifier.parameters()).device)

# =========================
# Training setup
# =========================
epochs = 100
batch_size = 32
learning_rate = 1e-4
patience = 10

best_val_loss = float("inf")
counter = 0

criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Keep total sample space based on images
train_files = sorted([
    f for f in os.listdir(images_train_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

val_files = sorted([
    f for f in os.listdir(images_val_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

train_N = len(train_files)
val_N = len(val_files)

train_ids = list(range(train_N))
val_ids = list(range(val_N))

# Metrics files
current_train_dir = "train4"
current_train_path = os.path.join(classifier_train_dir, current_train_dir)
os.makedirs(current_train_path, exist_ok=True)

train_loss_file = os.path.join(current_train_path, "train_loss.txt")
train_acc_file = os.path.join(current_train_path, "train_acc.txt")
val_loss_file = os.path.join(current_train_path, "val_loss.txt")
val_acc_file = os.path.join(current_train_path, "val_acc.txt")
model_save_path = os.path.join(current_train_path, "classifier.pth")

# Optional: clear old logs at start
open(train_loss_file, "w").close()
open(train_acc_file, "w").close()
open(val_loss_file, "w").close()
open(val_acc_file, "w").close()

# =========================
# Helper
# =========================
def load_feature(feature_path):
    features = np.load(feature_path)

    if features.ndim != 1:
        features = features.reshape(-1)

    return features.astype(np.float32)

# =========================
# Training loop
# =========================
for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}/{epochs}...")

    random.shuffle(train_ids)
    random.shuffle(val_ids)

    # -------------------------
    # Training
    # -------------------------
    classifier.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    num_batches = (train_N + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, train_N, batch_size)):
        print(f"Epoch {epoch + 1}/{epochs}, train batch {batch_idx + 1}/{num_batches}")

        end = min(start + batch_size, train_N)
        batch_ids = train_ids[start:end]

        batch_x = []
        batch_y = []

        for idx in batch_ids:
            image_name = train_files[idx]
            image_stem = os.path.splitext(image_name)[0]

            feature_path = os.path.join(features_train_dir, f"{image_stem}.npy")
            lbl_path = os.path.join(labels_train_dir, f"{image_stem}.txt")

            # Feature may not exist because preprocessing skipped undetected samples
            if not os.path.exists(feature_path):
                continue

            if not os.path.exists(lbl_path):
                continue

            features = load_feature(feature_path)

            with open(lbl_path, "r", encoding="utf-8") as f:
                label = f.read().strip()

            if label not in ("del", "space", "nothing"):
                label = label.upper()

            if label not in char_to_idx:
                print(f"Skipping unknown label: {label}")
                continue

            batch_x.append(features)
            batch_y.append(char_to_idx[label])

        if len(batch_x) == 0:
            continue

        batch_x = torch.tensor(np.array(batch_x), dtype=torch.float32, device=DEVICE)
        batch_y = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()

        logits = classifier(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * batch_x.size(0)

        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
    train_acc = train_correct / train_total if train_total > 0 else 0.0

    # -------------------------
    # Validation
    # -------------------------
    classifier.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        num_val_batches = (val_N + batch_size - 1) // batch_size

        for batch_idx, start in enumerate(range(0, val_N, batch_size)):
            print(f"Epoch {epoch + 1}/{epochs}, val batch {batch_idx + 1}/{num_val_batches}")

            end = min(start + batch_size, val_N)
            batch_ids = val_ids[start:end]

            batch_x = []
            batch_y = []

            for idx in batch_ids:
                image_name = val_files[idx]
                image_stem = os.path.splitext(image_name)[0]

                feature_path = os.path.join(features_val_dir, f"{image_stem}.npy")
                lbl_path = os.path.join(labels_val_dir, f"{image_stem}.txt")

                if not os.path.exists(feature_path):
                    continue

                if not os.path.exists(lbl_path):
                    continue

                features = load_feature(feature_path)

                with open(lbl_path, "r", encoding="utf-8") as f:
                    label = f.read().strip()

                if label not in ("del", "space", "nothing"):
                    label = label.upper()

                if label not in char_to_idx:
                    print(f"Skipping unknown label: {label}")
                    continue

                batch_x.append(features)
                batch_y.append(char_to_idx[label])

            if len(batch_x) == 0:
                continue

            batch_x = torch.tensor(np.array(batch_x), dtype=torch.float32, device=DEVICE)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)

            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)

            val_loss_sum += loss.item() * batch_x.size(0)

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # -------------------------
    # Log metrics
    # -------------------------
    with open(train_loss_file, "a", encoding="utf-8") as f:
        f.write(f"{train_loss}\n")

    with open(train_acc_file, "a", encoding="utf-8") as f:
        f.write(f"{train_acc}\n")

    with open(val_loss_file, "a", encoding="utf-8") as f:
        f.write(f"{val_loss}\n")

    with open(val_acc_file, "a", encoding="utf-8") as f:
        f.write(f"{val_acc}\n")

    # -------------------------
    # Save best model
    # -------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(classifier.state_dict(), model_save_path)
        print(f"Best model saved to: {model_save_path}")
    else:
        counter += 1
        print(f"Patience counter: {counter}/{patience}")

    if counter >= patience:
        print("Early stopping triggered")
        break

# Final save
torch.save(classifier.state_dict(), model_save_path)
print(f"Model saved to: {model_save_path}")