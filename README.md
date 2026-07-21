# Hand Gesture Alphabet Model

This project recognizes American Sign Language alphabet gestures from hand landmarks. It uses MediaPipe to detect 21 hand keypoints, converts those keypoints into a normalized 63-value feature vector, and classifies letters `A` to `Z` with a PyTorch neural network.

## Project Structure

```text
Alphabet_model/
+-- Classifier.py                  # PyTorch classifier model
+-- DetectorInference.py           # Webcam hand landmark visualization
+-- FeatureExtraction.py           # Extracts MediaPipe landmark features from images
+-- Pipeline.py                    # Realtime webcam letter prediction
+-- Preprocessing.py               # Builds train/validation image and label folders
+-- TestPipeline.py                # Tests classifier on one validation image
+-- TrainClassifer.py              # Trains the alphabet classifier
+-- hand_landmarker.task           # MediaPipe hand landmark model
+-- classifier_dataset/
|   +-- images/
|   |   +-- train/
|   |   +-- val/
|   +-- labels/
|   |   +-- train/
|   |   +-- val/
|   +-- features/
|       +-- train/
|       +-- val/
+-- classifier_train/
    +-- train1/
    +-- train2/
    +-- train3/
    +-- train4/
```

## Features

- Detects one hand with MediaPipe Hand Landmarker.
- Extracts wrist-relative 3D hand landmark features.
- Trains a neural network classifier for 26 alphabet classes.
- Runs realtime prediction from a webcam.
- Draws hand landmarks, hand connections, predicted letter, and confidence score.

## Requirements

- Python 3.10 or newer recommended
- Webcam for realtime inference
- Python packages:
  - `opencv-python`
  - `mediapipe`
  - `numpy`
  - `torch`

Install dependencies:

```bash
pip install opencv-python mediapipe numpy torch
```

If you want GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation page.

## Dataset

`Preprocessing.py` expects the ASL alphabet dataset outside this folder at:

```text
../asl_dataset/asl_alphabet_train/asl_alphabet_train/
```

Expected source layout:

```text
asl_alphabet_train/
+-- A/
+-- B/
+-- C/
...
+-- Z/
```

The preprocessing script copies 400 images per class:

- 360 images per class into `classifier_dataset/images/train`
- 40 images per class into `classifier_dataset/images/val`
- one text label per copied image into the matching `labels` folder

Current local dataset snapshot:

- Training images: 9360
- Validation images: 1040
- Training feature files: 7223
- Validation feature files: 797

The feature count can be lower than the image count because images with no detected hand are skipped.

## Workflow

Run all commands from the project root:

```bash
cd Alphabet_model
```

### 1. Prepare Images and Labels

```bash
python Preprocessing.py
```

This creates or fills:

```text
classifier_dataset/images/train
classifier_dataset/images/val
classifier_dataset/labels/train
classifier_dataset/labels/val
```

### 2. Extract Hand Landmark Features

```bash
python FeatureExtraction.py
```

This runs MediaPipe on each image and saves `.npy` feature vectors to:

```text
classifier_dataset/features/train
classifier_dataset/features/val
```

Each feature file contains 63 values:

```text
21 landmarks * (x, y, z)
```

Coordinates are normalized relative to the wrist landmark.

### 3. Train the Classifier

```bash
python TrainClassifer.py
```

Training uses:

- Model: `Classifier.py`
- Input size: 63
- Output classes: 26
- Loss: cross entropy
- Optimizer: Adam
- Early stopping patience: 10 epochs

The current script saves logs and the trained model to:

```text
classifier_train/train4/
```

Important output files:

```text
classifier_train/train4/classifier.pth
classifier_train/train4/train_loss.txt
classifier_train/train4/train_acc.txt
classifier_train/train4/val_loss.txt
classifier_train/train4/val_acc.txt
```

### 4. Test on One Image

```bash
python TestPipeline.py
```

By default, this tests:

```text
classifier_dataset/images/val/IMG162.jpeg
```

Change `IMAGE_PATH` in `TestPipeline.py` to test another image.

### 5. Run Realtime Webcam Prediction

```bash
python Pipeline.py
```

The webcam window shows:

- detected hand landmarks
- predicted alphabet letter
- confidence score
- `?` when confidence is below the threshold

Press `q` to close the webcam window.

## Extra Utility

To view MediaPipe hand landmark detection without classification:

```bash
python DetectorInference.py
```

Press `q` to exit.

## Model Details

The classifier is a small multilayer perceptron:

```text
63 -> 64 -> 128 -> 64 -> 26
```

It predicts one of:

```text
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
```

`Pipeline.py` currently loads:

```text
classifier_train/train4/classifier.pth
```

and uses a confidence threshold of:

```text
0.6
```

## Notes

- The project currently classifies static alphabet gestures only.
- Dynamic letters such as `J` and `Z` may be difficult because the model uses a single frame of landmarks.
- `TrainClassifer.py` is intentionally named as it appears in the repository, including the spelling.
- `Pipeline.py` and `TestPipeline.py` depend on `hand_landmarker.task` and the trained classifier file being present.
