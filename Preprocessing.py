import os
import shutil
import random

project_dir = os.path.dirname(os.path.abspath(__file__))
asl_dataset_dir = os.path.join(project_dir, "asl_dataset", "asl_alphabet_train", "asl_alphabet_train")

dataset_dir = os.path.join(project_dir, "classifier_dataset")
images_dir = os.path.join(dataset_dir, "images")
labels_dir = os.path.join(dataset_dir, "labels")

images_train_dir = os.path.join(images_dir, "train")
images_val_dir = os.path.join(images_dir, "val")
labels_train_dir = os.path.join(labels_dir, "train")
labels_val_dir = os.path.join(labels_dir, "val")

chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
         "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
         "U", "V", "W", "X", "Y", "Z"]

train_counter = 0
val_counter = 0

random.seed(42)

for char in chars:
    char_dir = os.path.join(asl_dataset_dir, char)

    files = os.listdir(char_dir)
    random.shuffle(files)

    for i, file in enumerate(files):
        if i < 360:
            shutil.copy(os.path.join(char_dir, file), os.path.join(images_train_dir, f"IMG{train_counter}.jpeg"))
            with open(os.path.join(labels_train_dir, f"IMG{train_counter}.txt"), "w") as f:
                f.write(char)
            train_counter += 1
        elif i < 400:
            shutil.copy(os.path.join(char_dir, file), os.path.join(images_val_dir, f"IMG{val_counter}.jpeg"))
            with open(os.path.join(labels_val_dir, f"IMG{val_counter}.txt"), "w") as f:
                f.write(char)
            val_counter += 1
        print(f"Transferred training samples: {train_counter}")
        print(f"Transferred validating samples: {val_counter}")