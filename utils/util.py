import os
import shutil
import hashlib
import random
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def remove_duplicates(dataset_path, target_classes=None):
    """
    Remove duplicates in target classes inside dataset_path
    target_classes: list of class folder names to include
    """
    seen_hashes = {}
    clean_data = []
    duplicates = []

    if target_classes is None:
        target_classes = os.listdir(dataset_path)  # all classes

    for label in target_classes:
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            print(f"Skipping {label}, not a folder")
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Skip if not a file
            if not os.path.isfile(img_path):
                continue

            # Only process images
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_hash = get_image_hash(img_path)

            if img_hash not in seen_hashes:
                seen_hashes[img_hash] = img_path
                clean_data.append((img_path, label))
            else:
                duplicates.append(img_path)  # store duplicate
                print(f"Duplicate found: {img_path} (same as {seen_hashes[img_hash]})")

    print(f"Total images after cleaning: {len(clean_data)}")
    print(f"Total duplicates found: {len(duplicates)}")
    return clean_data



def show_test_predictions(model, dataset, classes, device, num_images=6):
    model.eval()
    plt.figure(figsize=(15, 8))

    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_batch)
            _, pred = torch.max(outputs, 1)

        plt.subplot(2, 3, i + 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"True: {classes[label]}\nPred: {classes[pred.item()]}")
        plt.axis("off")
    plt.show()
    

