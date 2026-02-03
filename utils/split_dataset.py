import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm



def stratified_split_and_save(
    clean_data,
    output_dir="dataset",
    val_ratio=0.2,
    test_ratio=0.1
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = [item[0] for item in clean_data]
    labels = [item[1] for item in clean_data]

    # -------- First split: train+val vs test --------
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=42
    )

    # -------- Second split: train vs val --------
    val_size = val_ratio / (1 - test_ratio)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=42
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    # -------- Save into folders --------
    
    for split, (paths, labels) in splits.items():
        for img_path, label in tqdm(zip(paths, labels), desc=f"Saving {split}"):
            save_dir = os.path.join(output_dir, split, label)
            os.makedirs(save_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))

    print("✅ Dataset split & saved successfully")


