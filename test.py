# test.py
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model.PlantCnn_model import PlantCNN
from data.data_loader import get_dataloaders
from utils.util import show_test_predictions


def main():
    # ===== Load config =====
    with open("config.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load ONLY test data =====
    _, _, test_loader = get_dataloaders(
        data_dir="dataset",
        batch_size=config["batch_size"]
    )

    # ===== Load model =====
    model = PlantCNN(num_classes=len(config["classes"]))
    model.load_state_dict(
        torch.load("saved/models/best_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    # ===== Testing =====
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ===== Metrics =====
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n📊 Test Results")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config["classes"],
        yticklabels=config["classes"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ===== Show sample predictions =====
    show_test_predictions(
        model=model,
        dataset=test_loader.dataset,
        classes=config["classes"],
        device=device,
        num_images=6
    )


if __name__ == "__main__":
    main()
