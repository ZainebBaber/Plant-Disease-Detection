from logger.visualization import evaluate_model, plot_curves
from model.PlantCnn_model import PlantCNN
import torch
import os
import torch.nn as nn
import torch.optim as optim
import json
from trainer.trainer import Trainer
from utils.split_dataset import stratified_split_and_save
from utils.util import remove_duplicates
from data.data_loader import get_dataloaders

def main():
    print("\nStep1\n")
    with open("config.json") as f:
        config=json.load(f)

    print("\nStep2\n")
    #cleaning data
    clean_data=remove_duplicates(config["data_dir"], target_classes=config["classes"])

    print("\n Step 2.5\n")
    if not os.path.exists("dataset/train"):
        stratified_split_and_save(clean_data)

    
    print("\nStep3\n")
    #loading data
    train_loader, val_loader, test_loader=get_dataloaders("dataset", batch_size=config["batch_size"])

    print("\nStep4\n") 
    # Model
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = PlantCNN(num_classes=len(config["classes"])).to(device)

    print("\nStep5\n")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"],weight_decay=1e-4)


    print("\nStep6\n") 
    trainer=Trainer(model, train_loader, val_loader,  optimizer,criterion, device)
    train_losses, val_losses, train_accs, val_accs =trainer.train(config["epochs"])

    print("\nStep7\n") 
    # ===== Visualization & Evaluation =====
    plot_curves(train_losses, val_losses, train_accs, val_accs)  # plots loss & accuracy curves
    evaluate_model(model, test_loader, device, config["classes"])  # computes test accuracy + confusion matrix



if __name__ == "__main__":
    main()