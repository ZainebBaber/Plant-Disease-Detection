import os
import torch
import json


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer,criterion,device):
        self.model=model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')

        self.train_losses=[]
        self.val_losses=[]
        self.train_accs=[]
        self.val_accs=[]



    def train(self, epochs):

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # ===== TRAINING =====
            self.model.train()

            correct = 0
            total = 0
            running_loss=0

            for images, labels in self.train_loader:
                images = images.to(self.device)  #moving data to device either cpu or gpu
                labels = labels.to(self.device)

                self.optimizer.zero_grad()        #  clear old gradients
                outputs = self.model(images)      #  forward pass ,passing images to model
                loss = self.criterion(outputs, labels)  # compute loss
                loss.backward()              # backpropagation
                self.optimizer.step()             # update weights

                running_loss += loss.item() #overall loss by adding each item loss

                _, predicted = torch.max(outputs, 1) #finds class with highest score that is the model prediction
                correct += (predicted == labels).sum().item() #total correct predictions
                total += labels.size(0)


            self.train_losses.append(running_loss / len(self.train_loader))
            self.train_accs.append(correct / total)

            # ===== VALIDATION =====
            self.model.eval()
            correct, total, running_loss = 0, 0, 0

            with torch.no_grad(): #no gradients , no weight updates
                for images, labels in self.val_loader:
                    images = images.to(self.device)  #everything else is same as training loop
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

            self.val_losses.append(running_loss / len(self.val_loader))
            self.val_accs.append(correct / total)

            print(f"Epoch {epoch+1}: "
                f"Train Acc={self.train_accs[-1]:.3f}, "
                f"Val Acc={self.val_accs[-1]:.3f}")
            # ===== SAVE BEST MODEL =====
            epoch_val_loss = sum(self.val_losses) / len(self.val_losses)  # mean validation loss for the epoch

            if epoch_val_loss < self.best_val_loss:
                self.best_val_loss = epoch_val_loss
                save_dir = "saved/models"
                os.makedirs(save_dir, exist_ok=True)

                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, "best_model.pth")
                )
                print("Best model saved!")


        return self.train_losses, self.val_losses, self.train_accs, self.val_accs