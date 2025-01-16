import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from utils_medmamba import focal_loss, plot_confusion_matrix
import matplotlib.pyplot as plt
import logging
from MedMamba import VSSM as medmamba
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, recall_score, f1_score, precision_score
import warnings
warnings.filterwarnings('ignore')


def save_class_indices(train_dataset):
    """Save class indices mapping to JSON."""
    class_to_idx = train_dataset.class_to_idx
    class_mapping = {val: key for key, val in class_to_idx.items()}
    with open('class_indices.json', 'w') as json_file:
        json.dump(class_mapping, json_file, indent=4)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # Data preprocessing
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.6104, 0.5033, 0.4965), (0.2507, 0.2288, 0.2383))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.6104, 0.5033, 0.4965), (0.2507, 0.2288, 0.2383))
        ])
    }

    # Ensure dataset paths exist
    train_path = "../../../dataset/train/"
    val_path = "../../../dataset/val/"
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Training or validation dataset path does not exist.")

    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=val_path, transform=data_transform["val"])

    # Save class indices mapping
    save_class_indices(train_dataset)

    # Dataset properties
    train_num = len(train_dataset)
    val_num = len(validate_dataset)
    class_weights = [0.4916, 0.5084]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # DataLoader
    batch_size = 32
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Adjust for system capability
    print(f'Using {num_workers} dataloader workers per process.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # Model, Loss, and Optimizer
    net = medmamba(num_classes=2)
    net.to(device)
    patience = 10  # Number of epochs to wait for improvement
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    best_val_loss = float('inf')
    epochs_without_improvement = 0


    # Training parameters
    epochs = 100
    best_acc = 0.0
    best_loss = float('inf')
    save_path = './mamba_skinBimaryCELoss_AdamNet.pth'

    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for images, labels in train_bar:
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicts = torch.max(outputs, 1)
            correct_train += torch.sum(predicts == labels.to(device)).item()
            total_train += labels.size(0)

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss.item():.3f}"

        train_accuracy = correct_train / total_train
        # train kappa score
        train_kappa = cohen_kappa_score(labels.cpu().numpy(), predicts.cpu().numpy())

        # Validation phase
        net.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_images, val_labels in val_bar:
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

                _, predicts = torch.max(outputs, 1)
                correct_val += torch.sum(predicts == val_labels.to(device)).item()
                total_val += val_labels.size(0)

                all_labels.append(val_labels.to(device))
                all_preds.append(predicts)

            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

        val_accuracy = correct_val / total_val
        val_loss /= len(validate_loader)

        # Metrics
        conf_matrix = confusion_matrix(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        kappa = cohen_kappa_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        recall = recall_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        precision = precision_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='weighted')
        class_report = classification_report(all_labels.cpu().numpy(), all_preds.cpu().numpy())

        # print(f'Confusion Matrix:\n{conf_matrix}')
        print(f'Classification Report:\n{class_report}')

        # Log metrics
        metrics_file = "./mamba_binary_CELoss_Adam_metrics.csv"
        if not os.path.exists(metrics_file):
            pd.DataFrame(columns=["Epoch", "Accuracy", "Precision", "Recall", "F1-Score", "Loss", "Kappa"]).to_csv(
                metrics_file, index=False
            )
        pd.DataFrame(
            [[epoch + 1, val_accuracy, precision, recall, f1, val_loss, kappa]],
            columns=["Epoch", "Accuracy", "Precision", "Recall", "F1-Score", "Loss", "Kappa"]
        ).to_csv(metrics_file, mode='a', header=False, index=False)

        # Save confusion matrix plot
        try:
            plot_confusion_matrix(conf_matrix, classes=["Nevus", "Others"], title="Validation Confusion Matrix")
            plt.savefig(f"./mamba_binary_CELoss_Adam_confusion_matrix_epoch_{epoch + 1}.png")
        except Exception as e:
            logging.error(f"Error while plotting confusion matrix: {e}")

        print(f"[Epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} "
              f"train_accuracy: {train_accuracy:.3f} train_kappa: {train_kappa:.3f} val_loss: {val_loss:.3f} val_accuracy: {val_accuracy:.3f} val_kappa: {kappa:.3f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save(net.state_dict(), save_path)
                print("Saved model with best accuracy.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Accuracy: {best_acc:.3f}")
    print(f"Best Loss: {best_loss:.3f}")
    print("Finished Training")


if __name__ == '__main__':
    main()
