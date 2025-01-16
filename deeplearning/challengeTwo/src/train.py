import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from torchvision import datasets, transforms
import time
import os
import copy
import logging
from utils import focal_loss, EarlyStopper, plot_confusion_matrix
from models import (
    dense_net,
    mobilenet_v2,
    efficientnetv2_l,
    resnext50,
    convnextv2_base,
    convit_small,
    deit_base_patch16_224,
    vitbase16,
    swin_base_patch4_window7_224,
    swinv2_base_window16_256,
)
from omegaconf.omegaconf import OmegaConf
import hydra
import warnings
from sklearn.metrics import classification_report
from torch.utils.data import random_split
warnings.filterwarnings("ignore")


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    ########################################################## Dataset ##########################################################
    # prepare the data
    mean = np.array([0.6104, 0.5033, 0.4965])
    std = np.array([0.2507, 0.2288, 0.2383])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(cfg.model.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    if cfg.train.use_fraction_as_val:
      train_dataset = datasets.ImageFolder(
        root=f"../{cfg.model.train_data_path}", transform=data_transforms["train"]
      )
      val_fraction = cfg.train.val_split
      train_size = int((1 - val_fraction) * len(train_dataset))
      val_size = len(train_dataset) - train_size
      train_datasetx, val_dataset = random_split(train_dataset, [train_size, val_size])
      # load data
      train_dataloader = torch.utils.data.DataLoader(
          train_datasetx, batch_size=cfg.train.batch_size, shuffle=True, num_workers=10
      )
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=10
      )
    else:
      train_dataset = datasets.ImageFolder(
          root=f"../{cfg.model.train_data_path}", transform=data_transforms["train"]
      )
      val_dataset = datasets.ImageFolder(
          root=f"../{cfg.model.val_data_path}", transform=data_transforms["val"]
      )
      # load data
      train_dataloader = torch.utils.data.DataLoader(
          train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=10
      )
      val_dataloader = torch.utils.data.DataLoader(
          val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=10
      )

    class_names = train_dataset.classes
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    model_name = cfg.model.model_name
    class_counts = {"bcc": 1993, "mel": 2713, "scc":376}
    class_weights = []
    for i in range(len(class_names)):
        class_weights.append(1.0 / class_counts[class_names[i]])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights = class_weights / sum(class_weights)
    
    print(f"DataSet Sizes: {dataset_sizes}")
    print(f"Class Names: {class_names}")
    print(f"Class Weights: {class_weights}")
    print("Data Loaded")
    
    ########################################################## Model ##########################################################
    print("Multiple Models Training Started")
    all_models = [
        "mobilenet",
        # "densenet",
        # "resnext50",
        # "efficientnetv2_l",
        # "convnextv2_base",
        # "convit_small",
        # "deit_base_patch16_224",
        # "vitbase16",
    ]
    for model_name in all_models:
        print(f"Training {model_name}")
        # model selection
        if model_name == "densenet":
            model = dense_net(output=3)
        elif model_name == "mobilenet":
            model = mobilenet_v2(output=3)
        elif model_name == "efficientnetv2_l":
            model = efficientnetv2_l(output=3)
        elif model_name == "resnext50":
            model = resnext50(output=3)
        elif model_name == "convnextv2_base":
            model = convnextv2_base(output=3)
        elif model_name == "convit_small":
            model = convit_small(output=3)
        elif model_name == "deit_base_patch16_224":
            model = deit_base_patch16_224(output=3)
        elif model_name == "vitbase16":
            model = vitbase16(output=3)
        elif model_name == "swin_base_patch4_window7_224":
            model = swin_base_patch4_window7_224(output=3)
        elif model_name == "swinv2_base_window16_256":
            model = swinv2_base_window16_256(output=3)
        else:
            print("Please specify a valid model name")
        model.to(device)
        # criterion selection
        if cfg.train.loss == "FocalLoss":
            criterion = focal_loss(alpha=class_weights, gamma=2.0, reduction="mean")
        elif cfg.train.loss == "WeightedCELoss":
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        # optimizer selection
        if cfg.train.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
        elif cfg.train.optimizer == "SGD":
            optimizer = optim.SGD(
                model.parameters(), lr=cfg.train.learning_rate, momentum=0.9
            )
        elif cfg.train.optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=cfg.train.learning_rate)
        elif cfg.train.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        else:
            print("Please specify a valid optimizer")

        # scheduler selection
        if cfg.train.scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif cfg.train.scheduler == "MultiStepLR":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, milestones=[7, 10], gamma=0.1
            )
        elif cfg.train.scheduler == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        elif cfg.train.scheduler == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )
        elif cfg.train.scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        elif cfg.train.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=0, last_epoch=-1
            )
        else:
            print("Please specify a valid scheduler")

        since = time.time()

        print("Training Started")
        sys.stdout.flush()
        # Early stopping
        early_stopping = EarlyStopper(patience=cfg.train.patience, min_delta=10)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = np.inf
        num_epochs = cfg.train.epochs

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("=" * 30)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                    dataloader = train_dataloader
                else:
                    model.eval()
                    dataloader = val_dataloader

                running_loss = 0.0
                running_accuracy = 0.0

                all_labels = []
                all_preds = []

                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_accuracy += torch.sum(preds == labels.data)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_accuracy.double() / dataset_sizes[phase]
                epoch_kappa = cohen_kappa_score(all_labels, all_preds)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f} Kappa: {epoch_kappa:.4f}")
                sys.stdout.flush()

                # Deep copy the model
                if phase == "val" and (
                    epoch_accuracy > best_acc or epoch_loss < best_loss
                ):
                    best_acc = epoch_accuracy
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if not os.path.exists("../checkpoints"):
                        os.makedirs("../checkpoints")
                    torch.save(
                        model.state_dict(),
                        f"../checkpoints/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_best_model_{epoch + 1}.pth",
                    )
                    logging.info(f"Checkpoint {epoch + 1} saved !")

                # Compute and plot confusion matrix for validation phase
                if phase == "val":
                    cm = confusion_matrix(all_labels, all_preds)
                    print(
                        classification_report(
                            all_labels, all_preds, target_names=class_names
                        )
                    )
                    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
                    precision = np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(3)])
                    recall = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(3)])
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    val_kappa = cohen_kappa_score(all_labels, all_preds)
                    # log metrics
                    logging.info(
                        f"Epoch {epoch + 1}  Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1-Score: {f1_score:.4f} Loss: {epoch_loss:.4f} Val Kappa: {val_kappa:.4f}"
                    )
                    # save metrics
                    if not os.path.exists("../metrics"):
                        os.makedirs("../metrics")
                    if not os.path.exists(
                        f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_metrics.csv"
                    ):
                        pd.DataFrame(
                            columns=[
                                "Epoch",
                                "Accuracy",
                                "Precision",
                                "Recall",
                                "F1-Score",
                                "Loss",
                                "Kappa",
                            ]
                        ).to_csv(
                            f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_metrics.csv",
                            index=False,
                        )
                    pd.DataFrame(
                        [
                            [
                                epoch + 1,
                                accuracy,
                                precision,
                                recall,
                                f1_score,
                                epoch_loss,
                                val_kappa,
                            ]
                        ],
                        columns=[
                            "Epoch",
                            "Accuracy",
                            "Precision",
                            "Recall",
                            "F1-Score",
                            "Loss",
                            "Kappa",
                        ],
                    ).to_csv(
                        f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_metrics.csv",
                        mode="a",
                        header=False,
                        index=False,
                    )
                    try:
                        plot_confusion_matrix(
                            cm,
                            classes=class_names,
                            title=f"{phase} Confusion Matrix",
                        )
                        plt.savefig(
                            f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_confusion_matrix_{epoch + 1}.png"
                        )
                    except Exception as e:
                        logging.error(f"Error while plotting confusion matrix: {e}")
            # Update the scheduler
            if scheduler is not None:
                if phase == "val":
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

            # Check if early stopping criteria is met
            if early_stopping.early_stop(epoch_loss):
                print("Early stopping triggered. Training stopped.")
                sys.stdout.flush()
                break
        time_elapsed = time.time() - since

        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))
        print("Best val Loss: {:4f}".format(best_loss))
        sys.stdout.flush()
        # load best model weights
        model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    main()
