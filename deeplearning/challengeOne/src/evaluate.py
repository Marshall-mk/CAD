import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
import hydra
from omegaconf.omegaconf import OmegaConf
import warnings
from utils import FocalLoss
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, cohen_kappa_score
from utils import calculate_accuracy, plot_confusion_matrix
import random
import csv
import os
import timm
import matplotlib.pyplot as plt
import sys

warnings.filterwarnings("ignore")


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)

    # prepare the data
    if cfg.train.use_image_net_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = np.array([0.6679, 0.5354, 0.5194])
        std = np.array([0.2224, 0.2053, 0.2171])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }
    test_dataset = datasets.ImageFolder(
        root=f"../{cfg.model.test_data_path}", transform=data_transforms["test"]
    )

    # load data
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=10
    )
    class_names = test_dataset.classes
    dataset_sizes = len(test_dataset)
    model_name = cfg.model.model_name
    print(dataset_sizes)
    print(class_names)
    sys.stdout.flush()
    all_models = [
        "mobilenet",
        "densenet",
        "resnext50",
        "efficientnetv2_l",
        "convnextv2_base",
        "convit_small",
        "deit_base_patch16_224",
        "vitbase16",
    ]
    for model_name in all_models:
        print(model_name)
        sys.stdout.flush()
        
        if model_name == "densenet":
            # load trained model and evaluate
            model = models.densenet161(pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/densenet_FocalLoss_Adam_ReduceLROnPlateau__val_frac_True_best_model_19.pth"
                    )
                )
                
        elif model_name == "mobilenet":
            model = models.mobilenet_v2(pretrained=True)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/mobilenet_FocalLoss_Adam_ReduceLROnPlateau__val_frac_True_best_model_18.pth"
                    )
                )
        elif model_name == "efficientnetv2_l":
            model = timm.create_model("tf_efficientnetv2_l_in21k", pretrained=True)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/efficientnetv2_l_FocalLoss_Adam_ReduceLROnPlateau__val_frac_True_best_model_19.pth"
                    )
                )

        elif model_name == "resnext50":
            model = models.resnext50_32x4d(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/resnext50_FocalLoss_Adam_ReduceLROnPlateau__val_frac_True_best_model_23.pth"
                    )
                )
        elif model_name == "convit_small":
            model = timm.create_model("convit_small", pretrained=True)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/convit_small_FocalLoss_AdamW_ReduceLROnPlateau__val_frac_True_best_model_19.pth"
                    )
                )
        elif model_name == "deit_base_patch16_224":
            model = timm.create_model("deit_base_patch16_224", pretrained=True)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/deit_base_patch16_224_FocalLoss_AdamW_ReduceLROnPlateau__val_frac_True_best_model_21.pth"
                    )
                )
        elif model_name == "vitbase16":
            model = timm.create_model("vit_base_patch16_224", pretrained=True)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/vitbase16_FocalLoss_AdamW_ReduceLROnPlateau__val_frac_True_best_model_23.pth"
                    )
                )
        elif model_name == "convnextv2_base":
            model = timm.create_model("convnextv2_base", pretrained=True)
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, 1)
            if not cfg.train.use_image_net_stats:
                model.load_state_dict(
                    torch.load(
                        "../checkpoints/80_20_full/convnextv2_base_FocalLoss_AdamW_ReduceLROnPlateau__val_frac_True_best_model_21.pth"
                    )
                )

        else:
            print("Please specify a valid model")
            pass

        model.eval()
        model.to(device)

        # criterion selection
        if cfg.train.loss == "FocalLoss":
            criterion = FocalLoss(logits=True)
        elif cfg.train.loss == "WeightedCELoss":
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.4916, 0.5084]))
        else:
            criterion = torch.nn.BCEWithLogitsLoss()

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
        sys.stdout.flush()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                outputs = outputs.view(-1)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += calculate_accuracy(
                outputs, labels
            ).item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy())

        epoch_loss = running_loss / len(test_dataloader.dataset)
        epoch_acc = running_corrects / len(test_dataloader.dataset)
        cm = confusion_matrix(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        # plot roc curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc}")
        print(f"Cohen's Kappa: {kappa}")
        tn, fp, fn, tp = cm.ravel()
        print(cm)
        print(f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}")
        print(f"Precision: {tp/(tp+fp)} Recall: {tp/(tp+fn)}  F1: {2*tp/(2*tp+fp+fn)}")
        print(f"TPR: {tp/(tp+fn)} FPR: {fp/(fp+tn)}")
        print(f"Sensitivity: {tp/(tp+fn)} Specificity: {tn/(tn+fp)}")
        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(classification_report(all_labels, all_preds))
        sys.stdout.flush()
        # save metrics
        if not os.path.exists(
            f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_metrics.csv"
        ):
            os.makedirs("../metrics", exist_ok=True)
        with open(
            f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_imagenet_weights_{cfg.train.use_image_net_stats}_test_metrics.txt",
            "w",
        ) as f:
            f.write(f"ROC AUC: {roc_auc}\n")
            f.write(f"TN: {tn} FP: {fp} FN: {fn} TP: {tp}\n")
            f.write(
                f"Precision: {tp/(tp+fp)} Recall: {tp/(tp+fn)}  F1: {2*tp/(2*tp+fp+fn)}\n"
            )
            f.write(f"TPR: {tp/(tp+fn)} FPR: {fp/(fp+tn)}\n")
            f.write(f"Sensitivity: {tp/(tp+fn)} Specificity: {tn/(tn+fp)}\n")
            f.write(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n")
            f.write(f"Cohen's Kappa: {kappa}\n")
            f.write(classification_report(all_labels, all_preds))
        # save misclassified images
        save_misclassified_images(
            test_dataset,
            test_dataloader,
            all_labels,
            all_preds,
            class_names,
            f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_imagenet_weights_{cfg.train.use_image_net_stats}_misclassified_images.csv",
        )
        sys.stdout.flush()

        try:
            plot_confusion_matrix(
                cm, classes=class_names, title="Test Confusion Matrix"
            )
            plt.savefig(
                f"../metrics/{model_name}_{cfg.train.loss}_{cfg.train.optimizer}_{cfg.train.scheduler}_imagenet_weights_{cfg.train.use_image_net_stats}_confusion_matrix_Test.png"
            )
        except Exception as e:
            print(f"Error while plotting confusion matrix: {e}")
    return epoch_loss, epoch_acc


def visualize_misclassified_images(
    test_dataset, test_dataloader, all_labels, all_preds, class_names, num_images=5
):
    misclassified_indices = [
        i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]
    ]
    random.shuffle(misclassified_indices)
    misclassified_indices = misclassified_indices[:num_images]
    all_labels = [int(a) for a in all_labels]

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i, index in enumerate(misclassified_indices):
        image, _ = test_dataset[index]
        image = image.permute(1, 2, 0).numpy()
        label = class_names[all_labels[index]]
        prediction = class_names[all_preds[index]]

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}\nPrediction: {prediction}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def save_misclassified_images(
    test_dataset, test_dataloader, all_labels, all_preds, class_names, csv_file
):
    misclassified_indices = [
        i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]
    ]
    misclassified_images = []
    misclassified_labels = []
    all_labels = [int(a) for a in all_labels]
    for index in misclassified_indices:
        image_path, _ = test_dataset.samples[index]
        label = class_names[all_labels[index]]
        misclassified_images.append(image_path)
        misclassified_labels.append(label)

    with open(csv_file, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "True Label"])
        for i in range(len(misclassified_images)):
            writer.writerow([misclassified_images[i], misclassified_labels[i]])


if __name__ == "__main__":
    main()
