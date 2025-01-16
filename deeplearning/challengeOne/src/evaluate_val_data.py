import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
import hydra
from omegaconf.omegaconf import OmegaConf
import timm
import sys

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)

    # Prepare the data
    if cfg.train.use_image_net_stats:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = np.array([0.6679, 0.5354, 0.5194])
        std = np.array([0.2224, 0.2053, 0.2171])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(cfg.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    # Load the validation dataset
    val_dataset = datasets.ImageFolder(
        root=f"../{cfg.model.val_data_path}", transform=data_transforms["val"]
    )

    # Load data
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=10
    )

    # class_names = val_dataset.classes
    all_models = [
        # "mobilenet",
        # "densenet",
        # "resnext50",
        "efficientnetv2_l",
        "convnextv2_base",
        "convit_small",
        "deit_base_patch16_224",
        # "vitbase16",
    ]

    for model_name in all_models:
        print(f"Evaluating model: {model_name}")
        sys.stdout.flush()

        # Load the model
        if model_name == "densenet":
            model = models.densenet161(pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/densenet_FocalLoss_Adam_ReduceLROnPlateau_best_model_15.pth"))
        elif model_name == "mobilenet":
            model = models.mobilenet_v2(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/mobilenet_FocalLoss_Adam_ReduceLROnPlateau_best_model_21.pth"))
        elif model_name == "efficientnetv2_l":
            model = timm.create_model("tf_efficientnetv2_l_in21k", pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/efficientnetv2_l_FocalLoss_Adam_ReduceLROnPlateau_best_model_23.pth"))
        elif model_name == "resnext50":
            model = models.resnext50_32x4d(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/resnext50_FocalLoss_Adam_ReduceLROnPlateau_best_model_21.pth"))
        elif model_name == "convit_small":
            model = timm.create_model("convit_small", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/convit_small_FocalLoss_AdamW_ReduceLROnPlateau_best_model_24.pth"))
        elif model_name == "convnextv2_base":
            model = timm.create_model("convnextv2_base", pretrained=False)
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/convnextv2_base_FocalLoss_AdamW_ReduceLROnPlateau_best_model_22.pth"))
        elif model_name == "deit_base_patch16_224":
            model = timm.create_model("deit_base_patch16_224", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/deit_base_patch16_224_FocalLoss_AdamW_ReduceLROnPlateau_best_model_22.pth"))
        elif model_name == "vitbase16":
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 1)
            model.load_state_dict(torch.load("../checkpoints/vitbase16_FocalLoss_AdamW_ReduceLROnPlateau_best_model_24.pth"))
        else:
            raise ValueError(f"Model {model_name} not supported")

        model.eval()
        model.to(device)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        # Save predictions, probabilities, and ground truth to text files
        np.savetxt(f"../metrics/{model_name}_val_predictions.txt", all_preds, fmt='%d')
        np.savetxt(f"../metrics/{model_name}_val_probabilities.txt", all_probs, fmt='%.6f')
        np.savetxt(f"../metrics/{model_name}_val_ground_truth.txt", all_labels, fmt='%d')

        print(f"Results saved for {model_name}")

if __name__ == "__main__":
    main() 