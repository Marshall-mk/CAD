import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
import hydra
from omegaconf.omegaconf import OmegaConf
import os
import timm
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torch.utils.data import Dataset
import sys
import warnings
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
    from PIL import Image

    class CustomImageDataset(Dataset):
        def __init__(self, folder_path, transform=None):
            self.folder_path = folder_path
            self.transform = transform
            self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.folder_path, self.image_files[idx])
            image = Image.open(img_path).convert("RGB")  # Load the image as a PIL Image
            if self.transform:
                image = self.transform(image)  # Apply transformations
            return image, self.image_files[idx] 

    # Prepare the data
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

    # Load the dataset
    test_dataset = CustomImageDataset(
        folder_path=f"../{cfg.model.test_data_path}", transform=data_transforms["test"]
    )
    # test_dataset = datasets.ImageFolder(
    #     root=f"../{cfg.model.test_data_path}", transform=data_transforms["test"]
    # )

    # load data
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=10
    )
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
        print(f"Testing model: {model_name}")
        sys.stdout.flush()
        
        # Load the model
        if model_name == "densenet":
            model = models.densenet161(pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/densenet_FocalLoss_Adam_ReduceLROnPlateau_best_model_17.pth"))
        elif model_name == "mobilenet":
            model = models.mobilenet_v2(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/mobilenet_FocalLoss_Adam_ReduceLROnPlateau_best_model_24.pth"))
        elif model_name == "efficientnetv2_l":
            model = timm.create_model("tf_efficientnetv2_l_in21k", pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/efficientnetv2_l_FocalLoss_Adam_ReduceLROnPlateau_best_model_25.pth"))
        elif model_name == "resnext50":
            model = models.resnext50_32x4d(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/resnext50_FocalLoss_Adam_ReduceLROnPlateau_best_model_25.pth"))
        elif model_name == "convit_small":
            model = timm.create_model("convit_small", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/convit_small_FocalLoss_AdamW_ReduceLROnPlateau_best_model_19.pth"))
        elif model_name == "convnextv2_base":
            model = timm.create_model("convnextv2_base", pretrained=False)
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/convnextv2_base_FocalLoss_Adam_ReduceLROnPlateau_best_model_25.pth"))
        elif model_name == "deit_base_patch16_224":
            model = timm.create_model("deit_base_patch16_224", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/deit_base_patch16_224_FocalLoss_AdamW_ReduceLROnPlateau_best_model_24.pth"))
        elif model_name == "vitbase16":
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, 3)
            model.load_state_dict(torch.load("../checkpoints/vitbase16_FocalLoss_AdamW_ReduceLROnPlateau_best_model_25.pth"))
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        model.eval()
        model.to(device)

        all_preds = []
        all_probs = []
        all_image_names = []  # New list to store image names
        with torch.no_grad():
            for inputs, image_names in test_dataloader:  # Unpack image names
                inputs = inputs.to(device)  # Get only the images
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_image_names.extend(image_names)  # Collect image names

        # Save predictions to a text file
        predictions_file = f"../metrics/{model_name}_predictionsxx.txt"
        predictions_file_probs = f"../metrics/{model_name}_predictions_probsxx.txt"
        image_names_file = f"../metrics/{model_name}_image_names.txt"  # New file for image names

        np.savetxt(predictions_file, all_preds, fmt='%.6f')
        np.savetxt(predictions_file_probs, all_probs, fmt='%.6f')
        np.savetxt(image_names_file, all_image_names, fmt='%s')  # Save image names
        print(f"Predictions saved to {predictions_file}")
        print(f"Image names saved to {image_names_file}")

if __name__ == "__main__":
    main() 