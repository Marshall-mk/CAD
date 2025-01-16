import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def calculate_mean_std(data_path, batch_size=32, num_workers=4):
    # Define basic transformations (only ToTensor is needed for this calculation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize variables
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    print("Calculating mean and std...")
    
    # First pass: calculate mean
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
    mean /= len(dataloader)

    # Second pass: calculate std
    for inputs, _ in dataloader:
        for i in range(3):
            std[i] += ((inputs[:, i, :, :] - mean[i])**2).mean()
    std = torch.sqrt(std / len(dataloader))

    return mean.numpy(), std.numpy()

if __name__ == "__main__":
    # Set your data path here
    train_data_path = "../../../dataset/3_class_train/"
    
    mean, std = calculate_mean_std(train_data_path)
    
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")