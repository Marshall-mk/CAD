import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
from tqdm.notebook import tqdm, trange

def get_dataloader(image_folder, transform, batch_size=64, shuffle=True, val_split=None):
    dataset = datasets.ImageFolder(
        root=image_folder,
        transform=transform
    )
    # Ignore the labels by extracting only images
    dataset.imgs = [img[0] for img in dataset.imgs]

    # Custom dataset to return only images
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths, transform):
            self.img_paths = img_paths
            self.transform = transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            img = datasets.folder.default_loader(img_path)  # Load image
            if self.transform:
                img = self.transform(img)
            return img

    dataset = ImageDataset(dataset.imgs, transform)

    if val_split:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
        return train_loader, val_loader

    # If no split, use the entire dataset for training
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10)
    return train_loader

def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )
        self.linear = nn.Linear(1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = self.linear(x.reshape(bs, -1))
        return x
    
class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()

        self.conv_layers = nn.Sequential(
            conv_block(in_channels, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            conv_block(512, 1024),
        )

        # Define fully connected layers for mean and log-variance
        self.mu = nn.Linear(14*14*1024, latent_dims)
        self.logvar = nn.Linear(14*14*1024, latent_dims)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv_layers(x)
        x = x.reshape(bs, -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return (mu, logvar)
        
def conv_transpose_block(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=2,
    padding=1,
    output_padding=0,
    with_act=True,
):
    modules = [
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
    ]
    if with_act:  # Controling this will be handy later
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU())
    return nn.Sequential(*modules)

class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dims):
        super().__init__()

        self.linear = nn.Linear(latent_dims, 1024 * 14 * 14)
        self.t_conv_layers = nn.Sequential(
            conv_transpose_block(1024, 512,output_padding=1),
            conv_transpose_block(512, 256, output_padding=1),
            conv_transpose_block(256, 128, output_padding=1),
            conv_transpose_block(
                128, out_channels, output_padding=1, with_act=False
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs = x.shape[0]
        x = self.linear(x)
        x = x.reshape((bs, 1024, 14, 14))
        x = self.t_conv_layers(x)
        x = self.sigmoid(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))
          
class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def encode(self, x):
        # Returns mu, log_var
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Obtain parameters of the normal (Gaussian) distribution
        mu, logvar = self.encode(x)

        # Sample from the distribution
        std = torch.exp(0.5 * logvar)
        z = self.sample(mu, std)

        # Decode the latent point to pixel space
        reconstructed = self.decode(z)

        # Return the reconstructed image, and also the mu and logvar
        # so we can compute a distribution loss
        return reconstructed, mu, logvar

    def sample(self, mu, std):
        # Reparametrization trick
        # Sample from N(0, I), translate and scale
        eps = torch.randn_like(std)
        return mu + eps * std
    
def vae_loss(batch, reconstructed, mu, logvar):
    bs = batch.shape[0]

    # Reconstruction loss from the pixels - 1 per image
    reconstruction_loss = F.mse_loss(
        reconstructed.reshape(bs, -1),
        batch.reshape(bs, -1),
        reduction="none",
    ).sum(dim=-1)

    # KL-divergence loss, per input image
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    # Combine both losses and get the mean across images
    loss = (reconstruction_loss + kl_loss).mean(dim=0)

    return (loss, reconstruction_loss, kl_loss)

# def train_model(model, train_dataloader, val_dataloader, num_epochs=25, device="cuda", model_name="vae_model"):
#     losses = {
#         "loss": [],
#         "reconstruction_loss": [],
#         "kl_loss": [],
#     }
    
#     lr = 1e-4
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
    
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     for _ in (progress := trange(num_epochs, desc="Training")):
#         for phase in ["train", "val"]:
#             if phase == "train":
#                 model.train()  # Set model to training mode
#                 dataloader = train_dataloader
#             else:
#                 model.eval()   # Set model to evaluate mode
#                 dataloader = val_dataloader  # Use validation dataloader

#             running_loss = 0.0

#             # Iterate over data.
#             for inputs in dataloader:
#                 inputs = inputs.to(device)
#                 optimizer.zero_grad()

#                 # Forward
#                 with torch.set_grad_enabled(phase == "train"):
#                     reconstructed, mu, logvar = model(inputs)
#                     loss, reconstruction_loss, kl_loss = vae_loss(
#                         inputs, reconstructed, mu, logvar  # Use inputs instead of batch
#                     )

#                     # Display loss and store for plotting
#                     progress.set_postfix(loss=f"{loss.cpu().item():.3f}")
#                     losses["loss"].append(loss.item())
#                     losses["reconstruction_loss"].append(
#                         reconstruction_loss.mean().item()
#                     )
#                     losses["kl_loss"].append(kl_loss.mean().item())

#                     # Backward + optimize only if in training phase
#                     if phase == "train":
#                         loss.backward()
#                         optimizer.step()
#                 # Accumulate loss for validation phase
#                 if phase == "val":
#                     running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

#             # Step the scheduler after each epoch
#             if phase == "train":
#                 scheduler.step()

#             # Calculate average loss for validation phase
#             if phase == "val":
#                 epoch_loss = running_loss / len(val_dataloader.dataset)
#                 print(f"Validation Loss: {epoch_loss:.4f}")

#     # Save the trained model
#     torch.save(model.state_dict(), f'{model_name}.pth')  # Save model state dictionary

#     return losses  # Return the losses for further analysis

def train_model(model, train_dataloader, val_dataloader, num_epochs=25, device="cuda", model_name="vae_model"):
    losses = {
        "loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
    }
    
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-5)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_loss = float('inf')  # Initialize best validation loss
    for _ in (progress := trange(num_epochs, desc="Training")):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_dataloader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_dataloader  # Use validation dataloader

            running_loss = 0.0

            # Iterate over data.
            for inputs in dataloader:
                inputs = inputs.to(device)
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    reconstructed, mu, logvar = model(inputs)
                    loss, reconstruction_loss, kl_loss = vae_loss(
                        inputs, reconstructed, mu, logvar  # Use inputs instead of batch
                    )

                    # Display loss and store for plotting
                    progress.set_postfix(loss=f"{loss.cpu().item():.3f}")
                    losses["loss"].append(loss.item())
                    losses["reconstruction_loss"].append(
                        reconstruction_loss.mean().item()
                    )
                    losses["kl_loss"].append(kl_loss.mean().item())

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # Accumulate loss for validation phase
                if phase == "val":
                    running_loss += loss.item() * inputs.size(0)  # Multiply by batch size

            # Step the scheduler after each epoch
            if phase == "train":
                scheduler.step()

            # Calculate average loss for validation phase
            if phase == "val":
                epoch_loss = running_loss / len(val_dataloader.dataset)
                print(f"Validation Loss: {epoch_loss:.4f}")

                # Save the best model based on validation loss
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), f'{model_name}_best.pth')  # Save best model

    # Save the final trained model
    torch.save(model.state_dict(), f'{model_name}.pth')  # Save model state dictionary

    return losses  # Return the losses for further analysis