import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2 
import os

# random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Fully Convolutional Transformer
class ConvPatchEmbedding3D(nn.Module):
    """ 3D Patch Embedding using Convolutions """
    def __init__(self, in_channels=2, embed_dim=256, patch_size=(8, 8, 8)):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = nn.BatchNorm3d(embed_dim)
    def forward(self, x):
        x = self.conv(x) 
        x = self.norm(x)
        return x

class ConvTransformerBlock3D(nn.Module):
    """ Convolutional Transformer Block - Depthwise Conv replaces Attention """
    def __init__(self, embed_dim, kernel_size=3, expansion=4):
        super().__init__()
        self.conv1 = nn.Conv3d(embed_dim, embed_dim, kernel_size=kernel_size, padding=1, groups=embed_dim) 
        self.norm1 = nn.BatchNorm3d(embed_dim)
        self.conv2 = nn.Conv3d(embed_dim, embed_dim * expansion, kernel_size=1) 
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(embed_dim * expansion, embed_dim, kernel_size=1) 
        self.norm2 = nn.BatchNorm3d(embed_dim)
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.norm2(x)
        return x + residual

class FullyConvTransformer3D(nn.Module):
    """ Fully Convolutional Transformer for (128,128,64,2) Input """
    def __init__(self, in_channels=2, embed_dim=256, num_blocks=6, patch_size=(8, 8, 8), kernel_size=3):
        super().__init__()
        self.patch_embed = ConvPatchEmbedding3D(in_channels, embed_dim, patch_size)
        self.encoder = nn.Sequential(*[ConvTransformerBlock3D(embed_dim, kernel_size) for _ in range(num_blocks)])
        self.decoder = nn.ConvTranspose3d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.patch_embed(x)  # Convert input into patches
        x = self.encoder(x)  # Pass through transformer blocks
        x = self.decoder(x)  # Upsample back to original shape
        x = torch.sigmoid(x)  # Normalize output
        return x

class Microscopy3DDataset(Dataset):
    def __init__(self, image_dir, output_dir, augment=False):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.output_filenames = sorted(os.listdir(output_dir))
        self.augment = augment 
        
    def rotation(self, image, mask, angle):
        """Rotate the image and mask by a given angle."""
        rot_image = np.zeros(np.shape(image))
        rot_mask = np.zeros(np.shape(mask))
        center = (128 / 2, 128 / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        for z in range(rot_image.shape[2]):
            rot_image[:, :, z, :] = cv2.warpAffine(image[:, :, z, :], matrix, (128, 128))
            rot_mask[:, :, z, :] = cv2.warpAffine(mask[:, :, z, :], matrix, (128, 128))
        return rot_image, rot_mask

    def vertical_flip(self, image, mask):
        """Flip the image and mask vertically."""
        flipped_image = np.zeros(np.shape(image))
        flipped_mask = np.zeros(np.shape(mask))
        for z in range(flipped_image.shape[2]):
            flipped_image[:, :, z, :] = cv2.flip(image[:, :, z, :], 0)
            flipped_mask[:, :, z, :] = cv2.flip(mask[:, :, z, :], 0)
        return flipped_image, flipped_mask

    def horizontal_flip(self, image, mask):
        """Flip the image and mask horizontally."""
        flipped_image = np.zeros(np.shape(image))
        flipped_mask = np.zeros(np.shape(mask))
        for z in range(flipped_image.shape[2]):
            flipped_image[:, :, z, :] = cv2.flip(image[:, :, z, :], 1)
            flipped_mask[:, :, z, :] = cv2.flip(mask[:, :, z, :], 1)
        return flipped_image, flipped_mask

    def intensity(self, image, mask):
        """Apply brightness and contrast variations."""
        image = image.astype('float64')
        image = image * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
        image = image.astype('float64')
        return image, mask

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load the 3D image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        output_path = os.path.join(self.output_dir, self.output_filenames[idx])

        image = np.load(image_path)  # Shape: (128, 128, 64, 2)
        output = np.load(output_path)  # Shape: (128, 128, 64, 2)

        # Normalize
        image = image/255.0
        output = output/255.0

        # Apply data augmentation if enabled
        if self.augment:
            if random.random() < 0.5:
                angle = np.random.choice(np.arange(0,360,90))
                image, output = self.rotation(image, output, angle)
            if random.random() < 0.5:
                image, output = self.vertical_flip(image, output)
            if random.random() < 0.5:
                image, output = self.horizontal_flip(image, output)
            if random.random() < 0.5:
                image, output = self.intensity(image, output)

        # Convert to float32
        image = image.astype(np.float32)
        output = output.astype(np.float32)
                
        #transpose to (C, D, H, W) for PyTorch
        image = np.transpose(image, (3, 2, 0, 1))
        output = np.transpose(output, (3, 2, 0, 1))

        # Convert to tensors
        image = torch.tensor(image)
        output = torch.tensor(output)

        return image, output


class WeightedMSELoss(nn.Module):
    def __init__(self, foreground_weight=500.0, background_weight=5.0):
        """
        Weighted MSE Loss where nonzero voxels in the heatmap get higher weight.

        Args:
            foreground_weight (float): Weight for nonzero (foreground) voxels.
            background_weight (float): Weight for zero (background) voxels.
        """
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight

    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss.
        
        Args:
            predictions (tensor): Model output, shape (B, C, D, H, W).
            targets (tensor): Ground truth heatmaps, shape (B, C, D, H, W).
        
        Returns:
            Tensor: Weighted MSE loss.
        """
        # Compute base MSE loss
        mse_loss = (predictions - targets) ** 2

        # Assign weights: higher weight to nonzero (foreground) voxels
        weights = torch.where(targets > 0, self.foreground_weight, self.background_weight)

        # Apply weights
        weighted_loss = weights * mse_loss

        # Return mean loss
        return weighted_loss.mean()

# Data directories
train_image_dir = "/home/jovyan/Hemaxi/transformers/dataset/train/images"
train_output_dir = "/home/jovyan/Hemaxi/transformers/dataset/train/labels"
val_image_dir = "/home/jovyan/Hemaxi/transformers/dataset/val/images"
val_output_dir = "/home/jovyan/Hemaxi/transformers/dataset/val/labels"

# Create dataset (Enable augmentation only for training)
train_dataset = Microscopy3DDataset(train_image_dir, train_output_dir, augment=True)
val_dataset = Microscopy3DDataset(val_image_dir, val_output_dir, augment=False)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

#set device to GPU 0
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FullyConvTransformer3D().to(device)

criterion = WeightedMSELoss()  # weighted mean squared error
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training parameters
num_epochs = 5000
best_val_loss = float("inf")

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")
    torch.save(model.state_dict(), "current_model.pth")