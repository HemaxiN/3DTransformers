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
import copy
import math

# random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.5):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, img_shape=(64, 128, 128), input_dim=2, output_dim=2, embed_dim=768, patch_size=16, num_heads=12, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        output = torch.sigmoid(output)
        return output

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
    def __init__(self, foreground_weight=200.0, background_weight=5.0):
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
model = UNETR().to(device)

criterion = WeightedMSELoss()  # weighted mean squared error
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training parameters
num_epochs = 200
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
