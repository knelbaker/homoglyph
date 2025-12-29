"""
Training Script

This script trains the Convolutional Autoencoder on the generated dataset.
It loads images from the dataset directory, creates a PyTorch DataLoader,
and optimizes the model using Mean Squared Error (MSE) loss to minimize
reconstruction error.
"""

import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import ConvAutoencoder

# Configuration
DATA_DIR = "dataset/images"        # Directory containing generated images
MODEL_SAVE_PATH = "homoglyph_model.pth" # Path to save the trained model
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharDataset(Dataset):
    """
    Custom Dataset class for loading character images.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to validity images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.transform = transform
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}. Run generate_data.py first.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Open image and ensure it's grayscale (L)
            image = Image.open(img_path).convert("L") 
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image in case of error to fail gracefully
            return torch.zeros((1, 64, 64))

def train():
    """
    Main training loop.
    1. Sets up data loaders.
    2. Initializes model, criterion (MSE), and optimizer (Adam).
    3. Runs training for specified epochs.
    4. Saves the trained model weights.
    """
    print(f"Using device: {DEVICE}")

    # Standard transforms: Convert to tensor (scales to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    # Initialize Dataset and DataLoader
    dataset = CharDataset(DATA_DIR, transform=transform)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss() # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, data) # Compare output to input (Autoencoder)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save trained model state
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
