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
DATA_DIR = "dataset/images"
MODEL_SAVE_PATH = "homoglyph_model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.transform = transform
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {data_dir}. Run generate_data.py first.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("L") # Ensure grayscale
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((1, 64, 64))

def train():
    print(f"Using device: {DEVICE}")

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0, 1] range
    ])

    # Dataset and DataLoader
    dataset = CharDataset(DATA_DIR, transform=transform)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            
            # Forward
            outputs = model(data)
            loss = criterion(outputs, data)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
