"""
Convolutional Autoencoder Model

This module defines the PyTorch model architecture used to learn visual embeddings
of character images. We use a Convolutional Autoencoder which compresses the image
into a low-dimensional latent vector (embedding) and then attempts to reconstruct it.

The encoder part of this model is used later to generate embeddings for homoglyph discovery.
"""

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    A Convolutional Autoencoder for 64x64 grayscale images.
    
    Attributes:
        encoder (nn.Sequential): The encoder network (Conv2d layers).
        decoder_linear (nn.Linear): Linear layer to expand latent vector for decoding.
        decoder_conv (nn.Sequential): The decoder network (ConvTranspose2d layers).
    """
    def __init__(self, latent_dim=64):
        """
        Args:
            latent_dim (int): The size of the latent embedding vector. Default is 64.
        """
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Compresses 64x64 image to latent_dim
        self.encoder = nn.Sequential(
            # Input: 1 x 64 x 64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim) # -> latent_dim
        )

        # Decoder: Reconstructs 64x64 image from latent_dim
        self.decoder_linear = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 1 x 64 x 64
            nn.Sigmoid() # Output pixels are normalized to [0, 1] range
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input image tensor (batch_size, 1, 64, 64)
            
        Returns:
            torch.Tensor: Reconstructed image tensor.
        """
        z = self.encoder(x)
        reconstruction = self.decoder_linear(z)
        reconstruction = self.decoder_conv(reconstruction)
        return reconstruction

    def encode(self, x):
        """
        Encodes an input image into a latent vector.
        This is used for the homoglyph discovery phase.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Latent embedding vector.
        """
        return self.encoder(x)

if __name__ == "__main__":
    # Sanity check: Run a dummy input through the model to verify shapes
    model = ConvAutoencoder()
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape
    print("Model test passed.")
