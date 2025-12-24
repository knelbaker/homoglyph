import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 64 x 64
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 128 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )

        # Decoder
        self.decoder_linear = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> 1 x 64 x 64
            nn.Sigmoid() # Pixels are 0-1
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder_linear(z)
        reconstruction = self.decoder_conv(reconstruction)
        return reconstruction

    def encode(self, x):
        return self.encoder(x)

if __name__ == "__main__":
    # Simple test
    model = ConvAutoencoder()
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape
