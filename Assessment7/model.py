import torch
from  torch import nn
import torch.nn.functional  as F
from torch import nn

#Input img -> Hiddem dim -> mea, std -> Paramatrizatian trick ->
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim=784, h_dim1=512, h_dim2=256, z_dim=20):
        super().__init__()

        # Encoder (3 layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(h_dim2, z_dim)
        self.sigma_layer = nn.Linear(h_dim2, z_dim)

        # Decoder (3 layer MLP)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        sigma = torch.exp(self.sigma_layer(h))  # Ensure positivity
        return mu, sigma

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, sigma

# Uji model
if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)
    vae = VariationalAutoEncoder()
    out, mu, sigma = vae(x)
    print(out.shape, mu.shape, sigma.shape)
