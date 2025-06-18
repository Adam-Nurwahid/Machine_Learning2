import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from model import VariationalAutoEncoder
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM1 = 512
H_DIM2 = 256
Z_DIM = 20
BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4

# Buat folder hasil
os.makedirs("results", exist_ok=True)

# Dataset
transform = transforms.ToTensor()
dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, optimizer
model = VariationalAutoEncoder(INPUT_DIM, H_DIM1, H_DIM2, Z_DIM).to(DEVICE)
loss_fn = nn.BCELoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (x, _) in enumerate(tqdm(loader)):
        x = x.view(-1, INPUT_DIM).to(DEVICE)

        x_recon, mu, sigma = model(x)

        recon_loss = loss_fn(x_recon, x)
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        loss = recon_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.2f}")

    # Simpan hasil rekonstruksi contoh
    with torch.no_grad():
        sample = next(iter(loader))[0][:8].to(DEVICE)
        sample_flat = sample.view(-1, INPUT_DIM)
        recon, _, _ = model(sample_flat)
        comparison = torch.cat([sample, recon.view(-1, 1, 28, 28)])
        save_image(comparison.cpu(), f"results/reconstruction_epoch{epoch+1}.png", nrow=8)

# Inference function untuk generate gambar
def inference(model, digit, num_examples=1):
    model.eval()
    images = []
    labels = []

    for x, y in dataset:
        if y == digit:
            images.append(x)
            labels.append(y)
        if len(images) >= 1:
            break

    with torch.no_grad():
        mu, sigma = model.encode(images[0].view(1, -1).to(DEVICE))
        for i in range(num_examples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = model.decode(z)
            out = out.view(-1, 1, 28, 28)
            save_image(out.cpu(), f"results/generated_digit{digit}_ex{i}.png")

# Generate 1 gambar untuk tiap digit 0–9
for digit in range(10):
    inference(model, digit)


model.eval()
all_z = []
all_labels = []

for x, y in DataLoader(dataset, batch_size=512):
    x = x.view(-1, 784).to(DEVICE)
    with torch.no_grad():
        mu, _ = model.encode(x)
        all_z.append(mu.cpu())
        all_labels.extend(y.numpy())

z_concat = torch.cat(all_z).numpy()
tsne = TSNE(n_components=2)
z_2d = tsne.fit_transform(z_concat)

plt.figure(figsize=(10, 8))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=all_labels, cmap='tab10', s=5)
plt.colorbar()
plt.title("2D t-SNE of Latent Space")
plt.savefig("results/latent_space.png")
plt.show()