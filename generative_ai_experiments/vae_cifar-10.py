import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cat_indices = [i for i, (_, label) in enumerate(train_data) if label == 3]  # Cats are labeled as 3
cat_data = Subset(train_data, cat_indices)
train_loader = DataLoader(cat_data, batch_size=64, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 4 * 4, 20)
        self.fc_logvar = nn.Linear(64 * 4 * 4, 20)
        self.decoder_input = nn.Linear(20, 64 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z)
        z = z.view(-1, 64, 4, 4)
        return self.decoder(z), mu, logvar

# Initialize the model and optimizer
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

with open('cats_training.csv', 'w') as f:
    f.write('epoch,train_loss')

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader.dataset)}')
    with open('cats_training.csv', 'a') as f:
        f.write(f'\n{epoch},{train_loss / len(train_loader.dataset)}')

# Train the model
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train(epoch)

# Generate and display images
with torch.no_grad():
    z = torch.randn(10, 20).to(device)  # 10 samples, 20 dimensions
    sample = model.decoder(model.decoder_input(z).view(-1, 64, 4, 4))

    grid = make_grid(sample.cpu(), nrow=5)
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
   
