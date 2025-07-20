import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Load and preprocess the data
class SmilesDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.smiles = self.data.smiles.values
        self.chars = sorted(set(char for smile in self.smiles for char in smile))
        self.char_to_index = {char: i for i, char in enumerate(self.chars)}
        self.index_to_char = {i: char for i, char in enumerate(self.chars)}
        self.max_length = max(len(smile) for smile in self.smiles)

    def smile_to_one_hot(self, smile):
        encoded = np.zeros((self.max_length, len(self.chars)), dtype=np.float32)
        for i, char in enumerate(smile):
            encoded[i, self.char_to_index[char]] = 1.0
        return encoded.flatten()

    def one_hot_to_smile(self, encoded):
        smile = ''
        for i in range(0, len(encoded), len(self.chars)):
            idx = np.argmax(encoded[i:i+len(self.chars)])
            smile += self.index_to_char[idx]
        return smile.strip()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        encoded = self.smile_to_one_hot(smile)
        return torch.tensor(encoded, dtype=torch.float), torch.tensor(encoded, dtype=torch.float)  # X and Y are the same

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)
    
    def encode(self, x):
        h1 = nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL_div

# Training the VAE
def train(model, dataset, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch}, Average Loss: {total_loss / len(train_loader.dataset)}')

# Generating new SMILES
def generate_smiles(model, dataset, num_samples=10):
    model.eval()
    generated_smiles = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, latent_dim)
            generated = model.decode(z).cpu().numpy()
            smile = dataset.one_hot_to_smile(generated[0])
            generated_smiles.append(smile)
    return generated_smiles

# Main script
if __name__ == '__main__':
    csv_file = 'data/local_smiles_file.csv'  # Update this path to your dataset
    dataset = SmilesDataset(csv_file=csv_file)
    input_dim = dataset.max_length * len(dataset.chars)
    latent_dim = 56  # Can be adjusted based on your dataset and needs
    model = VAE(input_dim, latent_dim)
    train(model, dataset, epochs=100)
    smiles = generate_smiles(model, dataset, num_samples=100)
    print('\nGenerated SMILES:')
    with open('generated_Ssmiles.csv','a') as f:
        f.write('SMILES')
        for smile in smiles:
            f.write('\n'+smile)
            print(smile)
