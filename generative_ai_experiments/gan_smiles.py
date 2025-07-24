import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Helper function to encode SMILES strings
def encode_smiles(smiles, max_length, char_to_int):
    encoded = np.zeros((max_length, len(char_to_int)), dtype=np.float32)
    for i, char in enumerate(smiles):
        if char in char_to_int:
            encoded[i, char_to_int[char]] = 1.0
    return encoded

# Helper function to decode one-hot encoded SMILES back to string
def decode_smiles(encoded, int_to_char):
    smiles = ''
    for row in encoded:
        index = torch.argmax(row).item()
        smiles += int_to_char[index]
    return smiles.strip()

# Custom Dataset class for SMILES
class SMILESDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.smiles = self.data['smiles'].astype(str)
        self.chars = sorted(set(char for smile in self.smiles for char in smile))
        self.char_to_int = {char: i for i, char in enumerate(self.chars)}
        self.int_to_char = {i: char for i, char in enumerate(self.chars)}
        self.max_length = max(len(smile) for smile in self.smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        encoded = encode_smiles(smiles, self.max_length, self.char_to_int)
        return torch.tensor(encoded)

# Define the Generator and Discriminator Networks
class Generator(nn.Module):
    def __init__(self, input_dim, max_length, vocab_size):
        super(Generator, self).__init__()
        output_dim = max_length * vocab_size
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, max_length, vocab_size):
        super(Discriminator, self).__init__()
        input_dim = max_length * vocab_size
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def train_gan(dataset, epochs=200, batch_size=32, learning_rate=0.0002, noise_dim=100):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    generator = Generator(noise_dim, dataset.max_length, len(dataset.chars))
    discriminator = Discriminator(dataset.max_length, len(dataset.chars))
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, real_samples in enumerate(dataloader):
            real_samples_flat = real_samples.view(batch_size, -1)
            real_labels = torch.ones((batch_size, 1))
            fake_labels = torch.zeros((batch_size, 1))

            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_samples_flat), real_labels)
            noise = torch.randn(batch_size, noise_dim)
            fake_samples = generator(noise)
            fake_loss = criterion(discriminator(fake_samples), fake_labels)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, noise_dim)
            fake_samples = generator(noise)
            g_loss = criterion(discriminator(fake_samples), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Generate 10 new SMILES strings after training
    generator.eval()
    noise = torch.randn(10, noise_dim)
    with torch.no_grad():
        generated_samples = generator(noise)
        generated_samples = generated_samples.view(10, dataset.max_length, len(dataset.chars))
        generated_smiles = [decode_smiles(sample, dataset.int_to_char) for sample in generated_samples]
    
    print("Generated SMILES strings:")
    for smiles in generated_smiles:
        print(smiles.replace(' ', ''))  # Remove spaces from decoded strings

if __name__ == "__main__":
    dataset = SMILESDataset('local_SMILES_file.csv')
    train_gan(dataset)
