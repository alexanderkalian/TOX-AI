import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load dataset
df = pd.read_csv('mutagenicity.csv').dropna(subset=['smiles', 'mutagenicity'])
df['mutagenicity'] = df['mutagenicity'].astype(int)

# Balance training set
pos = df[df['mutagenicity'] == 1]
neg = df[df['mutagenicity'] == 0]
n = min(len(pos), len(neg))
train_pos, testval_pos = train_test_split(pos, train_size=0.7, random_state=SEED)
train_neg, testval_neg = train_test_split(neg, train_size=0.7, random_state=SEED)
train_df = pd.concat([train_pos[:n], train_neg[:n]]).sample(frac=1, random_state=SEED)
testval_df = pd.concat([testval_pos, testval_neg])
val_df, test_df = train_test_split(testval_df, test_size=0.5, random_state=SEED)

# Tokenizer and vocab
tokenizer = BasicSmilesTokenizer()
all_tokens = set(tok for sm in df['smiles'] for tok in tokenizer.tokenize(sm))
vocab = {tok: i + 1 for i, tok in enumerate(sorted(all_tokens))}
vocab['<PAD>'] = 0
pad_token = '<PAD>'
pad_id = 0
max_len = 128

# Encoding function
def encode(smiles):
    tokens = tokenizer.tokenize(smiles)
    ids = [vocab.get(tok, pad_id) for tok in tokens[:max_len]]
    attn = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        attn.append(0)
    return ids, attn

# Dataset class
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.data = [encode(sm) for sm in smiles_list]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, attn = self.data[idx]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.bool),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )

# Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, dim_ff=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, attn_mask):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.embed(x) + self.pos_embed(pos)
        x = self.encoder(x, src_key_padding_mask=~attn_mask)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(1)

# Dataloaders
train_ds = SMILESDataset(train_df['smiles'].tolist(), train_df['mutagenicity'].tolist())
val_ds = SMILESDataset(val_df['smiles'].tolist(), val_df['mutagenicity'].tolist())
test_ds = SMILESDataset(test_df['smiles'].tolist(), test_df['mutagenicity'].tolist())

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# Setup model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerClassifier(vocab_size=len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

# Evaluation function
def evaluate(loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    bin_preds = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, bin_preds)
    auc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, bin_preds).ravel()
    sens = tp / (tp + fn + 1e-6)
    spec = tn / (tn + fp + 1e-6)
    return acc, auc, sens, spec

# Training loop
results = []
for epoch in range(1, 11):
    model.train()
    total_loss = 0
    for x, mask, y in tqdm(train_loader, desc=f'Epoch {epoch}'):
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_metrics = evaluate(train_loader)
    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    results.append({
        'epoch': epoch,
        'loss': total_loss / len(train_loader),
        'train_accuracy': train_metrics[0],
        'train_auc': train_metrics[1],
        'train_sensitivity': train_metrics[2],
        'train_specificity': train_metrics[3],
        'val_accuracy': val_metrics[0],
        'val_auc': val_metrics[1],
        'val_sensitivity': val_metrics[2],
        'val_specificity': val_metrics[3],
        'test_accuracy': test_metrics[0],
        'test_auc': test_metrics[1],
        'test_sensitivity': test_metrics[2],
        'test_specificity': test_metrics[3],
    })

# Save results
pd.DataFrame(results).to_csv('transformer_clf_results.csv', index=False)
print('Training complete. Results saved to transformer_clf_results.csv')
