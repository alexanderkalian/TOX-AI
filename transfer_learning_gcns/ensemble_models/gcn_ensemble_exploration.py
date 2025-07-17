import os
import json
import warnings
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pysmiles import read_smiles
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
import torch.nn.functional as F

# Suppress future warnings to reduce clutter.
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
endpoint = 'dili'  # Choose endpoint: 'dili', 'diri', or 'dict'
fold_nums = list(range(5))  # Five-fold cross-validation folds
folds_data = f'../datasets/training_datasets/{endpoint}_folds.csv'
best_model_file = f'../training_gcns/{endpoint}/results/round_1/{endpoint}_pretrained_CEETOX_H295R_OHPROG_values_fold0_best_model.pth'

best_model_fold = int((best_model_file.split('fold')[1]).split().split('_')[0])

# Load atom properties dictionary
with open(f'../{endpoint}/atom_properties_dict.json', 'r') as f:
    atom_properties_dict = json.load(f)

# Determine input size (number of node features)
num_node_features = len(atom_properties_dict['C']) + 1

# --- Helper Functions ---
def generate_tensor(pysmiles_graph, y_val):
    """Convert a molecular NetworkX graph to PyTorch Geometric Data."""
    elements = nx.get_node_attributes(pysmiles_graph, 'element')
    hcounts = nx.get_node_attributes(pysmiles_graph, 'hcount')

    node_features = [
        [prop if prop is not None else 0 for prop in atom_properties_dict[elements[n]].values()] + [hcounts[n]]
        for n in elements
    ]

    edge_index = torch.tensor([
        [u, v] for u, v in pysmiles_graph.edges for _ in (0, 1)
    ], dtype=torch.long).t().contiguous()

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.long)

    return Data(x=x, y=y, edge_index=edge_index)

def build_dataset(smiles_list, y_values):
    """Build list of Data objects from SMILES strings."""
    dataset = []
    for s, y in zip(smiles_list, y_values):
        if 'te' in s:
            s = s.replace('te', 'Te')
        mol_graph = read_smiles(s)
        dataset.append(generate_tensor(mol_graph, y))
    return dataset

class GCN(torch.nn.Module):
    """Three-layer GCN with BatchNorm and MLP classifier."""
    def __init__(self, hidden_channels, linear_neurons):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels, linear_neurons)
        self.lin2 = Linear(linear_neurons, linear_neurons)
        self.lin3 = Linear(linear_neurons, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=0.1, training=self.training)
        x = F.dropout(F.relu(self.lin2(x)), p=0.1, training=self.training)
        x = F.dropout(self.lin3(x), p=0.1, training=self.training)
        return torch.flatten(x)

def test(model, loader):
    """Evaluate model on the test loader."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            preds = (out > 0.5).float().tolist()
            predictions.extend(preds)
    return predictions

# --- Main Fold Loop ---
for fold_n in fold_nums:
    print(f"Fold {fold_n}")
    pretrained_path = f'../training_gcns/{endpoint}/results/round_1/'
    files = [f for f in os.listdir(pretrained_path) if f'fold{fold_n}' in f]

    # Collect .pth and .csv file info per model tag
    files_dict = {}
    for f in files:
        tag = f.split(f'_fold{fold_n}')[0]
        files_dict.setdefault(tag, {})
        if f.endswith('.pth'):
            files_dict[tag]['weights'] = f
        elif f.endswith('.csv'):
            df = pd.read_csv(os.path.join(pretrained_path, f))
            files_dict[tag]['data'] = f
            files_dict[tag]['max_acc'] = max(df['test_overall_accuracy'])

    # Read SMILES and labels for current fold
    df = pd.read_csv(folds_data)
    df = df[df['fold'] == fold_n]
    test_SMILES = list(df['smiles'])
    test_y = list(df[endpoint.upper()])

    # Filter out problematic SMILES
    valid = [s for s in test_SMILES if 'te' not in s or 'Te' in s]
    test_SMILES = []
    test_y_clean = []
    for s, y in zip(valid, test_y):
        try:
            _ = read_smiles(s)
            test_SMILES.append(s)
            test_y_clean.append(y)
        except Exception:
            continue

    test_dataset = build_dataset(test_SMILES, test_y_clean)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    predictions_dict = {s: [] for s in test_SMILES}

    for tag, model_info in tqdm(files_dict.items()):
        # Load model weights
        model_path = os.path.join(pretrained_path, model_info['weights'])
        model = GCN(hidden_channels=120, linear_neurons=300)
        model.load_state_dict(torch.load(model_path))
        preds = test(model, test_loader)
        for s, p in zip(test_SMILES, preds):
            predictions_dict[s].append(p)

    # --- Ensemble Voting ---
    output_file = f'ensemble_results/evc_{endpoint}_fold{fold_n}.csv'
    with open(output_file, 'w') as f:
        headers = ['smiles', 'true_label'] + list(files_dict) + ['pred_equal_vote', 'pred_weighted_vote']
        f.write(','.join(headers) + '\n')

        eq_preds, wt_preds = [], []
        for s, true_label in zip(test_SMILES, test_y_clean):
            preds = predictions_dict[s]
            row = [s, true_label] + preds
            equal_vote = round(sum(preds) / len(preds))
            eq_preds.append(equal_vote)
            weights = [((2 * p) - 1) * (files_dict[tag]['max_acc'] ** 1) for p, tag in zip(preds, files_dict)]
            weighted_vote = int(sum(weights) / len(weights) > 0)
            wt_preds.append(weighted_vote)
            row += [equal_vote, weighted_vote]
            f.write(','.join(map(str, row)) + '\n')

    # Compute final accuracy for both ensemble methods
    eq_acc = sum(p == t for p, t in zip(eq_preds, test_y_clean)) / len(test_y_clean)
    wt_acc = sum(p == t for p, t in zip(wt_preds, test_y_clean)) / len(test_y_clean)
    print(f'EVC (Equal) Accuracy: {eq_acc:.3f}')
    print(f'EVC (Weighted) Accuracy: {wt_acc:.3f}')

# --- Evaluate Best Global Model ---
print('Best model evaluation:')
df = pd.read_csv(folds_data)
df = df[df['fold'] == best_model_fold]
test_SMILES = list(df['smiles'])
test_y = list(df[endpoint.upper()])

# Filter invalid SMILES
valid = []
test_y_clean = []
for s, y in zip(test_SMILES, test_y):
    try:
        _ = read_smiles(s)
        valid.append(s)
        test_y_clean.append(y)
    except Exception:
        continue

test_dataset = build_dataset(valid, test_y_clean)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

model = GCN(hidden_channels=120, linear_neurons=300)
model.load_state_dict(torch.load(best_model_file))
all_preds = test(model, test_loader)

accuracy = sum(int(p == t) for p, t in zip(all_preds, test_y_clean)) / len(test_y_clean)
print(f'Overall accuracy: {accuracy:.3f}')
