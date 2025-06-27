import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn import metrics
from pysmiles import read_smiles
import networkx as nx
from bayes_opt import BayesianOptimization

# ---------------------- Config ---------------------- #

# List of supported bioassays from CompTox Chemicals Dashboard
assays = ['ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1',
          'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG',
          'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE']

# Choose assay index (0-6) to process
assay_index = 4
assay_name = assays[assay_index]

# Define input and output folders
input_dir = Path('../01_assay_data_processing')
output_dir = Path('GIN')
output_dir.mkdir(exist_ok=True)

# Configure main output results file (records AUC for each hyperparameter config)
results_file = output_dir / f'GIN_{assay_name}_bayesian_optimisation.csv'
results_file.write_text('avg_max_test_auc,num_conv_layers,num_channels,num_MLP_layers,MLP_layer_neurons,learning_rate,prob_dropout,batch_size_config')

# ---------------------- Data Loading ---------------------- #

# Load any JSON file from disk
def load_json(file: Path):
    with open(file, 'r') as f:
        return json.load(f)

# Load fold-split dictionary and atom properties
folds_dict = load_json(input_dir / f'{assay_name}_folds.json')
atom_properties_dict = load_json(input_dir / 'atom_properties_dict.json')

# Define number of node (atom) features
num_node_features = len(atom_properties_dict['C']) + 1

# Convert SMILES-derived molecule to PyTorch Geometric tensor
def smiles_to_tensor(smiles_graph: nx.Graph, label: int) -> Data:
    elements = nx.get_node_attributes(smiles_graph, 'element')
    hcounts = nx.get_node_attributes(smiles_graph, 'hcount')

    # Construct node features (physicochemical properties + hydrogen count)
    node_features = [
        [prop if prop is not None else 0 for prop in atom_properties_dict[e]] + [hcounts[i]]
        for i, e in elements.items()
    ]

    # Format bidirectional edges
    edge_list = [list(e) for edge in smiles_graph.edges for e in (edge, edge[::-1])]

    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    )

# Construct dataset from SMILES strings and binary activity labels
def build_dataset(smiles: List[str], labels: List[int], tag: str):
    print(f'Generating PyTorch tensors for {tag} data...')
    return [smiles_to_tensor(read_smiles(s), y) for s, y in zip(smiles, labels)]

# ---------------------- GIN Model Definition ---------------------- #

# Define the GIN model architecture
class GINClassifier(torch.nn.Module):
    def __init__(self, num_conv_layers, hidden_channels, num_MLP_layers, MLP_neurons, dropout):
        super().__init__()
        torch.manual_seed(1)

        # Internal MLP used by each GINConv layer
        def create_mlp(input_dim, output_dim):
            return Sequential(
                Linear(input_dim, hidden_channels), ReLU(),
                Linear(hidden_channels, hidden_channels), ReLU(),
                Linear(hidden_channels, output_dim)
            )

        # Stack of GINConv layers
        self.convs = torch.nn.ModuleList([
            GINConv(create_mlp(num_node_features if i == 0 else hidden_channels, hidden_channels))
            for i in range(num_conv_layers)
        ])

        # Stack of fully connected MLP layers
        self.fcs = torch.nn.ModuleList([
            Linear(hidden_channels if i == 0 else MLP_neurons, MLP_neurons)
            for i in range(num_MLP_layers)
        ])

        # Output classification layer (2 classes)
        self.out = Linear(MLP_neurons, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        for fc in self.fcs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = fc(x).relu()

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)

# ---------------------- Training & Evaluation ---------------------- #

# Function to train model on a batch

def train_model(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Function to evaluate model and compute AUC and accuracy

def evaluate(model, loader):
    model.eval()
    correct, y_true, y_prob = 0, [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1).cpu().numpy()
            correct += int((pred == data.y).sum())
            y_true += list(data.y)
            y_prob += [p[1] for p in prob]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, correct / len(loader.dataset)

# ---------------------- Optimisation Objective ---------------------- #

# Function to evaluate one configuration of hyperparameters

def run_model(num_conv_layers, num_channels, num_MLP_layers, MLP_layer_neurons,
              learning_rate, prob_dropout, batch_size_config):

    # Round hyperparameters to valid discrete values
    num_conv_layers = int(round(num_conv_layers))
    num_channels = int(round(num_channels))
    num_MLP_layers = int(round(num_MLP_layers))
    MLP_layer_neurons = int(round(MLP_layer_neurons))
    batch_size_config = int(round(batch_size_config))

    # Configure local results file for current hyperparameter combination
    config_name = f'GIN_{num_conv_layers}_{num_channels}_{num_MLP_layers}_{MLP_layer_neurons}_{learning_rate}_{prob_dropout}_{batch_size_config}'
    config_dir = output_dir / assay_name
    config_dir.mkdir(parents=True, exist_ok=True)
    log_file = config_dir / f'{config_name}.csv'
    log_file.write_text('fold,epoch,train_acc,train_auc,test_acc,test_auc')

    max_test_aucs = []

    # Loop over all cross-validation folds
    for fold, split in folds_dict['folds'].items():
        print(f'Processing Fold: {fold}')

        train_dataset = build_dataset(list(split['train'].keys()), [d['activity'] for d in split['train'].values()], 'train')
        test_dataset = build_dataset(list(split['test'].keys()), [d['activity'] for d in split['test'].values()], 'test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size_config, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_config)

        model = GINClassifier(num_conv_layers, num_channels, num_MLP_layers, MLP_layer_neurons, prob_dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        test_auc_history = []

        # Train model for up to 500 epochs
        for epoch in range(1, 501):
            train_model(model, train_loader, optimizer, criterion)
            train_auc, train_acc = evaluate(model, train_loader)
            test_auc, test_acc = evaluate(model, test_loader)
            test_auc_history.append(test_auc)

            print(f'Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'\n{fold},{epoch},{train_acc},{train_auc},{test_acc},{test_auc}')

        # Store best AUC from this fold
        max_test_aucs.append(max(test_auc_history))

    # Compute average of best AUCs across all folds
    avg_auc = sum(max_test_aucs) / len(max_test_aucs)
    with open(results_file, 'a') as f:
        f.write(f'\n{avg_auc},{num_conv_layers},{num_channels},{num_MLP_layers},'
                f'{MLP_layer_neurons},{learning_rate},{prob_dropout},{batch_size_config}')
    return avg_auc

# ---------------------- Bayesian Optimisation ---------------------- #

# Define bounds for hyperparameter search
pbounds = {
    'num_conv_layers': (3, 7),
    'num_channels': (20, 200),
    'num_MLP_layers': (3, 7),
    'MLP_layer_neurons': (50, 700),
    'learning_rate': (1e-5, 1e-3),
    'prob_dropout': (0.1, 0.5),
    'batch_size_config': (8, 256)
}

# Set up Bayesian optimiser from bayes_opt
optimizer = BayesianOptimization(
    f=run_model,
    pbounds=pbounds,
    verbose=2,  # Full verbosity
    random_state=1
)

# Begin optimisation
optimizer.maximize(init_points=10, n_iter=100)
