import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn import metrics
from pysmiles import read_smiles
import networkx as nx
from bayes_opt import BayesianOptimization

# ---------------------- Config ---------------------- #

# List of assay names, from Tox21-like bioassay dataset
assays = [
    'ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1',
    'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG',
    'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE'
]

# Choose assay index to train on
assay_index = 0  # Modify as needed
assay_name = assays[assay_index]

# Define input and output directories
input_dir = Path('../01_assay_data_processing')
output_dir = Path('GCN')
output_dir.mkdir(exist_ok=True)

# Configure global output file to track final optimisation results
results_file = output_dir / f'GCN_{assay_name}_bayesian_optimisation.csv'
results_file.write_text('avg_max_test_auc,num_conv_layers,num_channels,num_MLP_layers,MLP_layer_neurons,learning_rate,prob_dropout,batch_size_config')

# ---------------------- Data Loading ---------------------- #

# Function to load JSON from file
def load_json(file: Path):
    with open(file, 'r') as f:
        return json.load(f)

# Load fold-split dictionary and atom feature dictionary
folds_dict = load_json(input_dir / f'{assay_name}_folds.json')
atom_properties_dict = load_json(input_dir / 'atom_properties_dict.json')

# Determine number of node features (atom properties + hydrogen count)
num_node_features = len(atom_properties_dict['C']) + 1

# Function to convert a molecule graph to PyTorch Geometric tensor
def smiles_to_tensor(smiles_graph: nx.Graph, label: int) -> Data:
    # Extract atom element types and hydrogen counts
    elements = nx.get_node_attributes(smiles_graph, 'element')
    hcounts = nx.get_node_attributes(smiles_graph, 'hcount')

    # Construct feature matrix (atom features + H count)
    node_features = [
        [prop if prop is not None else 0 for prop in atom_properties_dict[e]] + [hcounts[i]]
        for i, e in elements.items()
    ]

    # Double edge list to make it undirected
    edge_list = [list(e) for edge in smiles_graph.edges for e in (edge, edge[::-1])]

    # Return PyG Data object
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    )

# Build dataset for a given split (train/test)
def build_dataset(smiles: List[str], labels: List[int], tag: str):
    print(f'Generating PyTorch tensors for {tag} data...')
    return [smiles_to_tensor(read_smiles(s), y) for s, y in zip(smiles, labels)]

# ---------------------- GCN Model Definition ---------------------- #

# Define GCN architecture with variable number of GCN and MLP layers
class GCNClassifier(torch.nn.Module):
    def __init__(self, num_conv_layers, hidden_channels, num_MLP_layers, MLP_neurons, dropout):
        super().__init__()
        torch.manual_seed(1)

        # Stack of GCN layers
        self.convs = torch.nn.ModuleList([
            GCNConv(num_node_features if i == 0 else hidden_channels, hidden_channels)
            for i in range(num_conv_layers)
        ])

        # Stack of fully connected MLP layers
        self.fcs = torch.nn.ModuleList([
            Linear(hidden_channels if i == 0 else MLP_neurons, MLP_neurons)
            for i in range(num_MLP_layers)
        ])

        # Final classifier output (2 classes)
        self.out = Linear(MLP_neurons, 2)
        self.dropout = dropout

    # Forward pass
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # Global mean pooling across graph nodes
        x = global_mean_pool(x, batch)

        for fc in self.fcs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = fc(x).relu()

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)

# ---------------------- Training & Evaluation ---------------------- #

# Training loop
def train_model(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Evaluation loop
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
    # Compute AUC score
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, correct / len(loader.dataset)

# ---------------------- Objective for Bayesian Optimisation ---------------------- #

def run_model(num_conv_layers, num_channels, num_MLP_layers, MLP_layer_neurons,
              learning_rate, prob_dropout, batch_size_config):

    # Round continuous suggestions from optimiser to integers where needed
    num_conv_layers = int(round(num_conv_layers))
    num_channels = int(round(num_channels))
    num_MLP_layers = int(round(num_MLP_layers))
    MLP_layer_neurons = int(round(MLP_layer_neurons))
    batch_size_config = int(round(batch_size_config))

    # Generate output filename based on config
    config_name = f'GCN_{num_conv_layers}_{num_channels}_{num_MLP_layers}_{MLP_layer_neurons}_{learning_rate}_{prob_dropout}_{batch_size_config}'
    config_dir = output_dir / assay_name
    config_dir.mkdir(parents=True, exist_ok=True)
    log_file = config_dir / f'{config_name}.csv'
    log_file.write_text('fold,epoch,train_acc,train_auc,test_acc,test_auc')

    max_test_aucs = []  # Track best AUC per fold

    # Loop over CV folds
    for fold, split in folds_dict['folds'].items():
        print(f'Processing Fold: {fold}')

        # Load data for this fold
        train_dataset = build_dataset(list(split['train'].keys()), [d['activity'] for d in split['train'].values()], 'train')
        test_dataset = build_dataset(list(split['test'].keys()), [d['activity'] for d in split['test'].values()], 'test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size_config, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_config)

        # Initialise model, optimizer, loss
        model = GCNClassifier(num_conv_layers, num_channels, num_MLP_layers, MLP_layer_neurons, prob_dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        test_auc_history = []

        # Train over epochs
        for epoch in range(1, 501):
            train_model(model, train_loader, optimizer, criterion)
            train_auc, train_acc = evaluate(model, train_loader)
            test_auc, test_acc = evaluate(model, test_loader)
            test_auc_history.append(test_auc)

            print(f'Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'\n{fold},{epoch},{train_acc},{train_auc},{test_acc},{test_auc}')

        # Store best test AUC for this fold
        max_test_aucs.append(max(test_auc_history))

    # Compute average of max AUCs across folds
    avg_auc = sum(max_test_aucs) / len(max_test_aucs)
    with open(results_file, 'a') as f:
        f.write(f'\n{avg_auc},{num_conv_layers},{num_channels},{num_MLP_layers},'
                f'{MLP_layer_neurons},{learning_rate},{prob_dropout},{batch_size_config}')
    return avg_auc

# ---------------------- Run Bayesian Optimisation ---------------------- #

# Define search bounds for each hyperparameter
pbounds = {
    'num_conv_layers': (3, 7),
    'num_channels': (20, 200),
    'num_MLP_layers': (3, 7),
    'MLP_layer_neurons': (50, 700),
    'learning_rate': (1e-5, 1e-3),
    'prob_dropout': (0.1, 0.5),
    'batch_size_config': (8, 256)
}

# Initialise Bayesian optimiser
optimizer = BayesianOptimization(
    f=run_model,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)

# Run optimisation (with 10 random starts, 100 iterations)
optimizer.maximize(init_points=10, n_iter=100)
