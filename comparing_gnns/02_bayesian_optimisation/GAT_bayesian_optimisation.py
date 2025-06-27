import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn import metrics
from pysmiles import read_smiles
import networkx as nx
from bayes_opt import BayesianOptimization


# ---------------------- Config ---------------------- #

# List of assays
assays = [
    'ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1',
    'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG',
    'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE'
]

# Choose assay by index
assay_index = 0
assay_name = assays[assay_index]

# Specify input and output paths
input_dir = Path('../01_assay_data_processing')
output_dir = Path('GAT')
# Configures directory
output_dir.mkdir(exist_ok=True)

# Configures results file
results_file = output_dir / f'GAT_{assay_name}_bayesian_optimisation.csv'
results_file.write_text('avg_max_test_auc,num_conv_layers,num_channels,num_MLP_layers,MLP_layer_neurons,learning_rate,prob_dropout,batch_size_config,n_heads')

# ---------------------- Data Loading ---------------------- #

# Function for reading json file
def load_json(file: Path):
    with open(file, 'r') as f:
        return json.load(f)

# Obtain folds and atom features data
folds_dict = load_json(input_dir / f'{assay_name}_folds.json')
atom_properties_dict = load_json(input_dir / 'atom_properties_dict.json')

# Increase number of node (atom) features, as another (non-generalisable) one (implied H-atoms) will be calculated.
num_node_features = len(atom_properties_dict['C']) + 1

# Function for converting SMILES-derived graph into PyG tensor format
def smiles_to_tensor(smiles_graph: nx.Graph, label: int) -> Data:
    
    # Find elements and H-acounts of atoms in molecular graph
    elements = nx.get_node_attributes(smiles_graph, 'element')
    hcounts = nx.get_node_attributes(smiles_graph, 'hcount')
    
    # Append H-acounts to node features
    node_features = [
        [prop if prop is not None else 0 for prop in atom_properties_dict[e]]
        + [hcounts[i]] for i, e in elements.items()
    ]
    
    # Format edge list
    edge_list = [
        list(e) for edge in smiles_graph.edges for e in (edge, edge[::-1])
    ]
    
    # Return PyG-compatible object, containing graphs in tensor data
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    )

# Function for constructing dataset
def build_dataset(smiles_list: List[str], labels: List[int], tag: str):
    print(f'Generating PyTorch tensors for {tag} data...')
    return [smiles_to_tensor(read_smiles(s), l) for s, l in zip(smiles_list, labels)]

# ---------------------- Model ---------------------- #

# Define GAT classifier
class GATClassifier(torch.nn.Module):
    def __init__(self, num_conv_layers, hidden_channels, num_MLP_layers, MLP_neurons, n_heads, dropout):
        super().__init__()
        torch.manual_seed(1)
        
        # Arbitrary number of GAT layers
        self.convs = torch.nn.ModuleList([
            GATConv(num_node_features if i == 0 else hidden_channels * n_heads, hidden_channels, heads=n_heads)
            for i in range(num_conv_layers)
        ])
        
        # Arbitrary number of MLP layers
        self.fcs = torch.nn.ModuleList([
            Linear(hidden_channels * n_heads if i == 0 else MLP_neurons, MLP_neurons)
            for i in range(num_MLP_layers)
        ])
        
        # 2 neurons in output layer
        self.out = Linear(MLP_neurons, 2)
        self.dropout = dropout
        
    # Define forward function
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            
        # Include global mean pooling, after GAT layers
        x = global_mean_pool(x, batch)
        for fc in self.fcs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = fc(x).relu()
        return self.out(F.dropout(x, p=self.dropout, training=self.training))

# ---------------------- Training / Evaluation ---------------------- #

# Model evaluation function
def evaluate(model, loader, criterion):
    # Initiate evluation mode
    model.eval()
    # Configure variable and lists for monitoring performance
    correct, y_true, y_prob = 0, [], []
    # Iterate over batches
    with torch.no_grad():
        for data in loader:
            # Find predictions and compare against true labels
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1).cpu().numpy()
            # Update lists for tracking performance
            correct += int((pred == data.y).sum())
            y_true += list(data.y)
            y_prob += [p[1] for p in prob]
            
    # Compute ROC AUC score
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, correct / len(loader.dataset)

# Define training function
def train_model(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# ---------------------- Optimisation Objective ---------------------- #

# Define function for running model (for Bayesian optimisation)
def run_model(num_conv_layers, num_channels, num_MLP_layers,
              MLP_layer_neurons, learning_rate, prob_dropout,
              batch_size_config, n_heads):
    
    # Force some discrete variables, to prevent errors from continuous values being explored
    num_conv_layers = int(round(num_conv_layers))
    num_channels = int(round(num_channels))
    num_MLP_layers = int(round(num_MLP_layers))
    MLP_layer_neurons = int(round(MLP_layer_neurons))
    batch_size_config = int(round(batch_size_config))
    n_heads = int(round(n_heads))
    
    # Configure files for saving data and logs
    config_name = f'GAT_{num_conv_layers}_{num_channels}_{num_MLP_layers}_{MLP_layer_neurons}_{learning_rate}_{prob_dropout}_{batch_size_config}_{n_heads}'
    config_dir = output_dir / assay_name
    config_dir.mkdir(parents=True, exist_ok=True)
    log_file = config_dir / f'{config_name}.csv'
    log_file.write_text('fold,epoch,train_acc,train_auc,test_acc,test_auc')
    
    # List to store maximum AUC scores on test set
    max_test_aucs = []
    
    # Iterate over folds
    for fold, data_split in folds_dict['folds'].items():
        print(f'\n[Fold: {fold}]')
        
        # Configure training set
        train_dataset = build_dataset(
            list(data_split['train'].keys()),
            [d['activity'] for d in data_split['train'].values()],
            'train'
        )
        # Configure testing set
        test_dataset = build_dataset(
            list(data_split['test'].keys()),
            [d['activity'] for d in data_split['test'].values()],
            'test'
        )
        
        # Configure data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size_config, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_config)
        
        # Define model
        model = GATClassifier(num_conv_layers, num_channels, num_MLP_layers,
                              MLP_layer_neurons, n_heads, prob_dropout)
        # Define optimisation algorithm and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Record historic AUC values on testing set, for the given fold
        test_auc_history = []
        
        # Training/testing loop
        for epoch in range(1, 501):
            # Train model and log performance on training set
            train_model(model, train_loader, optimizer, criterion)
            train_auc, train_acc = evaluate(model, train_loader, criterion)
            # Test model and log performance on testing set
            test_auc, test_acc = evaluate(model, test_loader, criterion)
            test_auc_history.append(test_auc)
            
            # Update console with progress
            print(f'Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}')
            # Record progress
            with open(log_file, 'a') as f:
                f.write(f'\n{fold},{epoch},{train_acc},{train_auc},{test_acc},{test_auc}')
        
        # Store best AUC score to wider list of maximum AUC scores, over all folds
        max_test_aucs.append(max(test_auc_history))
    
    # Find average max AUC score across all folds
    avg_auc = sum(max_test_aucs) / len(max_test_aucs)
    # Record these final results
    with open(results_file, 'a') as f:
        f.write(f'\n{avg_auc},{num_conv_layers},{num_channels},{num_MLP_layers},'
                f'{MLP_layer_neurons},{learning_rate},{prob_dropout},{batch_size_config},{n_heads}')
    return avg_auc

# ---------------------- Bayesian Optimisation ---------------------- #

# Define search space of hyperparameter optimisation
pbounds = {
    'num_conv_layers': (3, 7),
    'num_channels': (20, 200),
    'num_MLP_layers': (3, 7),
    'MLP_layer_neurons': (50, 700),
    'learning_rate': (1e-5, 1e-3),
    'prob_dropout': (0.1, 0.5),
    'batch_size_config': (8, 256),
    'n_heads': (2, 10)
}

# Configure Bayesian optimisation
optimizer = BayesianOptimization(
    f=run_model,
    pbounds=pbounds,
    verbose=2,
    random_state=1
)

# Initiate optimisation
optimizer.maximize(init_points=10, n_iter=100)
