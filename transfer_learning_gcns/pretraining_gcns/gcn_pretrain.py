import os
import json
import torch
import pandas as pd
import numpy as np
from torch.nn import Linear
from torch.nn.functional import dropout, relu
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pysmiles import read_smiles
import networkx as nx

# Configuration constants
n_folds = 5  # Number of folds for cross-validation
num_epochs = 1000  # Number of training epochs
atom_properties_file = 'atom_properties_dict.json'  # Atom properties JSON

# Directory paths
data_dir = '../datasets/pretraining_datasets'
results_dir = 'results'

# Default model parameters
default_params = {
    'hidden_channels': 120,
    'linear_neurons': 300,
    'batch_size': 64,
    'dropout': 0.1,
    'lr': 1e-4
}

# Dataset configurations
file_configs = [
    ['Ames_mutagenicity_SMILES_TU-Berlin_2009', 'labels', None],
    ['bbbp_intact', 'labels', None],
    ['ATG_PXRE_CIS', 'labels', None],
    ['CEETOX_H295R_OHPROG', 'labels', None],
    ['ATG_PXRE_CIS', 'values', 'log_val'],
    ['CEETOX_H295R_OHPROG', 'values', 'log_val'],
    ['P35968_inhibition', 'values', 'values'],
    ['P03372_EC50', 'values', 'log_val'],
    ['P14416_Ki', 'values', 'log_val']
]

# Load atom properties from JSON
with open(atom_properties_file, 'r') as f:
    atom_properties_dict = json.load(f)

# Define number of features per atom
num_node_features = len(atom_properties_dict['C']) + 1

# Function to generate a torch_geometric Data object from a molecular graph
def generate_tensor(graph, target_val):
    elements = nx.get_node_attributes(graph, 'element')  # Get element type per node
    hcounts = nx.get_node_attributes(graph, 'hcount')  # Get hydrogen counts per node

    node_features = []  # Initialize node features list
    for node in elements:
        props = atom_properties_dict[elements[node]]  # Lookup atom properties
        features = [props[k] if props[k] is not None else 0 for k in props]  # Replace None with 0
        features.append(hcounts[node])  # Append hydrogen count
        node_features.append(features)

    edges = list(graph.edges)  # List of edges
    edge_index = torch.tensor(edges + [e[::-1] for e in edges], dtype=torch.long).t().contiguous()  # Undirected edges
    x = torch.tensor(node_features, dtype=torch.float)  # Convert node features to tensor
    y = torch.tensor([target_val], dtype=torch.long)  # Target tensor

    return Data(x=x, y=y, edge_index=edge_index)  # Return Data object

# Function to build dataset from SMILES strings and target values
def build_dataset(smiles_list, target_list, dataset_type):
    print(f'Building dataset: {dataset_type}')
    dataset = []  # Initialize dataset list
    for smiles, y in zip(smiles_list, target_list):
        if 'te' in smiles:
            smiles = smiles.replace('te', 'Te')  # Fix tellurium symbol if needed
        graph = read_smiles(smiles)  # Parse SMILES to graph
        dataset.append(generate_tensor(graph, y))  # Generate tensor and add to dataset
    return dataset

# Define GCN model class
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, linear_neurons, dropout_prob):
        super().__init__()  # Call superclass init
        torch.manual_seed(1)  # For reproducibility

        # Define GCN layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        # Define fully connected layers
        self.fc1 = Linear(hidden_channels, linear_neurons)
        self.fc2 = Linear(linear_neurons, linear_neurons)
        self.fc3 = Linear(linear_neurons, 1)

        self.dropout_prob = dropout_prob  # Dropout probability

    def forward(self, x, edge_index, batch):
        x = relu(self.bn1(self.conv1(x, edge_index)))
        x = relu(self.bn2(self.conv2(x, edge_index)))
        x = relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = dropout(relu(self.fc1(x)), p=self.dropout_prob, training=self.training)
        x = dropout(relu(self.fc2(x)), p=self.dropout_prob, training=self.training)
        x = dropout(self.fc3(x), p=self.dropout_prob, training=self.training)
        return x.flatten()

# Main training loop
for i, (filetag, dtype, val_col) in enumerate(file_configs):
    for fold in range(n_folds):

        df = pd.read_csv(f'{data_dir}/{filetag}_{dtype}.csv')  # Load dataset CSV

        acc_type = 'overall_accuracy' if dtype == 'labels' else 'r2_score'  # Accuracy metric
        value_column = 'label' if dtype == 'labels' else val_col  # Which column to use for targets
        criterion = torch.nn.BCEWithLogitsLoss() if dtype == 'labels' else torch.nn.MSELoss()  # Loss function

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)  # Create results directory if missing

        run_name = f'{filetag}_{dtype}_fold{fold}'  # Run identifier
        result_path = os.path.join(results_dir, f'{run_name}.csv')  # Output CSV path

        with open(result_path, 'w') as f:
            f.write(f'epoch,train_loss,test_loss,train_{acc_type},test_{acc_type}\n')  # Write header

        train_df = df[df['fold'] != fold]  # Training split
        test_df = df[df['fold'] == fold]  # Testing split

        train_targets = train_df[value_column]
        test_targets = test_df[value_column]

        # Normalize regression values
        if dtype == 'values':
            scaler = StandardScaler().fit(train_targets.values.reshape(-1, 1))
            train_targets = scaler.transform(train_targets.values.reshape(-1, 1)).flatten()
            test_targets = scaler.transform(test_targets.values.reshape(-1, 1)).flatten()

        train_dataset = build_dataset(train_df['smiles'], train_targets, 'train')
        test_dataset = build_dataset(test_df['smiles'], test_targets, 'test')

        train_loader = DataLoader(train_dataset, batch_size=default_params['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=default_params['batch_size'], shuffle=False)

        model = GCN(default_params['hidden_channels'], default_params['linear_neurons'], default_params['dropout'])
        optimizer = torch.optim.Adam(model.parameters(), lr=default_params['lr'])

        best_test_loss = float('inf')  # Initialize best loss

        # Evaluation function
        def evaluate(loader):
            model.eval()
            total_loss = 0
            predictions = []
            labels = []
            with torch.no_grad():
                for batch in loader:
                    output = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y.float()
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    pred = (output > 0.5).float() if dtype == 'labels' else output
                    predictions.append(pred.cpu())
                    labels.append(target.cpu())
            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            if dtype == 'values':
                acc = r2_score(labels.numpy(), predictions.numpy())
            else:
                acc = (predictions == labels).float().mean().item()
            return total_loss / len(loader), acc

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, batch.y.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss, train_acc = evaluate(train_loader)
            test_loss, test_acc = evaluate(test_loader)

            print(f'{run_name} | Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
            
            # Save results
            with open(result_path, 'a') as f:
                f.write(f'{epoch},{train_loss},{test_loss},{train_acc},{test_acc}\n')
            
            # Save best model trained state
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(results_dir, f'{run_name}_best_model.pth'))