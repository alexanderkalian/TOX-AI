import torch
from torch_geometric.data import Data
import json
from pysmiles import read_smiles
import networkx as nx
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm


import warnings

# This will ignore all warnings
warnings.filterwarnings('ignore')


### Basic variables.

# Endpoint names.
endpoints = ['dili', 'diri', 'dict']
files = [f'../datasets/training_datasets/{e}_folds.csv' for e in endpoints]

# Specify number of rounds and number of folds used - advisable to use defaaults here.
n_rounds = 1
n_folds = 5


# Function for calculating all performance metrics relevant to binar classification.
def calculate_metrics(all_labels, all_preds):
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision or Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Return as a dictionary
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1,
        "MCC": mcc
    }


# Iterates through endpoints and files.
for endpoint, filename in zip(endpoints, files):
    

    # Obtains dict of physicochemical atom properties.
    with open(f'{endpoint}/atom_properties_dict.json','r') as f:
        content = f.read()
    atom_properties_dict = json.loads(content)
    
    
    # Obtains list of best pre-trained QSAR models.
    pretrained_csv = '../01 - Pre-Training GCNs/analysis/best_GCNs.csv'
    df = pd.read_csv(pretrained_csv)
    pretrained_files = ['../01 - Pre-Training GCNs/results/'+f[:-4]+'_best_model.pth' for f in list(df['params_file'])]+[None]
    pretrained_tags = [f'{a}_{d}' for a,d in zip(list(df['assay']),list(df['dtype']))]+['None']
    
    # Function for creating tensors of arbitrary molecular graphs.
    num_node_features = len(atom_properties_dict['C'])+1
    NoneType = type(None)
    def generate_tensor(pysmiles_graph, y_val):
        elements = nx.get_node_attributes(pysmiles_graph, 'element')
        hcounts = nx.get_node_attributes(pysmiles_graph, 'hcount')
        node_features = []
        for n in elements:
            properties = atom_properties_dict[elements[n]]
            node_features.append([0 if isinstance(properties[att], NoneType) else properties[att] for att in properties])
            node_features[-1].append(hcounts[n])
        edge_list = []
        for e in pysmiles_graph.edges:
            edge_list.append(list(e))
            edge_list.append(list(reversed(e)))
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([y_val], dtype=torch.long)
        data = Data(x=x, y=y, edge_index=edge_index.t().contiguous())
        return data
    
    # Function for building a dataset of PyTorch tensors, for arbitrary SMILES.
    def build_dataset(SMILES, y_values, data_type):
        print('Generating PyTorch tensors, for molecular graphs:',data_type)
        dataset = []
        for s, y_val in zip(SMILES, y_values):
            # Catches a very specific error where Te is misrepresented as te.
            if 'te' in s:
                s = s.replace('te','Te')
            mol_graph = read_smiles(s)
            data = generate_tensor(mol_graph, y_val)
            dataset.append(data)
        return dataset

    
    # Iterates through rounds.
    for r in range(1,n_rounds+1):
        
        print(f'\n{endpoint}, round: {r}')
        
        # Creates folders for results, if not already in existence.
        if not os.path.exists(f'{endpoint}/results'):
            os.makedirs(f'{endpoint}/results')
        if not os.path.exists(f'{endpoint}/results/round_{r}'):
            os.makedirs(f'{endpoint}/results/round_{r}')
        
        
        ### Reading data.
        
        print('Reading data.')
        
        # Reads file into pandas dataframe.
        df = pd.read_csv(filename)
        
        # Finds relevant pretrained model weights files  for transfer learning.
        start_i_pt = 0
        pretrained_files = pretrained_files[start_i_pt:]
        pretrained_tags = pretrained_tags[start_i_pt:]
        
        # Iterates through pretrained files.
        for model_file, pretrained_tag in zip(pretrained_files, pretrained_tags):
            
            # Load pre-trained model parameters from .pth file
            if pretrained_tag != 'None':
                pretrained_state_dict = torch.load(model_file)
            
            # Iterates through folds.
            for n in range(n_folds):
                
                
                ### Identifies any files missing.
                
                local_config = f'{endpoint}_pretrained_{pretrained_tag}_fold{n}'
                params_file = f'{endpoint}/results/round_{r}/{local_config}_best_model.pth'
                
                if not os.path.exists(params_file):
                    
                    print(f'{pretrained_tag}, fold: {n}')
                    
                    
                    ### Defines model.
                        
                    # Ensures that positive integer values remain as positive integers, during Bayesian optimisation.
                    num_channels = 120
                    num_linear_layers = 3
                    linear_layer_neurons = 300
                    batch_size_config = 64
                    dropout_linear = 0.1
                    learning_rate = 0.0001
                    
                    # Finds how the accuracy, loss function and value type should be handled.
                    acc_type = 'overall_accuracy'
                    val = endpoint.upper()
                    loss_func = torch.nn.BCEWithLogitsLoss()
                    
                    # Defines local output file (results over current config).
                    local_config = f'{endpoint}_pretrained_{pretrained_tag}_fold{n}'
                    directory = f'{endpoint}/results/round_{r}'
                    if not os.path.exists(directory):
                        os.mkdir(directory)
                    local_output_file = directory+'/'+local_config+'.csv'
                    with open(local_output_file,'w') as f:
                        f.write(f'epoch,train_loss,test_loss,train_{acc_type},test_sens,test_spec,test_{acc_type},test_PPV,test_NPV,test_F1,test_MCC,train_AUC,test_AUC')
                    
                        
                    print('Processing training and testing data.')
                    
                    # Creates training and testing datasets.
                    
                    train_folds = [i for i in range(n_folds) if i != n]
                    df = df.loc[df['fold'] != -1]
                    
                    train_df = df[df['fold'].isin(train_folds)]
                    train_SMILES = list(train_df['smiles'])
                    train_y = list(train_df[val])
                    test_df = df[~df['fold'].isin(train_folds)]
                    test_SMILES = list(test_df['smiles'])
                    test_y = list(test_df[val])
                    
                    # Filters out problematic SMILES in the training set
                    problematic = []
                    for s in train_SMILES:
                        try:
                            mol_graph = read_smiles(s)
                        except Exception:
                            problematic.append(s)
                    train_y = [yi for yi in train_y if train_SMILES[train_y.index(yi)] not in problematic]
                    train_SMILES = [s for s in train_SMILES if s not in problematic]
                    
                    # Filters out problematic SMILES in the testing set
                    problematic = []
                    for s in test_SMILES:
                        try:
                            mol_graph = read_smiles(s)
                        except Exception:
                            problematic.append(s)
                    test_y = [yi for yi in test_y if test_SMILES[test_y.index(yi)] not in problematic]
                    test_SMILES = [s for s in test_SMILES if s not in problematic]
                    
                    # Sanity checks, by providing amounts of different classes in training and testing sets - should be balanced.
                    print('Class counts:')
                    print('Train (1/0):', train_y.count(1), train_y.count(0))
                    print('Test (1/0):', test_y.count(1), test_y.count(0))
                    
                    # Builds PyG comaptible data structures.
                    train_dataset = build_dataset(train_SMILES, train_y, 'train')
                    test_dataset = build_dataset(test_SMILES, test_y, 'test')
                    
                    # Builds DataLoader
                    train_loader = DataLoader(train_dataset, batch_size=batch_size_config, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size_config, shuffle=False)  
                    
                    # Defines the model itself.
                    
                    class GCN(torch.nn.Module):
                        def __init__(self, hidden_channels):
                            super(GCN, self).__init__() #num_node_features
                            torch.manual_seed(1)
                            # Convolutional layers.
                            self.conv1 = GCNConv(num_node_features, hidden_channels)
                            self.bn1 = BatchNorm(hidden_channels)
                            self.conv2 = GCNConv(hidden_channels, hidden_channels)
                            self.bn2 = BatchNorm(hidden_channels)
                            self.conv3 = GCNConv(hidden_channels, hidden_channels)
                            self.bn3 = BatchNorm(hidden_channels)
                            
                            # Fully connected layers
                            self.lin1 = Linear(hidden_channels, linear_layer_neurons)
                            self.lin2 = Linear(linear_layer_neurons, linear_layer_neurons)
                            self.lin3 = Linear(linear_layer_neurons, 1)
                    
                        def forward(self, x, edge_index, batch=None):
                            x = self.conv1(x, edge_index)
                            x = self.bn1(x).relu()
                            x = self.conv2(x, edge_index)
                            x = self.bn2(x).relu()
                            x = self.conv3(x, edge_index)
                            x = self.bn3(x).relu()
                    
                            # 2. Readout layer
                            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
                    
                            # 3. Apply a final classifier
                            x = F.dropout(x, p=dropout_linear, training=self.training)
                            x = self.lin1(x)
                            x = x.relu()
                            x = F.dropout(x, p=dropout_linear, training=self.training)
                            x = self.lin2(x)
                            x = x.relu()
                            x = F.dropout(x, p=dropout_linear, training=self.training)
                            x = self.lin3(x)
                            
                            return torch.flatten(x)
                    
                    # Builds model.
                    
                    print('Building model.')
                    
                    model = GCN(hidden_channels=num_channels) #100
                    # Loads pre-trained paramters.
                    if pretrained_tag != 'None':
                        model.load_state_dict(pretrained_state_dict)
                    #print(model)
                    
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #lr=0.0001
                    criterion = loss_func
                    
                    # Training function.
                    def train():
                        model.train()
                        
                        loss_tot = 0
                    
                        for data in train_loader:  # Iterate in batches over the training dataset.
                            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                            target = data.y.float()  # Ensure the target labels are floats
                            loss = criterion(out, target)  # Compute the loss.
                            loss_tot += loss.item()
                            loss.backward()  # Derive gradients.
                            optimizer.step()  # Update parameters based on gradients.
                            optimizer.zero_grad()  # Clear gradients.
                        
                        return loss_tot/len(train_loader)
                    
                    # Testing function / model performance evaluation.
                    
                    def test(loader):
                        model.eval()
                    
                        loss_tot = 0
                        
                        # Initialize lists to collect all predictions and true labels
                        all_preds = []
                        all_labels = []
                        
                        all_preds_raw = []
                        
                        with torch.no_grad():
                            for data in loader:  # Iterate in batches over the training/test dataset.
                                out = model(data.x, data.edge_index, data.batch) 
                                target = data.y.float()  # Ensure the target labels are floats
                                pred = (out > 0.5).float()  # Threshold at 0.5
                                loss = criterion(out, target)  # Compute the loss.
                                loss_tot += loss.item()
                                
                                # Collect predictions and labels for performance score calculation
                                all_preds.append(pred.detach().cpu())  # Make sure to detach and move to CPU if necessary
                                all_preds_raw.append(out.detach().cpu())
                                all_labels.append(data.y.detach().cpu())
                            
                        # Concatenate all predictions and labels
                        all_preds = torch.cat(all_preds)
                        all_preds_raw = torch.cat(all_preds_raw)
                        all_labels = torch.cat(all_labels)
                        
                        # Calculate metrics.
                        metrics = calculate_metrics(all_labels, all_preds)
                        
                        # AUC score.
                        auc_score = roc_auc_score(all_labels, all_preds_raw)
                        
                        return loss_tot / len(loader), metrics, auc_score
                    
                    
                    # Blank list to hold test loss scores over current config and fold.
                    test_loss_scores = []
                    test_auc_scores = []
                    test_acc_scores = []
                    
                    # Iterates over 1000 epochs.
                    num_epochs = 1000
                    print('Training model.')
                    for epoch in tqdm(range(num_epochs)):
                        
                        train()
                        train_loss, train_metrics, train_auc = test(train_loader)
                        test_loss, test_metrics, test_auc = test(test_loader)
                        test_loss_scores.append(test_loss)
                        test_auc_scores.append(test_auc)
                        test_acc_scores.append(test_metrics['accuracy'])
                        #print(f'Training: Pre-trained: {pretrained_tag}, Fold: {n}, Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f} , Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                        with open(local_output_file,'a') as f:
                            f.write('\n'+str(epoch)+','+str(train_loss)+','+
                                    str(test_loss)+','+str(train_metrics['accuracy'])+','+
                                    str(test_metrics['sensitivity'])+','+
                                    str(test_metrics['specificity'])+','+
                                    str(test_metrics['accuracy'])+','+
                                    str(test_metrics['PPV'])+','+
                                    str(test_metrics['NPV'])+','+
                                    str(test_metrics['F1'])+','+
                                    str(test_metrics['MCC'])+','+
                                    str(train_auc)+','+str(test_auc))
                
                        # Saves model state, if higher overall accuracy is reached.
                        if len(test_acc_scores) > 1:
                            #print(test_auc_scores[-1], min(test_auc_scores[:-1]))
                            if test_acc_scores[-1] > max(test_acc_scores[:-1]):
                                #print('saving!')
                                params_file = f'{endpoint}/results/round_{r}/{local_config}_best_model.pth'
                                if os.path.exists(params_file):
                                    os.remove(params_file)
                                torch.save(model.state_dict(), params_file)

    
    
    
    
    
            
    
    
    
    
    
    
    
    
    
    
    
