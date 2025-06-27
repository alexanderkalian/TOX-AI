import pandas as pd
from sklearn.model_selection import StratifiedKFold
import json


# Number of folds.
num_folds = 5


# List of assays.
assays = ['ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1', 
          'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG', 
          'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE']


# Variable for monitoring progress.
a = 0
num_assays = len(assays)

# Iterates through assays.
for assay in assays:
    
    # Updates progress.
    a += 1
    print('Processing assays:',a,'/',num_assays)
    
    # Defines filename.
    filename = assay+'.csv'
    
    # Builds dataframe, from csv file.
    df = pd.read_csv(filename)
    df.drop_duplicates()
    
    # Drops any rows with remaining duplicate SMILES or CAS numbers.
    df.drop_duplicates(subset=['SMILES'])
    df.drop_duplicates(subset=['CAS'])
    
    # Builds up dict of fold-organised data.
    folds_dict = {'folds':{n:{'train':{},'test':{}} for n in range(num_folds)}}
    
    # Randomly assigns train and test data, for chosen number of folds.
    skf = StratifiedKFold(n_splits=num_folds, random_state=1, shuffle=True)
    fold_splits = skf.split(df['SMILES'], df['Activity'])
    
    # Iterates through folds.
    for i, (train_index, test_index) in enumerate(fold_splits):
        
        # Builds smaller dict from train indices.
        train_dict = {str(df['SMILES'][n]):{'CAS':str(df['CAS'][n]), 
                                       'activity':int(df['Activity'][n]), 
                                       'AC50':float(df['AC50'][n])} for n in train_index}
        # Builds smaller dict from test indices.
        test_dict = {str(df['SMILES'][n]):{'CAS':str(df['CAS'][n]), 
                                           'activity':int(df['Activity'][n]), 
                                           'AC50':float(df['AC50'][n])} for n in test_index}
        
        # Assigns data to folds_dict
        folds_dict['folds'][i]['train'] = train_dict
        folds_dict['folds'][i]['test'] = test_dict
        
        '''
        # Checks for balance
        balance_monitor = [df['Activity'][n] for n in train_index]
        print(balance_monitor.count(0), balance_monitor.count(1))
        '''
        
        # Saves to json file.
        js = json.dumps(folds_dict)
        with open(assay+'_folds.json','w') as f:
            f.write(js)
    


        
        


