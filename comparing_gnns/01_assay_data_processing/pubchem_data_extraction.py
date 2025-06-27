import pandas as pd
import pubchempy as pcp
import sys


# List of assays.
assays = ['ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1', 
          'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG', 
          'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE']


# For monitoring progress.
i1 = 0
tot_len1 = len(assays)


# Iterates through assays.
for assay in assays:
    
    # Updates progress variable.
    i1 += 1
    
    # Creates file to contain data.
    output_file = assay+'.csv'
    with open(output_file, 'w') as f:
        f.write('SMILES,CAS,Activity,AC50')
    
    # Defines filename, where assay data can be found.
    filename = '../00_selecting_assays/assay_data/Assay List '+assay+'-2024-01-14.csv'
    
    # Creates dataframe for this data.
    df = pd.read_csv(filename)
    df['Activity'] = [1 if a == 'Active' else 0 for a in df['HIT CALL']]
    
    # Checks for any hit call that does not fall into either "Active" or "Inactive".
    if len([a for a in df['HIT CALL'] if a != 'Active' and a != 'Inactive']) > 0:
        print('Exception for HIT CALL found:')
        print([a for a in df['HIT CALL'] if a != 'Active' and a != 'Inactive'])
        sys.exit()
    
    # Metrics for progress monitoring.
    i2 = 0
    tot_len2 = len(list(df['CASRN']))
    
    # Iterates through CAS numbers.
    for cas, activity, AC50 in zip(df['CASRN'], df['Activity'], df['AC50']):
        
        # Updates progress.
        i2 += 1
        print('Assay:',i1,'/',tot_len1,' Processing:',i2,'/',tot_len2)
        
        # Attempts to find SMILES via PubChem API.
        try:
            
            # Queries PubChem API.
            query = pcp.get_properties(['CanonicalSMILES','IsomericSMILES'], 
                                       cas, 'name')
            
            # Prioritises canonical SMILES over isomeric SMILES, but will accept either.
            if 'CanonicalSMILES' in query[0]:
                SMILES = query[0]['CanonicalSMILES']
            elif 'IsomericSMILES' in query[0]:
                SMILES = query[0]['IsomericSMILES']
                
            # Updates output file.
            with open(output_file, 'a') as f:
                f.write('\n'+SMILES+','+str(cas)+','+str(activity)+','+str(AC50))
                
        except Exception:
            print('Problematic CASRN value:',cas)
            continue

