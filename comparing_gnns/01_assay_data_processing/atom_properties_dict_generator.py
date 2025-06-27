import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdchem
from mendeleev import element
import json


print("Reading data.")

# List of assays.
assays = ['ATG_PXRE_CIS', 'LTEA_HepaRG_CYP2C19', 'LTEA_HepaRG_UGT1A1', 
          'CCTE_Simmons_CellTiterGLO_HEK293T', 'CEETOX_H295R_OHPROG', 
          'CLD_CYP1A1_48hr', 'NVS_ENZ_hBACE']

# List (later set) that will hold SMILES.
SMILES = []

# Iterates through assay csv files, to find SMILES.
for assay in assays:
    
    # Creates dataframe.
    filename = assay+'.csv'
    df = pd.read_csv(filename)
    
    # Obtains SMILES.
    SMILES += list(df['SMILES'])

# Filters out duplicates.
SMILES = set(SMILES)


# Function for building dict of atom properties.
def build_atom_properties_dict(SMILES_list):
    
    # Variables for monitoring progress.
    counter = 0
    tot_len = len(SMILES_list)
    
    # Defines a set of all atoms occuring in the molecular data.
    atoms_set = set()
    
    # Iterates through SMILES.
    for s in SMILES_list:
        
        # Updates progress.
        counter += 1
        print("Finding atoms for properties dict:",counter,"/",tot_len)
        
        # Attempts to turn SMILES into RDKit molecule representation.
        try:
            mol = MolFromSmiles(s)
            # Finds atoms.
            atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atoms_set.update(atom_symbols)
        except AttributeError:
            continue
    
    # Builds an empty dictionary, to hold atom properties.
    properties_dict = {a:{} for a in atoms_set}
    
    # Variables for monitoring progress.
    counter = 0
    tot_len = len(properties_dict)
    
    # Iterates through atoms in dict.
    for a in properties_dict:
        
        # Updates progress.
        counter += 1
        print("Calculating atomic properties:",counter,"/",tot_len)
        
        # Finds some properties via RDKit.
        atom = rdchem.Atom(a)
        properties_dict[a]["atomic_num"] = atom.GetAtomicNum()
        properties_dict[a]["atomic_mass"] = atom.GetMass()
        
        # Finds others via Mendeleev.
        properties_dict[a]["electronegativity"] = element(a).electronegativity_pauling()
        properties_dict[a]["atomic_radius"] = element(a).atomic_radius
        #properties_dict[a]["atomic_vol"] = element(a).atomic_volume
        #properties_dict[a]["evaporation_heat"] = element(a).evaporation_heat
        #properties_dict[a]["cov_radius"] = element(a).covalent_radius_bragg
        properties_dict[a]["disp_coeff"] = element(a).c6
        properties_dict[a]["dipole_polarizability"] = element(a).dipole_polarizability
        properties_dict[a]["fusion_heat"] = element(a).fusion_heat
        #properties_dict[a]["gas_basicity"] = element(a).gas_basicity
        #properties_dict[a]["electron_affinity"] = element(a).electron_affinity
        properties_dict[a]["proton_affinity"] = element(a).proton_affinity
        #properties_dict[a]["specific_heat_capacity"] = element(a).specific_heat_capacity
        #properties_dict[a]["thermal_conductivity"] = element(a).thermal_conductivity
        #properties_dict[a]["vdw_radius"] = element(a).vdw_radius
    
    # Finds list of properties included.
    properties_list = [p for p in properties_dict[list(properties_dict.keys())[-1]]]
    
    # Returns the resulting dict and list, of atom properties.
    return properties_dict, properties_list


# Calls function, to build up atom properties dict and list.
atom_properties_dict, prop_list = build_atom_properties_dict(SMILES)

# Records atom properties dict to file.
with open('atom_properties_dict.json', 'w') as f:
    json.dump(atom_properties_dict, f)
