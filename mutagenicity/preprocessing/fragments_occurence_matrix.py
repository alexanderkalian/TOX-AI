from pysmiles import read_smiles
import networkx as nx
import itertools
import pandas as pd
from tqdm import tqdm
import logging


# Important variables.
min_frag_len = 3 # Specify a minimum fragment length (atoms, excl. H)
max_frag_len = 9 # Specify a maximum fragment length (atoms, excl. H)

# Specify file names.
source_file = '../data/processed/standardised_mutagenicity_dataset.csv' # Specify data file
output_file = '../data/processed/fragments_matrix.csv' # Specify output file

# Reads data to obtain structures.
df = pd.read_csv(source_file)
structures = df['smiles'].tolist()
labels = df['mutagenicity'].tolist()


# Suppress disruptive pysmiles logging warnings.
logging.getLogger('pysmiles').setLevel(logging.ERROR)


# Function for outputting k fragments, for a given molecular SMILES.
def girvan_newman_fragments(smiles,k):
    
    # Turns smiles into a Graph object.
    mol_graph = read_smiles(smiles)
    
    # Uses Girvan-Newman algorithm in order to find communities in graph.
    comp = nx.algorithms.community.girvan_newman(mol_graph)
    
    # Compiles into a useable list of nodes of communities.
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    communities = [list(c) for c in list(limited)[-1:][0]]
    
    # Compiles list of community nodes into a list of fragment Graph objects.
    fragment_graphs = []
    sufficient = True
    for community in communities:
        # Ensures that fragments are len <= 15.
        if len(community) > max_frag_len:
            sufficient = False
        # Ensures that only fragments of len >= 3 are recorded for later use.
        if len(community) >= min_frag_len:
            subgraph = mol_graph.subgraph(community)
            fragment_graphs.append(subgraph)
    
    return fragment_graphs, sufficient


# Iterates through the structures and runs the function, in order to obtain fragments.

# Blank list to contain fragments.
all_fragments = []

# Blank list for problematic molecules.
problematic_molecules = []

# Iterates through smiles structures.
for structure in tqdm(structures, desc='Calculating fragments'):
    # Starts with k value of 3.
    k_val = 3
    # Obtains initial results.
    try:
        fragments, success = girvan_newman_fragments(structure,k=k_val)
        # If the fragments are not sufficiently small, then repeats, with increased k values, until achieved.
        while not success:
            k_val += 1
            fragments, success = girvan_newman_fragments(structure,k=k_val)
        # Appends newly found fragments to all_fragments list
        all_fragments += fragments
    except (NotImplementedError, ValueError) as e:
        #print(f'Error found - ignoring molecule:\n{e}')
        problematic_molecules.append(structure)


# Function for checking graph isomorphism (both via matching nodes and edges).
def are_isomorphic(g1, g2, subgraph=False):
    gm = nx.isomorphism.GraphMatcher(
        g1,
        g2,
        node_match=lambda n1, n2: n1.get('element') == n2.get('element'),
        edge_match=lambda e1, e2: e1.get('order') == e2.get('order'),
    )
    if subgraph:
        return gm.subgraph_is_isomorphic()
    else:
        return gm.is_isomorphic()

# Function for finding only unique fragments.
def keep_unique_fragments(fragments):
    # List to hold unique graphs.
    unique_graphs = []
    
    # Iterate through fragment graphs.
    for g in tqdm(all_fragments, desc='Graph isomorphism checks'):
        
        # Only append graph to unique graphs list if no isomorphic graphs are found.
        if not any(are_isomorphic(g, ug) for ug in unique_graphs):
            unique_graphs.append(g)

    return unique_graphs

# Find unique graphs, through graph isomorphism checks.
unique_fragments = keep_unique_fragments(all_fragments)

print(f'Unique fragments: {len(unique_fragments)}/{len(all_fragments)}')


# Blank list to hold fragment occurences (1) - or inoccurences (0).
fragments_matrix = []

# Finds non-problematic structures.
structures_refined = [s for s in structures if s not in problematic_molecules]
labels_refined = [l for s,l in zip(structures, labels) if s not in problematic_molecules]

# Iterate through smiles and find fragment occurences.
for structure in tqdm(structures_refined, desc='Finding fragment occurences'):
    
    # Generate molecular graph.
    mol_graph = read_smiles(structure)
    
    # Initialises row of matrix.
    row = []
    
    # Iterate through fragment graphs and check for isomorphism.
    for frag_graph in unique_fragments:
        
        # Finds occurence value and appends to row.
        occurence = int(are_isomorphic(mol_graph, frag_graph, subgraph=True))
        row.append(occurence)
    
    # Appends row to matrix.
    fragments_matrix.append(row)


print('Done - saving to file.')

# Configures fragment names.
fragment_names = [f'fragment{n}' for n in range(len(unique_fragments))]

# Creates dataframe.
results_df = pd.DataFrame(fragments_matrix, columns=fragment_names)
results_df.insert(0, 'smiles', structures_refined)  # Add SMILES as first column.
results_df['mutagenicity'] = labels_refined

# Saves to file.
results_df.to_csv(output_file, index=False)
