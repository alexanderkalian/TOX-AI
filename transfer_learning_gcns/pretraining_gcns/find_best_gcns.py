import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### Basic variables.

# Number of cross-validation folds used during model training
n_folds = 5

# List of curated datasets along with their types and human-readable names
# Format: [filename prefix, 'labels' or 'values', display name for plots]
files = [
    ['Ames_mutagenicity_SMILES_TU-Berlin_2009', 'labels', 'Ames Mutagenicity'],
    ['bbbp_intact', 'labels', 'Blood Brain\nBarrier Penetration'],
    ['ATG_PXRE_CIS', 'labels', 'ATG PXRE CIS\n(Activity)'],
    ['CEETOX_H295R_OHPROG', 'labels', 'CEETOX H295R\nOHPROG (Activity)'],
    ['ATG_PXRE_CIS', 'values', 'ATG PXRE CIS\n(AC50)'],
    ['CEETOX_H295R_OHPROG', 'values', 'CEETOX H295R\nOHPROG (AC50)'],
    ['P35968_inhibition', 'values', 'VEGFR-2 Binding\n(% Inhibition)'],
    ['P03372_EC50', 'values', 'ESR-1 Binding\n(EC50)'],
    ['P14416_Ki', 'values', 'DRD-2 Binding\n(Ki)']
]

# Extract the human-readable names for use in plotting
names = [f[2] for f in files]

# Path for output CSV file where summary of best models will be saved
output_file = 'analysis/best_gcns.csv'

# Open and initialize the output file with column headers
with open(output_file, 'w') as w:
    w.write('assay,dtype,best_test_score,test_loss,fold,params_file')

# Get the default matplotlib color cycle for plotting distinct bars
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
first_9_colors = colors[:9]  # Select only the first 9 colors

# Flag to conditionally trigger classification plot rendering
condition = True

# Loop through each dataset configuration and associated plot color
for (f, dtype, name), colour in zip(files, first_9_colors):
    
    print(f'{f} - {name}')  # Log current dataset name to stdout
    
    # Set appropriate test score column based on classification or regression
    if dtype == 'labels':
        test_score = 'test_overall_accuracy'
    elif dtype == 'values':
        test_score = 'test_R2_score'

        # If classification plot hasn't yet been shown, show it now
        if condition:
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor')  # Rotate x-axis labels
            plt.xlabel('Biomedical Assay Dataset')  # Label x-axis
            plt.ylabel('Best Overall Accuracy Score')  # Label y-axis
            plt.title('Classification Based GCN Pre-Training')  # Plot title
            plt.ylim([0, 1])  # Set y-axis limits for classification
            plt.minorticks_on()  # Enable minor ticks
            plt.grid(which='major', linestyle='-', linewidth='0.75', color='darkgray')  # Major grid
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid
            plt.gca().set_axisbelow(True)  # Send grid behind bars
            plt.savefig('analysis/classification_pretraining.png', dpi=500, bbox_inches='tight')  # Save figure
            plt.show()  # Display figure
            condition = False  # Ensure this only runs once

    best_scores = []  # List to track best scores across folds
    match_losses = []  # List to track matching test losses for best scores
    
    # Evaluate each fold for the current dataset
    for n in range(n_folds):
        filename = f'results/{f}_{dtype}_fold{n}.csv'  # Construct CSV file path
        
        df = pd.read_csv(filename)  # Load fold results as dataframe
        
        test_scores = list(df[test_score])  # Extract test score column
        local_best = max(test_scores)  # Find best score in this fold
        
        best_scores.append(local_best)  # Store best score
        
        test_loss = list(df['test_loss'])  # Extract test losses
        match_loss = test_loss[test_scores.index(local_best)]  # Match loss to best score
        match_losses.append(match_loss)  # Store matching loss
    
    best_score = max(best_scores)  # Determine the highest score across folds
    print(f'Best {test_score}: {best_score}')  # Print it
    
    match_loss2 = match_losses[best_scores.index(best_score)]  # Get the associated test loss
    print(f'Matching test_loss: {match_loss2}')  # Print it
    
    best_fold = best_scores.index(best_score)  # Find the fold index with the best score
    print(f'Best fold: {best_fold}')  # Print it
    
    avg_score = np.mean(best_scores)  # Average of best scores (for bar height)
    
    # Error bars for plotting (lower and upper bounds around mean)
    yerr_list = [[avg_score - min(best_scores)], [max(best_scores) - avg_score]]
    
    # Append result to output CSV
    with open(output_file, 'a') as w:
        w.write(f'\n{f},{dtype},{best_score},{match_loss2},{best_fold},{f}_{dtype}_fold{best_fold}.pth')

    # Plot bar for current dataset
    plt.bar(name, avg_score, color=colour, yerr=yerr_list, capsize=5)

# Finalize regression plot after looping through all datasets
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')  # Rotate x-axis labels
plt.ylim([0, 0.6])  # Adjust y-axis limits for R^2
plt.xlabel('Biomedical Assay Dataset')  # X-axis label
plt.ylabel('Best ${R}^{2}$ Score')  # Y-axis label with LaTeX for R²
plt.title('Regression Based GCN Pre-Training')  # Plot title
plt.minorticks_on()  # Enable minor ticks
plt.grid(which='major', linestyle='-', linewidth='0.75', color='darkgray')  # Major grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')  # Minor grid
plt.gca().set_axisbelow(True)  # Grid behind the bars
plt.savefig('analysis/regression_pretraining.png', dpi=500, bbox_inches='tight')  # Save figure
plt.show()  # Display plot
