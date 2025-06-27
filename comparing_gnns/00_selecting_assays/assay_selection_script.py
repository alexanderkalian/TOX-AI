import pandas as pd
import numpy as np
import random


# Specify filename.
filename = 'assay-lists-2024-01-10.csv'
# Load into a pandas dataframe.
df = pd.read_csv(filename)

# Provides percentages of active chemicals, as a new column.
df['Perc Active'] = df['Multi Conc Active']*100/df['Multi Conc Total']

# Filters for assays with 40%-60% of the data being active.
df = df.loc[df['Perc Active'] >= 45].loc[
            df['Perc Active'] <= 55].loc[df['Multi Conc Total'] >= 200]
# Sort by number of datapoints.
df = df.sort_values('Multi Conc Total', ascending=False)
print(df[['List Acronym', 'Multi Conc Total']]) 


# Assays chosen as follows...
# ATG_PXRE_CIS, LTEA_HepaRG_CYP2C19, LTEA_HepaRG_UGT1A1, 
# CCTE_Simmons_CellTiterGLO_HEK293T, CEETOX_H295R_OHPROG, 
# CLD_CYP1A1_48hr, NVS_ENZ_hBACE


# Randomly chooses 6 assays, from this curated data.
chosen_indices = []
dist = np.linspace(0,82,7)
ranges = [[round(dist[i],0),round(dist[i+1],0)] for i in range(len(dist)-1)]
rand_pos = [int(a+round(random.random()*((b-a)-1),0)) for a,b in ranges]
print(rand_pos)


# Displays chosen assays and saves to csv.
print(df.iloc[rand_pos])
df.iloc[rand_pos].to_csv('selected_assays.csv',index=False)

