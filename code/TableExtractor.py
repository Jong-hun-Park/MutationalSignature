import numpy as np
import pandas as pd

Cancer_type = 'Skin-Melanoma'

data = pd.read_csv('../data/WES_TCGA.96.csv')

columns = data.columns

selected_col = ['Mutation type', 'Trinucleotide']

for c in columns:
    if Cancer_type in c :
        selected_col.append(c)

sub_data = data[selected_col]

sub_data.to_csv('../data/'+Cancer_type+'.csv',index=False)