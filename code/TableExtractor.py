import numpy as np
import pandas as pd

Cancer_type = 'Ovary-AdenoCa'
data = pd.read_csv('../data/WES_TCGA.96.csv')
columns = data.columns
selected_col = ['Mutation type', 'Trinucleotide']

for c in columns:
    # if c.startswith(Cancer_type):
    if Cancer_type in c:
        selected_col.append(c)
sub_data = data[selected_col]
sub_data['mean'] = sub_data.mean(axis=1)
selected_col = selected_col[2:]
for c in selected_col:
    sub_data[c] = sub_data[c] >= sub_data['mean']
sub_data = sub_data.drop('mean', 1)
sub_data[list(sub_data.columns)[2:]] = sub_data[list(sub_data.columns)[2:]].astype(int)
sub_data.to_csv('../data/' + Cancer_type + '.csv', index=False)


selected_col = ['Mutation type', 'Trinucleotide']
for c in columns:
    if Cancer_type in c:
        selected_col.append(c)
sub_data = data[selected_col]
sub_data = sub_data.replace(0,np.NaN)
sub_data['mean'] = sub_data.mean(axis=1)
selected_col = selected_col[2:]
for c in selected_col:
    sub_data[c] = sub_data[c] >= sub_data['mean']
sub_data = sub_data.drop('mean', 1)
sub_data[list(sub_data.columns)[2:]] = sub_data[list(sub_data.columns)[2:]].astype(int)
sub_data.to_csv('../data/' + Cancer_type + '_ignore_0.csv', index=False)