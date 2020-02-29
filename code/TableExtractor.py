import numpy as np
import pandas as pd


def extract_table(input_file, cancer_type):
    Cancer_type = cancer_type
    data = pd.read_csv(input_file)
    columns = data.columns
    selected_col = ['Mutation type', 'Trinucleotide']

    for c in columns:
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


if __name__ == "__main__":
    cancer_type = 'Skin-Melanoma'
    input = '../data/WES_TCGA.96.csv'
    extract_table(input, cancer_type)