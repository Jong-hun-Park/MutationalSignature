import pandas as pd
import numpy as np
from TableExtractor import extract_table
from graph_constructor import construct_graph
from clustering import cluster, predict, measure, get_individuals_from_file, get_dict_from_df, merge_dicts


# Extract Table
cancer_type = 'Melanoma'
#cancer_type = 'Kidney-RCC'
#cancer_type = 'Lung-SCC'
#cancer_type = 'Eso-AdenoCa'
input = '../data/WES_TCGA.96.csv'
# extract_table(input, cancer_type)
print("Matrix file is extracted")

# Construct Graph
jaccard_index_threshold = 0.7
#matrix_file = cancer_type + "_ignore_0" + ".csv"
matrix_file = cancer_type + ".csv"
G = construct_graph(matrix_file, jaccard_index_threshold)
print("Graph is constructed")
matrix_file = "../data/" + matrix_file
print("Number of edges in the graph: {}".format(len(G.edges)))
clusters = cluster(G)


# # to predict
# other_cancer_type = 'Lung-SCC'
# #other_cancer_type = 'Kidney-RCC'
# #other_cancer_type = 'Lymph-BNHL'
# skin_melanoma_matrix = get_dict_from_df(pd.read_csv(matrix_file))
# print("Melanoma individuals: ", len(skin_melanoma_matrix.keys()))
# lymph_matrix = get_dict_from_df(pd.read_csv('../data/' + other_cancer_type + '.csv'))
# mixed_matrix = merge_dicts(skin_melanoma_matrix, lymph_matrix)
#
# disease_predicted_individuals = predict(clusters, mixed_matrix)
# true_disease_individuals = skin_melanoma_matrix.keys()
#
# all_individuals = mixed_matrix
# measure(all_individuals, disease_predicted_individuals, true_disease_individuals)
