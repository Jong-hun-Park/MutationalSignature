import pandas as pd
import numpy as np
from TableExtractor import extract_table
from graph_constructor import construct_graph, get_jaccard_index_threshold_from_kde
from clustering import cluster, predict, measure, get_individuals_from_file, get_dict_from_df, merge_dicts


# Extract Table
cancer_type = 'Melanoma'
#cancer_type = 'Lung-SCC'
#cancer_type = 'Kidney-RCC'
#cancer_type = 'Lung-SCC'
#cancer_type = 'Eso-AdenoCa'
input = '../data/WES_TCGA.96.csv'
#extract_table(input, cancer_type)
print("Matrix file is extracted")

# Construct Graph
#matrix_file = cancer_type + "_ignore_0" + ".csv"
matrix_file = cancer_type + "_0_1_percentile.csv" #+ ".csv"
#matrix_file = cancer_type + ".csv"
jaccard_index_threshold = get_jaccard_index_threshold_from_kde(matrix_file, 0.97)
G = construct_graph(matrix_file, jaccard_index_threshold)
print("Graph is constructed")
print("Number of edges in the graph: {}".format(len(G.edges)))
print("clustering")
clusters = cluster(G, 'dbscan')
clusters = cluster(G, 'greedy')
#clusters = cluster(G, 'random')


# To predict
matrix_file = "../data/" + matrix_file
cancer_matrix = get_dict_from_df(pd.read_csv(matrix_file))
mixed_matrix = get_dict_from_df(pd.read_csv("../data/sara.csv"))

true_disease_individuals = cancer_matrix.keys()
all_individuals = mixed_matrix.keys()
print("all individuals: ", len(mixed_matrix.keys()))

print("on same dataset:")
for threshold in np.linspace(0, 1, 10):
    disease_predicted_individuals = predict(clusters, cancer_matrix, threshold)
    measure(cancer_matrix.keys(), disease_predicted_individuals, true_disease_individuals)
print("on mixed dataset:")
for threshold in np.linspace(0, 1, 10):
    disease_predicted_individuals = predict(clusters, mixed_matrix, threshold)
    measure(all_individuals, disease_predicted_individuals, true_disease_individuals)
