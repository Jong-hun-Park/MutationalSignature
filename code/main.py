from TableExtractor import extract_table
from graph_constructor import construct_graph


# Extract Table
cancer_type = 'Skin-Melanoma'
input = '../data/WES_TCGA.96.csv'
extract_table(input, cancer_type)
print("Matrix file is extracted")

# Construct Graph
jaccard_index_threshold = 0.3
matrix_file = cancer_type + "_ignore_0" + ".csv"
construct_graph(matrix_file, jaccard_index_threshold)
print("Graph is constructed")