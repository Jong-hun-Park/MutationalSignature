import networkx as nx


def compute_jaccard_index(set1, set2):
    intersect = set1 & set2
    jaccard_index = float(len(intersect)) / (len(set1) + len(set2))
    return jaccard_index


def construct_graph(jaccard_index_treshold=0.3):
    # Loading Matrix file
    data_folder = "../data/"
    matrix_file = data_folder + "Skin-Melanoma_ignore_0.csv"
    mutation_dict = dict()
    with open(matrix_file, "r") as f:
        header = f.readline()
        for line in f:
            split = line.strip().split(",")
            mutation_type = split[0]
            trinucleotide = split[1]
            mutation_dict[mutation_type + "/" + trinucleotide] = split[2:]

    # Construct graph
    G = nx.Graph()
    for mutation_type1, list1 in mutation_dict.items():
        for mutation_type2, list2 in mutation_dict.items():
            set1 = set([j for j in range(len(list1)) if list1[j] == '1'])
            set2 = set([j for j in range(len(list2)) if list2[j] == '1'])
            jaccard_index = compute_jaccard_index(set1, set2)
            if jaccard_index > jaccard_index_treshold:
                G.add_edge(mutation_type1, mutation_type2)

    # Output graph as adjacent list
    outfile = data_folder + matrix_file.split("/")[-1].split(".")[0] + "_graph" + ".csv"
    with open(outfile, "w") as f:
        for node in G.nodes_iter():
            f.write(node)
            f.write(":")  # adjacent list delimiter
            f.write(','.join(G[node].keys()))
            f.write("\n")


if __name__ == "__main__":
    jaccard_index_threshold = 0.3
    construct_graph(jaccard_index_threshold)