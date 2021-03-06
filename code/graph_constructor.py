import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np


def get_jaccard_index_threshold_from_kde(input_file, target_percentile=0.97):
    data_folder = "../data/"
    matrix_file = data_folder + input_file
    mutation_dict = dict()
    with open(matrix_file, "r") as f:
        header = f.readline()
        for line in f:
            split = line.strip().split(",")
            mutation_type = split[0]
            trinucleotide = split[1]
            mutation_dict[mutation_type + "/" + trinucleotide] = split[2:]

    jaccard_index_list = []
    for mutation_type1, list1 in mutation_dict.items():
        for mutation_type2, list2 in mutation_dict.items():
            set1 = set([j for j in range(len(list1)) if list1[j] == '1'])
            set2 = set([j for j in range(len(list2)) if list2[j] == '1'])
            jaccard_index_list.append(compute_jaccard_index(set1, set2))

    plt.figure()
    plt.title("Jaccard Index Distribution of {} Sample".format(input_file.split(".")[0]))
    plt.xlabel("Jaccard Index")
    plt.ylabel("Density")

    p = sns.kdeplot(jaccard_index_list, cumulative=False, shade=True)
    xmin, xmax, ymin, ymax = plt.axis()
    x, y = p.get_lines()[0].get_data()
    # print(x, y)
    # care with the order, it is first y
    # initial fills a 0 so the result has same length than x
    cdf = scipy.integrate.cumtrapz(y, x, initial=0)
    target_index = np.abs(cdf - target_percentile).argmin()

    x_point = x[target_index]
    y_point = y[target_index]
    plt.vlines(x_point, 0, ymax, colors="b", linestyles='--')
    # plt.show()

    return x_point


def compute_jaccard_index(set1, set2):
    intersect = set1 & set2
    jaccard_index = float(len(intersect)) / len(set1.union(set2))
    return jaccard_index


def construct_graph(input_file, jaccard_index_threshold=0.3, weighted=True, out_graph=False):
    # Loading Matrix file
    data_folder = "../data/"
    matrix_file = data_folder + input_file
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
            if mutation_type1 == mutation_type2:
                continue
            set1 = set([j for j in range(len(list1)) if list1[j] == '1'])
            set2 = set([j for j in range(len(list2)) if list2[j] == '1'])
            jaccard_index = compute_jaccard_index(set1, set2)
            if jaccard_index > jaccard_index_threshold:
                if weighted:
                    G.add_edge(mutation_type1, mutation_type2, weight=jaccard_index)
                else:
                    G.add_edge(mutation_type1, mutation_type2)

    # Output graph as adjacent list
    if out_graph:
        outfile = data_folder + matrix_file.split("/")[-1].split(".")[0] + "_graph_" + str(jaccard_index_threshold) + ".csv"
        with open(outfile, "w") as f:
            for node in G.nodes():
                f.write(node)
                f.write(":")  # adjacent list delimiter
                f.write(','.join(G[node].keys()))
                f.write("\n")

    return G


def draw_graph(G, jaccard_index_threshold=None):
    import matplotlib.pyplot as plt
    plt.figure()
    nx.draw_networkx(G)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_color='b')
    # nx.draw_networkx_edges(G, pos, width=1.0)
    if jaccard_index_threshold is not None:
        plt.title("Jaccard index threshold {}".format(jaccard_index_threshold))
    plt.savefig("../figures/" + "graph_jaccard_index_" + str(jaccard_index_threshold) + ".png")


def output_graph(graph, outfile_name="graph_edge_node.csv"):
    with open(outfile_name, "w") as f:
        for edge in graph.edges_iter():
            f.write(str(edge[0]) + "," + str(edge[1]) + "\n")
            if edge[0] != edge[1]:
                f.write(str(edge[0]) + "," + str(edge[1]) + "\n")

        for node in graph.nodes_iter():
            f.write(str(node) + "\n")


if __name__ == "__main__":
    jaccard_index_threshold = 0.4
    matrix_file = "Lung-SCC.csv"
    # matrix_file = "Skin-Melanoma.csv"
    # matrix_file = "Pheochromocytoma.csv"
    # matrix_file = "Cervix-CA.csv"
    # matrix_file = "Transitional-cell-carcinoma.csv"
    # matrix_file = "Eso-AdenoCa.csv"
    # matrix_file = "ColoRect-AdenoCa.csv"

    jaccard_index_threshold = get_jaccard_index_threshold_from_kde(matrix_file, 0.90)
    graph = construct_graph(matrix_file, jaccard_index_threshold, weighted=True)
    print([(u, v, edata['weight']) for u, v, edata in graph.edges(data=True) if 'weight' in edata])
    # draw_graph(graph, jaccard_index_threshold)
