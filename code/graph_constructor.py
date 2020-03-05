import networkx as nx


def compute_jaccard_index(set1, set2):
    intersect = set1 & set2
    jaccard_index = float(len(intersect)) / (len(set1) + len(set2))
    return jaccard_index


def construct_graph(input_file, jaccard_index_threshold=0.3):
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
            set1 = set([j for j in range(len(list1)) if list1[j] == '1'])
            set2 = set([j for j in range(len(list2)) if list2[j] == '1'])
            jaccard_index = compute_jaccard_index(set1, set2)
            if jaccard_index > jaccard_index_threshold:
                G.add_edge(mutation_type1, mutation_type2)

    # Output graph as adjacent list
    outfile = data_folder + matrix_file.split("/")[-1].split(".")[0] + "_graph_" + str(jaccard_index_threshold) + ".csv"
    with open(outfile, "w") as f:
        for node in G.nodes_iter():
            f.write(node)
            f.write(":")  # adjacent list delimiter
            f.write(','.join(G[node].keys()))
            f.write("\n")

    return G


def draw_graph(G, jaccard_index_threshold):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='b', node_size=50)
    nx.draw_networkx_edges(G, pos, width=1.0)
    plt.title("Jaccard index threshold {}".format(jaccard_index_threshold))
    plt.savefig("../figures/" + "graph_jaccard_index_" + str(jaccard_index_threshold) + ".png", size=(17,10))


def output_graph(graph, outfile_name="graph_edge_node.csv"):
    with open(outfile_name, "w") as f:
        for edge in graph.edges_iter():
            if edge[0] != edge[1]:
                f.write(str(edge[0]) + "," + str(edge[1]) + "\n")

        for node in graph.nodes_iter():
            f.write(str(node) + "\n")


if __name__ == "__main__":
    jaccard_index_threshold = 0.4
    matrix_file = "Skin-Melanoma.csv"
    graph = construct_graph(matrix_file, jaccard_index_threshold)
    # output_graph(graph)
    draw_graph(graph, jaccard_index_threshold)
