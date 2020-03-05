import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering
from graph_constructor import compute_jaccard_index

class DisjointSet:
    def __init__(self, vertices, parent):
        self.vertices = vertices
        self.parent = parent

    def find(self, item):
        if self.parent[item] == item:
            return item
        else:
            return self.find(self.parent[item])

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        self.parent[root1] = root2

def get_dist_matrix(G, name_to_id):
    n = len(G.nodes)
    matrix = [[1 for i in range(n)] for j in range(n)]
    for node in G.nodes:
        node_id = name_to_id[node]
        for neighbor in G.adj[node]:
            neighbor_id = name_to_id[neighbor]
            matrix[node_id][neighbor_id] = 0
            matrix[neighbor_id][node_id] = 0
    return matrix

def get_similarity_matrix(G, name_to_id):
    matrix = get_dist_matrix(G, name_to_id)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = 1 - matrix[i][j]
    return matrix

def get_node_id_map(G):
    name_to_id_map = {}
    id_to_name_map = {}

    for i in range(len(G.nodes)):
        node = list(G.nodes)[i]
        name_to_id_map[node] = i
        id_to_name_map[i] = node

    return name_to_id_map, id_to_name_map

def get_deficit(G, cluster_nodes):
    #print "computing deficit for: ", cluster_nodes
    sum_degrees = 0
    for node_1 in cluster_nodes:
        #print "neighbors:", G.adj[node_1]
        for node_2 in cluster_nodes:
            if node_1 == node_2:
                continue
            if node_2 in G.adj[node_1]:
                sum_degrees += 1
    edges_in_cluster = sum_degrees / 2.0
    cluster_n = len(cluster_nodes)
    #print "edges_in_cluster", edges_in_cluster
    return int(cluster_n * (cluster_n - 1) / 2.0 - edges_in_cluster)

def get_cut_capacity(G, cluster_nodes, non_cluster_nodes):
    cut_edges = 0
    for inside_node in cluster_nodes:
        for neighbor in G.adj[inside_node]:
            if neighbor in non_cluster_nodes:
                cut_edges += 1
    return cut_edges

def test_get_deficit():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("c", "d")
    print("Expected deficit: 4, actual: {}".format(get_deficit(G, ["a", "b", "c", "d"])))

def test_cut_capacity():
    G = nx.Graph()
    G.add_edge("a", "b")
    G.add_edge("a", "c")
    G.add_edge("c", "d")
    G.add_edge("b", "d")
    G.add_edge("a", "d")
    print("Expected cut capacity: 3, actual: {}".format(get_cut_capacity(G, ["a", "b"])))

def first_level_cluster(G):
    name_to_id, id_to_name = get_node_id_map(G)
    dist_matrix = get_dist_matrix(G, name_to_id)
    sim_matrix = get_similarity_matrix(G, name_to_id)
    clustering = DBSCAN(eps=0.5, min_samples=1).fit(dist_matrix)
    #clustering = AffinityPropagation(affinity='precomputed').fit(sim_matrix)
    #clusters = AgglomerativeClustering(linkage="average", connectivity=sim_matrix, affinity="precomputed").fit(dist_matrix)
    clusters = {}
    cluster_ids = set(clustering.labels_).difference({-1})
    for cluster in cluster_ids:
        clusters[cluster] = []
    for i in range(len(clustering.labels_)):
        if clustering.labels_[i] == -1:
            continue
        clusters[clustering.labels_[i]].append(id_to_name[i])

    return clusters

def objective(G, cluster, verbose=True):
    if verbose:
        print "for cluster ", cluster
    non_cluster_nodes = set(G.nodes).difference(set(cluster))
    if verbose:
        print "cut capacity: ", get_cut_capacity(G, cluster, non_cluster_nodes)
        print "deficit: ", get_deficit(G, cluster)
    alpha = 1.0 / len(cluster)**2
    obj = alpha * get_cut_capacity(G, cluster, non_cluster_nodes) + get_deficit(G, cluster)
    if verbose:
        print "obj is ", obj
    return obj

def get_neighboring_clusters(G, clusters):
    neighboring_clusters = {}
    for cluster_id_1 in clusters.keys():
        neighboring_clusters[cluster_id_1] = set()
        for cluster_id_2 in clusters.keys():
            if cluster_id_1 == cluster_id_2:
                continue
            for node_1 in clusters[cluster_id_1]:
                for node_2 in clusters[cluster_id_2]:
                    if node_2 in G.adj[node_1]:
                        neighboring_clusters[cluster_id_1].add(cluster_id_2)

    return neighboring_clusters

def get_merge_clusters(G, clusters):
    parent = {}
    for cluster_id in clusters:
        parent[cluster_id] = cluster_id
    groups = DisjointSet(clusters.keys(), parent)
    neighboring_clusters = get_neighboring_clusters(G, clusters)
    #print neighboring_clusters
    for cluster_id_1 in neighboring_clusters:
        for cluster_id_2 in neighboring_clusters[cluster_id_1]:
            # merge clusters if they it decreases the objective for both
            if cluster_id_1 != cluster_id_2 and objective(G, clusters[cluster_id_1] + clusters[cluster_id_2], False)\
                    < min(objective(G, clusters[cluster_id_1], False), objective(G, clusters[cluster_id_2], False)):
                groups.union(cluster_id_1, cluster_id_2)
                neighboring_clusters[cluster_id_1] = neighboring_clusters[cluster_id_1].union(neighboring_clusters[cluster_id_2])
                neighboring_clusters[cluster_id_2] = neighboring_clusters[cluster_id_1]
                clusters[cluster_id_1] = list(set(clusters[cluster_id_1] + clusters[cluster_id_2]))
                clusters[cluster_id_2] = clusters[cluster_id_1]
                print "merging clusters ", cluster_id_1, " and ", cluster_id_2
                #print clusters
                #print neighboring_clusters
    merged_clusters = {}
    for node in groups.vertices:
        parent = groups.find(node)
        if parent in merged_clusters:
            continue
        merged_clusters[parent] = clusters[parent]
    return merged_clusters, groups


def cluster(G):
    #test_get_deficit()
    #test_cut_capacity()
    clusters = first_level_cluster(G)
    print "Initial clusters: ", clusters
    for cluster_id in clusters.keys():
        objective(G, clusters[cluster_id])
    clusters, groups = get_merge_clusters(G, clusters)
    print "merged_clusters: ", clusters
    for cluster_id in clusters.keys():
        objective(G, clusters[cluster_id])
    #objective(G, clusters[2])
    #objective(G, clusters[3])
    #objective(G, clusters[2] + clusters[3])
    #objective(G, clusters[2] + clusters[3] + clusters[7])
    #objective(G, clusters[1])
    #objective(G, clusters[0] + clusters[2])
    #objective(G, ["C>T/TCC", "C>T/TCG", "C>T/CCC", "C>T/CCG"])
    return clusters

def get_individuals_from_file(file_name):
    mutation_matrix = pd.read_csv(file_name)
    return mutation_matrix.columns[2:]

def get_dict_from_df(df):
    dict = {"Mutation type": [], "Trinucleotide":[]}
    for ind in df:
        dict[ind] = []
        for j in df[ind]:
            dict[ind].append(j)
    for t in df["Mutation type"]:
        dict["Mutation type"].append(t)
    for t in df["Trinucleotide"]:
        dict["Trinucleotide"].append(t)

    return dict

def merge_dicts(dict_1, dict_2):
    for ind in dict_1:
        dict_2[ind] = dict_1[ind]
    return dict_2

def predict(clusters, matrix, threshold=0.3):
    individuals_with_signiture = []
    individuals = matrix.keys()
    print("len mixed individuals: ", len(individuals))
    for cluster in clusters.values():
        print "predicting for cluster: ", cluster
        #for individual_index in range(2, len(matrix.columns)):
        for individual in individuals:
            mutations_in_individual = set()
            for i in range(len(matrix[individual])):
                if matrix[individual][i] == 1:
                    mutations_in_individual.add(matrix["Mutation type"][i] + "/" + matrix["Trinucleotide"][i])
            shared_mutations = float(len(mutations_in_individual.intersection(set(cluster)))) / float(len(cluster))
            if shared_mutations >= threshold:
                individuals_with_signiture.append(individual)
    return list(set(individuals_with_signiture))

def measure(all_individuals, predicted_disease, true_disease):
    #print "predicted_disease: ", predicted_disease
    #print "true_disease: ", true_disease
    #print "all_individuals: ", all_individuals
    true_positive = len([ind for ind in predicted_disease if ind in true_disease])
    false_positive = len([ind for ind in predicted_disease if ind not in true_disease])
    false_negatives = len([ind for ind in true_disease if ind not in predicted_disease])
    precision = float(true_positive) / (true_positive + false_positive + 0.01)
    recall = float(true_positive) / (true_positive + false_negatives + 0.01)
    print("TP: {}, FP: {}, FN: {}, precision: {}, recall: {}".format(true_positive, false_positive, false_negatives, precision, recall))

if __name__ == "__main__":
    pass
