import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN


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
    matrix = get_dist_matrix(G, name_to_id)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(matrix)
    clusters = {}
    cluster_ids = set(clustering.labels_).difference({-1})
    for cluster in cluster_ids:
        clusters[cluster] = []
    for i in range(len(clustering.labels_)):
        if clustering.labels_[i] == -1:
            continue
        clusters[clustering.labels_[i]].append(id_to_name[i])

    return clusters

def objective(G, cluster):
    print "for cluster ", cluster
    non_cluster_nodes = set(G.nodes).difference(set(cluster))
    print "cut capacity: ", get_cut_capacity(G, cluster, non_cluster_nodes)
    print "deficit: ", get_deficit(G, cluster)
    alpha = 1.0 / len(cluster)**2
    obj = alpha * get_cut_capacity(G, cluster, non_cluster_nodes) + get_deficit(G, cluster)
    print "obj is ", obj
    return obj

def cluster(G):
    #test_get_deficit()
    #test_cut_capacity()
    clusters = first_level_cluster(G)
    print clusters
    for cluster_id in clusters.keys():
        objective(G, clusters[cluster_id])
    #objective(G, clusters[2])
    #objective(G, clusters[3])
    #objective(G, clusters[2] + clusters[3])
    #objective(G, clusters[2] + clusters[3] + clusters[7])
    #objective(G, clusters[1])
    objective(G, clusters[0] + clusters[2])
    objective(G, ["C>T/TCC", "C>T/TCG", "C>T/CCC", "C>T/CCG"])
    print len(G.adj['C>T/TCC'])
    return clusters


if __name__ == "__main__":
    pass
