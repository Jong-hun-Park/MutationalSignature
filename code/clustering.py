import numpy as np
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

def get_deficit(cluster):
    pass

def ge_deficit(cluster):
    pass


def cluster(G):
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
    print clusters

    return clusters


if __name__ == "__main__":
    pass
