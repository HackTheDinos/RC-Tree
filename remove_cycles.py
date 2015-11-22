edge_distances = {
    ('a', 'b'): 5,
    ('b', 'c'): 4,
    ('b', 'd'): 5,
    ('b', 'e'): 5,
    ('e', 'f'): 4,
    ('j', 'e'): 4,
    ('e', 'i'): 4,
    ('g', 'd'): 4,
    ('h', 'd'): 4,
    ('b', 'f'): 6
}

def construct_graph(edges):
    points = set([point for edge in edges for point in edge])
    point_neighbors = {}
    for point in points:
        point_neighbors[point] = set()
        for edge in edges:
            if point in edge:
                point_neighbors[point].update(list(edge))
        point_neighbors[point].discard(point)

    return point_neighbors

neighbors = construct_graph(edge_distances.keys())

def edges_from_neighbors(neighbors):
    edges = []
    for k, v in neighbors.iteritems():
        for w in v:
            edges.append((k, w))
    return edges

def remove_cycles(edges, edge_distances, root):
    def get_distance(edge):
        return edge_distances[edge]
    edges.sort(lambda x,y: get_distance(y) - get_distance(x))
    visited = set()
    queue = [(root, None)]
    while queue:
        node, parent = queue.pop()
        # print "======================"
        # print "q: {}".format(queue)
        # print "v: {}".format(visited)
        # print "edges: {}".format(edges)
        # print "- looking at node: {} -".format(node)
        # print "with parent {}".format(parent)
        for edge in edges:
            if parent in edge:
                # print "parent edge! skipping: {}".format(edge)
                continue
            # print "checking edge: {}".format(edge)
            if node in edge:
                if edge[0] in visited or edge[1] in visited:
                    # print "found a cycle, removing {}".format(edge)
                    # print "=============================="
                    # print "=============================="
                    # print "=============================="
                    edges.remove(edge)
                    return remove_cycles(edges, edge_distances, root)
                # print "found a neighbor"
                for nextnode in edge:
                    if nextnode != node:
                        # print "...adding {}".format((nextnode, node))
                        queue.append((nextnode, node))
                        last_edge = edge
        # print "adding to visited: {}".format(node)
        visited.add(node)


    return edges

remove_cycles(edge_distances.keys(), edge_distances, 'b')
