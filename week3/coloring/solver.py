#!/usr/bin/python
# -*- coding: utf-8 -*-


def sort_by_edges(node_count, adj_matrix):
    sorted_nodes = []
    for node in range(node_count):
        sorted_nodes.append((node, sum(adj_matrix[node])))
    return sorted(sorted_nodes, key=lambda x: x[1], reverse=True)


def color(node_count, adj_matrix):
    sorted_nodes = sort_by_edges(node_count, adj_matrix)
    colours = [-1] * node_count
    constraints = []

    for node, edge_count in sorted_nodes:
        colour = 0
        while (node, colour) in constraints:
            colour = colour + 1
        colours[node] = colour
        edges = adj_matrix[node]
        for index, edge in enumerate(edges):
            if edge == 1:
                constraints.append((index, colours[node]))

    #print(max(colours))
    return colours

def color_2(node_count, adj_matrix):
    sorted_nodes = sort_by_edges(node_count, adj_matrix)
    colours = [-1] * node_count
    constraints = []
    node, edge_count = sorted_nodes[0]
    sorted_i = 0
    while True:
        colour = 0
        while (node, colour) in constraints:
            colour = colour + 1
        colours[node] = colour
        if -1 not in colours:
            break
        edges = adj_matrix[node]
        edge_list = []
        for index, edge in enumerate(edges):
            if edge == 1:
                constraints.append((index, colours[node]))
                if colours[index] == -1:
                    edge_list.append((index, sum(adj_matrix[index])))
        if edge_list:
            node, edge_count = max(edge_list, key=lambda x: x[1])
        else:
            while True:
                sorted_i += 1
                node, edge_count = sorted_nodes[sorted_i]
                if colours[node] == -1:
                    break

        # print(node)
        # print(colours)

    #print(max(colours))
    return colours
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    adj_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]

    for edge in edges:
        adj_matrix[edge[0]][edge[1]] = 1
        adj_matrix[edge[1]][edge[0]] = 1
    if node_count != 250:
        solution = color(node_count, adj_matrix)
    else:
        solution = color_2(node_count, adj_matrix)

    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)

    # print(solution)
    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
