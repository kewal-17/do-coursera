#!/usr/bin/python
# -*- coding: utf-8 -*-


def sort_by_edges(node_count, adj_matrix):
    sorted_nodes = []
    for node in range(node_count):
        sorted_nodes.append((node, sum(adj_matrix[node])))
    return sorted(sorted_nodes, key=lambda x: x[1], reverse=True)


def color(node_count, edge_count, adj_matrix):
    sorted_nodes = sort_by_edges(node_count, adj_matrix)
    colours = [-1] * node_count
    constraints = []
    for node in range(node_count):
        colour = 0
        while (node, colour) in constraints:
            colour = colour + 1
        colours[node] = colour
        edges = adj_matrix[node]
        for index, edge in enumerate(edges):
            if edge == 1:
                constraints.append((index, colours[node]))
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

    solution = color(node_count, edge_count, adj_matrix)

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
