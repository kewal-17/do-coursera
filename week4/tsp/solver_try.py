#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
from collections import namedtuple
import numpy as np

Point = namedtuple("Point", ['x', 'y'])
NewPoint = namedtuple("NewPoint", ["index", "x", "y"])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def tsp_travel_cost(points, nodeCount, start_point=0):
    start = points[start_point]
    points_traversed = [start_point]
    while len(points_traversed) != nodeCount:
        j = 0
        min_len = 99999999
        next_point = None
        while j < nodeCount:
            if j not in points_traversed:
                this_len = length(start, points[j])
                if this_len < min_len:
                    min_len = this_len
                    next_point = j
            j += 1
        points_traversed.append(next_point)
        start = points[next_point]
    return points_traversed


def get_travel_cost(solution, points, nodeCount):
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])
    return obj


def get_travel_cost_2(solution, points, nodeCount):
    # calculate the length of the tour
    print(solution)
    print(solution[-1], points[solution[-1]])
    print(solution[0], points[solution[0]])
    print(length(points[solution[-1]], points[solution[0]]))
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])
    return obj

def adj_matrix(nodeCount, tour):
    adjacency_matrix = [[0 for _ in range(nodeCount)] for _ in range(nodeCount)]
    for current_vertex, next_vertex in zip(tour, tour[1:]):
        adjacency_matrix[current_vertex][next_vertex] = 1
    adjacency_matrix[tour[-1]][tour[0]] = 1
    #print(tour)
    #print(adjacency_matrix)
    return adjacency_matrix

def average_pairwise_distance(points, nodeCount):
    distance_matrix = [[0 for _ in range(nodeCount)] for _ in range(nodeCount)]
    pair_wise_distances = []
    for i in range(nodeCount):
        for j in range(nodeCount):
            distance_matrix[i][j] = length(points[i], points[j])

    # for i in range(len(distance_matrix)):
    #     for j in range(len(distance_matrix)):
    #         if j > i:
    #             pair_wise_distances.append(distance_matrix[i][j])
    #
    # total_distance = sum(pair_wise_distances)
    # num_pairs = len(pair_wise_distances)
    # print("**********************", total_distance, num_pairs)
    # print(total_distance/num_pairs)

    distances = [distance_matrix[i][j] for i in range(len(distance_matrix)) for j in
                 range(i + 1, len(distance_matrix[i]))]
    total_distance = sum(distances)
    num_pairs = len(distances)
    # print("**********************", total_distance, num_pairs)
    # print(total_distance / num_pairs)
    return total_distance/num_pairs

def calculate_tour(adjacency_matrix, start_node, nodeCount):
    tour = [start_node]
    next_node = start_node
    while len(tour) < nodeCount:
        for i, value in enumerate(adjacency_matrix[next_node]):
            if value == 1:
                tour.append(i)
                next_node = i
    return tour


def tsp(nodeCount, points):
    solution = points.copy()
    random.shuffle(solution)
    tour = [point.index for point in solution]
    adjacency_matrix = adj_matrix(nodeCount, tour)

    obj = length(points[tour[-1]], points[tour[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[tour[index]], points[tour[index + 1]])

    vertex_selected = []
    tours_dist = {}
    remaining_vertices = solution.copy()
    print(solution)
    print(remaining_vertices)
    while len(vertex_selected) < nodeCount:
        print("While Loop")
        vertex_1 = random.choice(solution)
        if vertex_1 not in vertex_selected:
            vertex_selected.append(vertex_1)
            remaining_vertices.remove(vertex_1)
            if solution.index(vertex_1) < nodeCount - 1:
                vertex_2 = solution[solution.index(vertex_1) + 1]
            else:
                vertex_2 = solution[0]
            print(vertex_1, vertex_2, "\n")
            distance = length(vertex_1, vertex_2)
            min_vertex = vertex_2
            min_index = None
            print("For Loop Start")
            for index, point in enumerate(solution):
                # print(point)
                if point not in [vertex_1, vertex_2]:
                    edge_distance = length(point, vertex_2)
                    if distance > edge_distance:
                        distance = edge_distance
                        min_vertex = point
                        min_index = index
            print("For Loop end", min_index, min_vertex)
            if min_index is not None:
                adjacency_matrix[vertex_1.index][vertex_2.index] = 0
                for i, row in enumerate(adjacency_matrix):
                    if row[min_vertex.index] == 1:
                        adjacency_matrix[i][min_vertex.index] = 0
                        break
                adjacency_matrix[vertex_2.index][i] = 0
                adjacency_matrix[i][vertex_2.index] = 1
                adjacency_matrix[vertex_2.index][min_vertex.index] = 1
                adjacency_matrix[vertex_1.index][i] = 1

                # for index, edge in enumerate(adjacency_matrix[min_vertex.index]):
                #     if edge == 1:
                #         adjacency_matrix[min_vertex.index][index] = 0
                #         adjacency_matrix[index][min_vertex.index] = 1
                # adjacency_matrix[vertex_1.index][index] = 1
                # print(adjacency_matrix)
                # print("********")
                # return 0

                # after vertex 2 place min vertex
                # solution.insert(solution.index(vertex_2, min_vertex))

                # incoming_to_min_vertex = solution[min_index-1]
                # incoming_to_min_vertex
                # del solution[min_index]
                # index_to_insert = solution.index(vertex_2)
                # solution.insert(index_to_insert, min_vertex)
                # # if min_index-1 == -1:
                # #     solution.insert(solution.index(vertex_1), solution[-1])
                # #     del solution[-1]
                # # else:
                # index_to_delete = solution.index(min_vertex)
                # solution.insert(solution.index(vertex_1), solution[min_index-1])
                # del solution[index_to_delete]
        tour = calculate_tour(adjacency_matrix, vertex_1.index, nodeCount)
        # tour = [point.index for point in solution]
        print(tour)
        print(points)
        print(tour[-1], points[tour[-1]])
        print(tour[0], points[tour[0]])
        obj = length(points[tour[-1]], points[tour[0]])
        for index in range(0, nodeCount - 1):
            print(points[tour[index]], points[tour[index+1]])
            obj += length(points[tour[index]], points[tour[index + 1]])
        tours_dist[obj] = tour
        print(tour)
        print("While iteration end")

    print(tours_dist)
    min_tour_key = min(tours_dist.keys())
    tour = tours_dist[min_tour_key]
    print("Min tour", tour, min_tour_key)
    return tour

def perform_2opt_move(tour):
    new_tour = tour[:]
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour[i:j] = reversed(new_tour[i:j])
    return new_tour

def get_initial_temp(tour, points, nodeCount):
    delta_costs = []
    initial_cost = get_travel_cost(tour, points, nodeCount)
    for _ in range(100):
        new_tour = perform_2opt_move(tour)
        new_cost = get_travel_cost(new_tour, points, nodeCount)
        delta_costs.append(new_cost - initial_cost)
    delta_costs_std = np.std(delta_costs)
    return delta_costs_std / math.log(0.8)


def accept_prob(current_distance, new_distance, temp):
    if new_distance < current_distance:
        return 1
    else:
        return pow(math.e, -(new_distance-current_distance)/temp)

def tsp_sa_attempt(points, nodeCount):
    # solution from a hint

    solution = tsp_travel_cost(points, nodeCount, 0)
    obj = get_travel_cost(solution, points, nodeCount)
    min_cost = obj
    i = 1
    while i < nodeCount:
        this_solution = tsp_travel_cost(points, nodeCount, i)
        this_cost = get_travel_cost(this_solution, points, nodeCount)
        # print(this_cost)
        if this_cost < min_cost:
            solution = this_solution
            obj = this_cost
        i += 1
    #print(solution)
    #print(obj)
    initial_tour = solution
    min_distance = obj
    # initial_tour = list(random.sample(points, nodeCount))
    #tour = [point.index for point in initial_tour]
    tour = initial_tour.copy()
    min_tour = initial_tour.copy()
    #min_tour = [point.index for point in initial_tour]
    # min_distance = get_travel_cost(min_tour, points, nodeCount)
    max_iterations = 1000
    temp = average_pairwise_distance(points, nodeCount)
    #temp = get_initial_temp(min_tour, points, nodeCount)
    #print("New Temp", temp)
    #temp = 1000
    cooling = 0.995

    for _ in range(max_iterations):
        new_tour = tour.copy()
        i, j = random.sample(range(nodeCount), 2)
        if i < j:
            new_tour[i:j+1] = reversed(new_tour[i:j+1])
        else:
            new_tour[j:i+1] = reversed(new_tour[j:i+1])

        current_distance = get_travel_cost(tour, points, nodeCount)
        new_distance = get_travel_cost(new_tour, points, nodeCount)

        if new_distance < current_distance or random.random() < accept_prob(current_distance, new_distance, temp):
            tour = new_tour.copy()
            if new_distance < min_distance:
                min_tour = new_tour.copy()
                min_distance = new_distance

        temp *= cooling

    print(min_distance)
    print(min_tour)
    return min_tour, min_distance



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    new_points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
        new_points.append(NewPoint(i - 1, float(parts[0]), float(parts[1])))
    solution = tsp_sa_attempt(new_points, nodeCount)
    return 0

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)

    solution = tsp(nodeCount, new_points)

    # calculate the length of the tour
    # obj = length(points[solution[-1]], points[solution[0]])
    # for index in range(0, nodeCount - 1):
    #     obj += length(points[solution[index]], points[solution[index + 1]])

    if nodeCount not in [1889, 33810]:
        solution = tsp_travel_cost(points, nodeCount, 0)
        obj = get_travel_cost(solution, points, nodeCount)
        min_cost = obj
        i = 1
        while i < nodeCount:
            this_solution = tsp_travel_cost(points, nodeCount, i)
            this_cost = get_travel_cost(this_solution, points, nodeCount)
            # print(this_cost)
            if this_cost < min_cost:
                solution = this_solution
                obj = this_cost
            i += 1
    else:
        points_1 = new_points.copy()
        random.shuffle(points_1)
        solution = [point.index for point in points_1]
        obj = length(points[solution[-1]], points[solution[0]])
        for index in range(0, nodeCount - 1):
            obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
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
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
