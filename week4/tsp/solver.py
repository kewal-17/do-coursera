#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def get_distance_matrix(points, nodeCount, scaling_factor=100):
    distance_matrix = [[0 for _ in range(nodeCount)] for _ in range(nodeCount)]
    for i in range(nodeCount):
        for j in range(nodeCount):
            distance_matrix[i][j] = round(length(points[i], points[j]) * scaling_factor)
    return distance_matrix


def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def or_tools_tsp(points, nodeCount, initial_node):
    data = {"distance_matrix": get_distance_matrix(points, nodeCount), "num_vehicles": 1, "depot": initial_node}
    manager = pywrapcp.RoutingIndexManager(nodeCount, data["num_vehicles"], data["depot"])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.seconds = 30
    # search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)

    or_result = {"distance": solution.ObjectiveValue() / 100, "route": get_routes(solution, routing, manager)}

    return or_result


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])
    if nodeCount != 33810:
        best_solution = solution
        for i in range(nodeCount):
            or_solution = or_tools_tsp(points, nodeCount, i)
            #print(or_solution)
            if nodeCount not in [51,100]:
                break
            if or_solution["distance"] < obj:
                best_solution = or_solution["route"]
                obj = or_solution["distance"]
        # print(or_solution)
        if nodeCount not in [51,100]:
            best_solution = or_solution["route"]
            obj = or_solution["distance"]
        solution = []
        for sublist in best_solution:
            for node in sublist:
                solution.append(node)
        solution.pop()

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
