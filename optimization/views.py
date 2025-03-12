from django.shortcuts import render
import osmnx as ox
import networkx as nx
import random
import numpy as np
import base64
from io import BytesIO
from shapely.geometry import Polygon, Point
from django.http import JsonResponse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend

import matplotlib.pyplot as plt

# Function to evaluate road layout fitness
def evaluate_layout(layout, graph, obstacles):
    score = 0
    total_distance = 0
    intersection_penalty = 1000  # Increased penalty for bad road placement

    for road in layout:
        x_start, y_start, x_end, y_end = road
        distance = np.sqrt((x_start - x_end) ** 2 + (y_start - y_end) ** 2)
        total_distance += distance

        # Penalize intersections with buildings
        for obstruction in obstacles:
            if road_intersects(road, obstruction):
                score -= intersection_penalty  

    # Encourage shorter networks but prioritize avoiding intersections
    score -= total_distance * 0.5  # Reduce weight of total distance
    return score

# Function to check road-building intersection
def road_intersects(road, obstruction):
    if isinstance(obstruction, Polygon):
        return obstruction.intersects(Point(road[:2])) or obstruction.intersects(Point(road[2:]))
    return False

# Function to generate a random road layout
def create_random_layout(graph):
    return [
        (
            graph.nodes[u]["x"], graph.nodes[u]["y"],
            graph.nodes[v]["x"], graph.nodes[v]["y"]
        ) 
        for u, v, _ in random.sample(list(graph.edges(data=True)), min(len(graph.edges), 10))
    ]

# Genetic Algorithm for road network optimization
def optimize_road_network(graph, obstacles, generations=10, pop_size=20, mutation_rate=0.3):
    population = [create_random_layout(graph) for _ in range(pop_size)]

    for gen in range(generations):
        fitness_scores = [evaluate_layout(ind, graph, obstacles) for ind in population]

        # Select the top half of individuals
        best_indices = np.argsort(fitness_scores)[-pop_size // 2:]
        selected_population = [population[i] for i in best_indices]

        next_population = []
        while len(next_population) < pop_size:
            parent_a, parent_b = random.sample(selected_population, 2)
            split = random.randint(1, len(parent_a) - 1)
            child = parent_a[:split] + parent_b[split:]

            # Mutation for diversity
            if random.random() < mutation_rate:
                mutate_index = random.randint(0, len(child) - 1)
                new_point1, new_point2 = random.sample(list(graph.nodes), 2)
                x1, y1 = graph.nodes[new_point1]["x"], graph.nodes[new_point1]["y"]
                x2, y2 = graph.nodes[new_point2]["x"], graph.nodes[new_point2]["y"]
                child[mutate_index] = (x1, y1, x2, y2)

            next_population.append(child)

        population = next_population

    return max(population, key=lambda x: evaluate_layout(x, graph, obstacles))

# Function to render and encode road network image
def render_network(graph, obstacles, roads=None, is_optimized=False):
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_color="blue", edge_color="gray")

    for obs in obstacles:
        if obs.geom_type == "Polygon":
            ax.plot(*obs.exterior.xy, color="red", alpha=0.5)
        elif obs.geom_type == "Point":
            ax.plot(obs.x, obs.y, "ro", markersize=10, alpha=0.8)

    if roads:
        for road in roads:
            color = "green" if is_optimized else "blue"  # Optimized roads in green
            linewidth = 4 if is_optimized else 2
            ax.plot([road[0], road[2]], [road[1], road[3]], color=color, linewidth=linewidth)

    buffer = BytesIO()
    fig.savefig(buffer, format="png",dpi=300)
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return encoded_image

# Django API to generate and optimize road networks
def generate_road_network(request):
    region = request.GET.get("region", "Austin, Texas, USA")

    road_graph = ox.graph_from_place(region, network_type="drive")

    # Define bounding box
    north, south, east, west = 30.28, 30.25, -97.72, -97.75  # Austin, TX

    try:
        building_data = ox.features.features_from_bbox(north, south, east, west, {"building": True})


        obstacle_list = building_data.geometry.dropna().tolist() if not building_data.empty else []
    except Exception as e:
        print(f"❌ Error fetching building data: {e}")
        obstacle_list = []

    # Generate original network image
    original_image_data = render_network(road_graph, obstacle_list)

    # Optimize road layout
    optimal_network = optimize_road_network(road_graph, obstacle_list, generations=10, pop_size=20, mutation_rate=0.3)

    # Generate optimized network image with new color
    optimized_image_data = render_network(road_graph, obstacle_list, optimal_network, is_optimized=True)

    return JsonResponse({
        #"original_image": original_image_data,
        "optimized_image": optimized_image_data
    })

from django.http import JsonResponse

def default_view(request):
    return JsonResponse({"message": "Welcome to the Road Network Optimization API!"})

def index(request):
    return render(request, 'optimization/index.html')