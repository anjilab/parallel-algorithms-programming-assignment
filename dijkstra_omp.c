#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>
#include <sys/time.h> 

typedef struct Edge {
    int neighbor;
    int weight;
    struct Edge *next;
} Edge;


typedef struct Graph {
    int num_nodes;
    Edge **adj_list; 
} Graph;



Edge* create_edge(int v, int w) {
    Edge* new_edge = (Edge*)malloc(sizeof(Edge));
    if (new_edge == NULL) { perror("Memory allocation failed for new edge"); exit(EXIT_FAILURE); }
    new_edge->neighbor = v;
    new_edge->weight = w;
    new_edge->next = NULL;
    return new_edge;
}

Graph* create_graph(int num_nodes) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    if (graph == NULL) { perror("Memory allocation failed for graph"); exit(EXIT_FAILURE); }
    graph->num_nodes = num_nodes;
    
    graph->adj_list = (Edge**)malloc(num_nodes * sizeof(Edge*));
    if (graph->adj_list == NULL) { perror("Memory allocation failed for adjacency list array"); free(graph); exit(EXIT_FAILURE); }

    for (int i = 0; i < num_nodes; i++) {
        graph->adj_list[i] = NULL;
    }
    return graph;
}

void add_edge(Graph* graph, int u, int v, int w) {
    // Undirected graph: add edge (u -> v) and (v -> u)
    Edge* new_u_v = create_edge(v, w);
    new_u_v->next = graph->adj_list[u];
    graph->adj_list[u] = new_u_v;

    Edge* new_v_u = create_edge(u, w);
    new_v_u->next = graph->adj_list[v];
    graph->adj_list[v] = new_v_u;
}


// --- 1. PARALLEL: Finding the Minimum Distance Node ---
int min_distance_parallel(int dist[], bool visited[], int num_nodes) {
    int min_val = INT_MAX;
    int min_index = -1;

    // Parallel region for searching the minimum distance node
    #pragma omp parallel default(none) shared(dist, visited, num_nodes, min_val, min_index)
    {
        int local_min_val = INT_MAX;
        int local_min_index = -1;

        // Loop over all nodes (each thread searches its own chunk)
        // Schedule 'static' or 'dynamic' can be used here. 'Static' is fine for this uniform work.
        #pragma omp for schedule(static)
        for (int v = 0; v < num_nodes; v++) {
            // Find the minimum distance node that has not been visited
            if (visited[v] == false && dist[v] < local_min_val) {
                local_min_val = dist[v];
                local_min_index = v;
            }
        }

        // Combine the local results into the global minimum using a critical section.
        // This is necessary because min_val and min_index are shared variables.
        #pragma omp critical
        {
            if (local_min_val < min_val) {
                min_val = local_min_val;
                min_index = local_min_index;
            }
        }
    }
    return min_index;
}


void dijkstra_openmp(Graph* graph, int src) {
    int num_nodes = graph->num_nodes;
    int *dist = (int*)malloc(num_nodes * sizeof(int));
    bool *visited = (bool*)malloc(num_nodes * sizeof(bool));
    int *parent = (int*)malloc(num_nodes * sizeof(int));

    if (!dist || !visited || !parent) {
        perror("Memory allocation failed for Dijkstra arrays");
        exit(EXIT_FAILURE);
    }

    // Parallel loop for initial setup
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_nodes; i++) {
        dist[i] = INT_MAX;
        visited[i] = false;
        parent[i] = -1;
    }

    dist[src] = 0;

    // Outer loop we cannot parallize so works sequentially
    for (int count = 0; count < num_nodes - 1; count++) {
        // 1. Minimum distance selection (Parallelized step)
        int u = min_distance_parallel(dist, visited, num_nodes);

        if (u == -1 || dist[u] == INT_MAX) {
            break;
        }

        visited[u] = true;

        // 2. Neighbor distance update 
    
        int neighbor_count = 0;
        Edge* current_edge_count = graph->adj_list[u];
        while (current_edge_count != NULL) {
            neighbor_count++;
            current_edge_count = current_edge_count->next;
        }

        // Allocate temporary arrays to hold neighbors and weights
        int *neighbors = (int*)malloc(neighbor_count * sizeof(int));
        int *weights = (int*)malloc(neighbor_count * sizeof(int));
        
        Edge* current_edge = graph->adj_list[u];
        for (int i = 0; i < neighbor_count; i++) {
            neighbors[i] = current_edge->neighbor;
            weights[i] = current_edge->weight;
            current_edge = current_edge->next;
        }

        // Parallel loop over the neighbors of selected node
        #pragma omp parallel for default(none) shared(dist, visited, parent, u, neighbors, weights, neighbor_count) schedule(static)
        for (int i = 0; i < neighbor_count; i++) {
            int v = neighbors[i];
            int weight = weights[i];

            // Relaxation check: This is safe as each thread updates a unique dist[v]
            if (!visited[v] && dist[u] != INT_MAX) {
                int new_dist = dist[u] + weight;
                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    parent[v] = u; 
                }
            }
        }
        
        free(neighbors);
        free(weights);
    }

    printf("\n--- Shortest Path Results ---\n");
    printf("Source node: %d\n", src);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: ", i);
        if (dist[i] == INT_MAX) {
            printf("INF\n");
        } else {
            printf("%d\n", dist[i]);
        }
    }

    free(dist);
    free(visited);
    free(parent);
}



int main(int argc, char *argv[]) {
    // Reading from the command `./graph_generator 10000 50000 100 weighted_graph.txt`
    const char *filename = "weighted_graph.txt"; 
    
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <source_node>\n", argv[0]);
        fprintf(stderr, "The program automatically reads graph data from '%s'.\n", filename);
        fprintf(stderr, "Example: ./dijkstra_openmp 0\n");
        return EXIT_FAILURE;
    }

    int src_node = atoi(argv[1]);

    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error: Could not open input file. Please ensure 'weighted_graph.txt' exists.");
        return EXIT_FAILURE;
    }

    int num_nodes, num_edges;
    if (fscanf(file, "%d %d", &num_nodes, &num_edges) != 2) {
        fprintf(stderr, "Error reading number of nodes and edges from the file.\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    if (src_node < 0 || src_node >= num_nodes) {
        fprintf(stderr, "Error: Source node %d is out of bounds [0, %d].\n", src_node, num_nodes - 1);
        fclose(file);
        return EXIT_FAILURE;
    }

    Graph* graph = create_graph(num_nodes);
    int u, v, w;

    for (int i = 0; i < num_edges; i++) {
        if (fscanf(file, "%d %d %d", &u, &v, &w) != 3) {
            fprintf(stderr, "Error reading edge %d. File corrupted or ended unexpectedly.\n", i + 1);
            fclose(file);
            return EXIT_FAILURE;
        }
        if (u >= num_nodes || v >= num_nodes) {
             fprintf(stderr, "Error: Edge nodes (%d, %d) exceed declared node count (%d).\n", u, v, num_nodes);
             fclose(file);
             return EXIT_FAILURE;
        }
        add_edge(graph, u, v, w);
    }
    fclose(file);

    printf("Successfully loaded graph with %d nodes and %d edges from '%s'.\n", num_nodes, num_edges, filename);
    printf("Running OpenMP Dijkstra's with %d threads.\n", omp_get_max_threads());
    
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    dijkstra_openmp(graph, src_node);
    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("\n--- Performance ---\n");
    printf("Execution time: %f seconds\n", elapsed_time);

    printf("----------------------------------------------------------------\nTotal Execution time: %.3f s\n", elapsed_time);
    

    // Final graph memory cleanup
    for (int i = 0; i < num_nodes; i++) {
        Edge* current = graph->adj_list[i];
        while (current != NULL) {
            Edge* temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(graph->adj_list);
    free(graph);

    return EXIT_SUCCESS;
}