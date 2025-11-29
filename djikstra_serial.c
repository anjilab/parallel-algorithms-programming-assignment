#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#define INF 1000  // substitute for "no connection"
#define MAX 7     // number of nodes

int node_index(char node, char nodes[]) {
    for (int i = 0; i < MAX; i++) {
        if (nodes[i] == node) return i;
    }
    return -1;
}

// Dijkstra function
void dijkstra(char nodes[], int adj[MAX][MAX], char src, char dest) {
    int visited[MAX] = {0};
    int distance[MAX];
    int parent[MAX];
    
    // Initialize distances
    for (int i = 0; i < MAX; i++) {
        distance[i] = INF;
        parent[i] = -1;
    }
    
    int s_idx = node_index(src, nodes);
    int d_idx = node_index(dest, nodes);
    distance[s_idx] = 0;
    
    for (int count = 0; count < MAX; count++) {
        // Find the unvisited node with smallest distance
        int min_dist = INF;
        int u = -1;
        for (int i = 0; i < MAX; i++) {
            if (!visited[i] && distance[i] < min_dist) {
                min_dist = distance[i];
                u = i;
            }
        }
        
        if (u == -1) break; // no reachable nodes remain
        visited[u] = 1;
        
        // Update distances of neighbors
        for (int v = 0; v < MAX; v++) {
            if (!visited[v] && adj[u][v] != INF) {
                int new_dist = distance[u] + adj[u][v];
                if (new_dist < distance[v]) {
                    distance[v] = new_dist;
                    parent[v] = u;
                }
            }
        }
    }
    
    // Reconstruct path
    int path[MAX];
    int count_path = 0;
    int current = d_idx;
    while (current != -1) {
        path[count_path++] = current;
        current = parent[current];
    }
    
    printf("Shortest path cost from %c to %c: %d\n", src, dest, distance[d_idx]);
    printf("Shortest path: ");
    for (int i = count_path - 1; i >= 0; i--) {
        printf("%c", nodes[path[i]]);
        if (i != 0) printf(" -> ");
    }
    printf("\n");
}

int main() {
    char nodes[MAX] = {'A','B','C','D','E','F','G'};
    
    // Adjacency matrix: INF if no edge
    int adj[MAX][MAX] = {
        {0, 3, 3, INF, INF, INF, INF},      // A
        {3, 0, INF, 3.5, 2.8, INF, INF},    // B
        {3, INF, 0, INF, 2.8, 3.5, INF},    // C
        {INF, 3.5, INF, 0, 3.1, INF, 10},   // D
        {INF, 2.8, 2.8, 3.1, 0, INF, 7},    // E
        {INF, INF, 3.5, INF, INF, 0, 2.5},  // F
        {INF, INF, INF, 10, 7, 2.5, 0}      // G
    };
    
    dijkstra(nodes, adj, 'B', 'F');
    dijkstra(nodes, adj, 'A', 'G');
    
    return 0;
}
