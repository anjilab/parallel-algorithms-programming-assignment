import numpy as np
def get_edge_wt(E, src_node, dest_node):

    if dest_node in E[src_node]:
        return E[src_node][dest_node]
    else:
        return 1000
    
def dijsktra_single_src(V, E, s, destination):
    shortest_path_arr = [s]
    parent = {s: None}
    print(shortest_path_arr)
    distance  = {}
    looping_arr = [node for node in V if node not in shortest_path_arr]
    print(looping_arr)
    for node in looping_arr:
        distance[node] = get_edge_wt(E, s, node)
        if distance[node] != 1000:
            parent[node] = s
        else:
            parent[node] = None
            
    count = 0        
    while(destination not in shortest_path_arr):
        u_vertex = ''
        minimum = 1000
        for node in looping_arr:
            if distance[node] < minimum:
                minimum = distance[node]
                u_vertex = node
        shortest_path_arr.append(u_vertex)
        
        looping_arr = [node for node in V if node not in shortest_path_arr] # V-V_t
        
        
        for node in looping_arr:
            new_distance = distance[u_vertex] + get_edge_wt(E, u_vertex, node)
            if new_distance < distance[node]:
                distance[node] = new_distance
                parent[node] = u_vertex
            # distance[node] = min(distance[node], distance[u_vertex] + get_edge_wt(E, u_vertex, node))
        
        
    
    # This is for getting the shortest path          
    path = []
    node = destination
    while node is not None:
        print(node, '=====heree')
        path.append(node)
        node = parent[node]
        
    path = path[::-1]
    print('Final destinaltion cost', distance)
    return f'The shortest path cost is going to be: {distance[destination]}, shortest path is: {path}'
            
        
                
            
        
    
    
    
    
print(dijsktra_single_src(['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                    {
                        'A': {'B': 3, 'C': 3}, 
                        'B': {'A': 3, 'E': 2, 'D': 3},
                        'C': {'A': 3, 'E': 2, 'F': 3},
                        'D': {'B': 3, 'E': 3, 'G': 10},
                        'E': {'D': 3, 'C': 2, 'B': 2, 'G': 7},
                        'F': {'C': 3 , 'G': 2},
                        'G': {'D': 10, 'E': 7, 'F': 2},
                    },
                    'B',
                    'F'
                    )
)

print(dijsktra_single_src(['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                    {
                        'A': {'B': 3, 'C': 3}, 
                        'B': {'A': 3, 'E': 2.8, 'D': 3.5},
                        'C': {'A': 3, 'E': 2.8, 'F': 3.5},
                        'D': {'B': 3.5, 'E': 3.1, 'G': 10},
                        'E': {'D': 3.1, 'C': 2.8, 'B': 2.8, 'G': 7},
                        'F': {'C': 3.5 , 'G': 2.5},
                        'G': {'D': 10, 'E': 7, 'F': 2.5},
                    },
                    'A',
                    'G'
                    )
)
  
    