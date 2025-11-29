#!/bin/bash

# Output CSV file
output_file="dijkstra_timings.csv"
echo "nodes,edges,max_weight,threads,parallel_time,serial_time" > $output_file

# Arrays of parameters to try
# nodes_list=(1000 5000 10000 15000)
# edges_list=(5000 25000 50000 75000)

nodes_list=(1000 5000 10000)
edges_list=(5000 25000 50000)

max_weight_list=(10)
threads_list=(1 8 16)

# Compile codes
gcc -fopenmp dijkstra_omp.c -o dijkstra_omp
gcc dijkstra_serial_graph.c -o dijkstra_serial_graph

# Loop over parameters
for nodes in "${nodes_list[@]}"; do
    for edges in "${edges_list[@]}"; do
        for max_wt in "${max_weight_list[@]}"; do
            echo "Generating graph: nodes=$nodes edges=$edges max_weight=$max_wt"
            ./graph_generator $nodes $edges $max_wt weighted_graph.txt

            # Run parallel versions with different threads
            for threads in "${threads_list[@]}"; do
                export OMP_NUM_THREADS=$threads
                echo "Running parallel Dijkstra with $threads threads"
                
                # Measure time
                start=$(date +%s.%N)
                ./dijkstra_omp 0 > /dev/null
                end=$(date +%s.%N)
                elapsed=$(echo "$end - $start" | bc -l)

                # Run serial version only once (optional: could skip for every threads)
                echo "Running serial Dijkstra"
                start_serial=$(date +%s.%N)
                ./dijkstra_serial_graph 0 > /dev/null
                end_serial=$(date +%s.%N)
                elapsed_serial=$(echo "$end_serial - $start_serial" | bc -l)

                # Save to CSV
                echo "$nodes,$edges,$max_wt,$threads,$elapsed,$elapsed_serial" >> $output_file
            done
        done
    done
done

echo "All experiments done. Results saved in $output_file"
