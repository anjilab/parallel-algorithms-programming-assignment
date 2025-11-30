


# How to run submitted code

1. To get the table in the report, run `bash run.sh`
2. To get the graph, first run above code and then only we can get `dijstra_timings.csv` which then can be plotted. Install `pip install pandas seaborn matplotlib`
3. If you want to run the code separately then, 
    - Serial version
        - To `./graph_generator 1000 5000 10 weighted_graph.txt` This will give us the input format for the graph. 
        - To get the compile code `gcc dijkstra_serial_graph.c  -o dijkstra_serial_graph`
        - To get the results `./dijkstra_serial_graph 0` here 0 is the source node.
    - Parallel version
        - `gcc -fopenmp dijkstra_omp.c -o dijkstra_omp`
        - `export OMP_NUM_THREADS=8`
        - `/dijkstra_omp 0` here 0 is the source node.





# OpenMp version
- Check if `gcc -fopenmp --version` 
- `gcc -fopenmp test.c -o test` 
- `./test`
- `gcc -fopenmp kmeans_omp.c -o kmeans_omp -lm`
- `./kmeans_omp` To speed up `export OMP_NUM_THREADS=8` and then run `./kmeans_omp`

# MPI version
- `which mpicc`
- ` mpicc test.c -o test_mpi -lm `
- `mpirun -np 4 ./test_mpi`

