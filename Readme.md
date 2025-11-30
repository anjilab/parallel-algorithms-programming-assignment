# For developer
- `gcc kmeans_serial.c -o kmeans_serial -lm`  [-lm is for]
- `./kmeans_serial`

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


# How to run a code