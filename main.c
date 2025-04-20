#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "all_pairwise.h"

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    Config config;
    TimingInfo timer;
    MPI_Comm_size(MPI_COMM_WORLD, &config.num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.rank);
    
    // Print debug information
    if (config.rank == 0) {
        printf("Number of processes: %d\n", config.num_procs);
        printf("Number of arguments: %d\n", argc);
        for (int i = 0; i < argc; i++) {
            printf("Argument %d: %s\n", i, argv[i]);
        }
    }
    
    // Only process 0 handles command line arguments
    if (config.rank == 0) {
        if (!parse_arguments(argc, argv, &config)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Broadcast configuration to all processes
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Print configuration
    if (config.rank == 0) {
        printf("Configuration:\n");
        printf("M: %d\n", config.M);
        printf("N: %d\n", config.N);
        printf("Data type: %s\n", config.is_float ? "float" : "double");
    }
    
    // Allocate matrices
    void *input_matrix = NULL;
    void *result_matrix = NULL;
    
    if (config.rank == 0) {
        // Process 0 allocates the full matrices
        input_matrix = allocate_matrix(config.N, config.M, config.is_float);
        result_matrix = allocate_matrix(config.M, config.M, config.is_float);
        
        // Initialize input matrix with random values
        initialize_matrix(input_matrix, &config);
    } else {
        // Other processes allocate their portion of the matrices
        // TODO: Calculate local matrix size based on process count
    }
    
    // Perform computation
    if (config.num_procs > 1) {
        // Distributed computation
        start_timer(&timer);
        compute_all_pairwise(input_matrix, result_matrix, &config);
        stop_timer(&timer);
        
        if (config.rank == 0) {
            timer.parallel_time = get_elapsed_time(&timer);
            printf("Parallel computation time: %f seconds\n", timer.parallel_time);
        }
    } else {
        // Sequential computation
        start_timer(&timer);
        compute_all_pairwise(input_matrix, result_matrix, &config);
        stop_timer(&timer);
        
        timer.sequential_time = get_elapsed_time(&timer);
        printf("Sequential computation time: %f seconds\n", timer.sequential_time);
    }
    
    // Print results if in process 0
    if (config.rank == 0) {
        print_results(result_matrix, &config);
    }
    
    // Clean up
    if (config.rank == 0) {
        free(input_matrix);
        free(result_matrix);
    }
    
    MPI_Finalize();
    return 0;
}
