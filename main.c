#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "all_pairwise.h"

// Define student ID (last 9 digits of SID)
#define STUDENT_ID 540156556

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    Config config;
    TimingInfo timer;
    MPI_Comm_size(MPI_COMM_WORLD, &config.num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.rank);
    
    // Set student ID
    config.student_id = STUDENT_ID;
    
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
            fprintf(stderr, "Error: Failed to parse arguments. Aborting.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    
    // Broadcast configuration to all processes
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Print configuration
    if (config.rank == 0) {
        printf("\nConfiguration:\n");
        printf("M (sequences): %d\n", config.M);
        printf("N (sequence length): %d\n", config.N);
        printf("Data type: %s\n", config.is_float ? "float" : "double");
        printf("Student ID: %d\n", config.student_id);
        printf("Number of processes: %d\n\n", config.num_procs);
    }
    
    // Allocate matrices
    void *input_matrix = NULL;
    void *result_matrix = NULL;
    void *sequential_result = NULL;
    
    if (config.rank == 0) {
        // Process 0 allocates the full matrices
        input_matrix = allocate_matrix(config.N, config.M, config.is_float);
        result_matrix = allocate_matrix(config.M, config.M, config.is_float);
        sequential_result = allocate_matrix(config.M, config.M, config.is_float);
        
        // Initialize input matrix with random values
        initialize_matrix(input_matrix, &config);
    } else {
        // Other processes allocate their portion of the matrices
        int local_rows = config.M / config.num_procs + 
                        (config.rank < (config.M % config.num_procs) ? 1 : 0);
        input_matrix = allocate_matrix(local_rows, config.N, config.is_float);
        result_matrix = allocate_matrix(local_rows, config.M, config.is_float);
    }
    
    // Perform computation
    if (config.num_procs > 1) {
        // First run sequential computation on process 0
        if (config.rank == 0) {
            start_timer(&timer);
            compute_all_pairwise(input_matrix, sequential_result, &config);
            stop_timer(&timer);
            timer.sequential_time = get_elapsed_time(&timer);
            printf("Sequential computation time: %f seconds\n", timer.sequential_time);
        }
        
        // Then run distributed computation
        start_timer(&timer);
        compute_all_pairwise(input_matrix, result_matrix, &config);
        stop_timer(&timer);
        
        if (config.rank == 0) {
            timer.parallel_time = get_elapsed_time(&timer);
            printf("Parallel computation time: %f seconds\n", timer.parallel_time);
            printf("Speedup: %f\n", timer.sequential_time / timer.parallel_time);
            
            // Verify results
            verify_results(sequential_result, result_matrix, &config);
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
        free(sequential_result);
    } else {
        free(input_matrix);
        free(result_matrix);
    }
    
    MPI_Finalize();
    return 0;
}
