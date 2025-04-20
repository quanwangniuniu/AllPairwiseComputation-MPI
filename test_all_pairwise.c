#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "all_pairwise.h"

// Test function for argument parsing
void test_parse_arguments() {
    Config config;
    char *valid_args[] = {
        "program_name",
        "1024",
        "2048",
        "float"
    };
    
    // Test valid arguments
    assert(parse_arguments(4, valid_args, &config) == true);
    assert(config.M == 1024);
    assert(config.N == 2048);
    assert(config.is_float == true);
    
    // Test invalid number of arguments
    char *invalid_args1[] = {"program_name", "100", "100"};
    assert(parse_arguments(3, invalid_args1, &config) == false);
    
    // Test invalid data type
    char *invalid_args2[] = {"program_name", "100", "100", "complex"};
    assert(parse_arguments(4, invalid_args2, &config) == false);
    
    // Test invalid number format
    char *invalid_args3[] = {"program_name", "100", "abcd", "float"};
    assert(parse_arguments(4, invalid_args3, &config) == false);
    
    // Test negative numbers
    char *invalid_args4[] = {"program_name", "100", "-200", "float"};
    assert(parse_arguments(4, invalid_args4, &config) == false);
    
    printf("All argument parsing tests passed!\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    Config config;
    MPI_Comm_size(MPI_COMM_WORLD, &config.num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.rank);
    
    // Redirect stdout to file for test results
    if (config.rank == 0) {
        freopen("test_results.txt", "w", stdout);
    }
    
    // Test case 1: Small matrix with float
    if (config.rank == 0) {
        printf("Test Case 1: Small matrix with float\n");
        printf("----------------------------------\n");
    }
    
    config.M = 5;
    config.N = 4;
    config.is_float = true;
    
    void *input_matrix = allocate_matrix(config.N, config.M, config.is_float);
    void *result_matrix = allocate_matrix(config.M, config.M, config.is_float);
    
    if (config.rank == 0) {
        initialize_matrix(input_matrix, &config);
    }
    
    compute_all_pairwise(input_matrix, result_matrix, &config);
    
    if (config.rank == 0) {
        print_results(result_matrix, &config);
        printf("\n");
    }
    
    free(input_matrix);
    free(result_matrix);
    
    // Test case 2: Medium matrix with double
    if (config.rank == 0) {
        printf("Test Case 2: Medium matrix with double\n");
        printf("------------------------------------\n");
    }
    
    config.M = 10;
    config.N = 8;
    config.is_float = false;
    
    input_matrix = allocate_matrix(config.N, config.M, config.is_float);
    result_matrix = allocate_matrix(config.M, config.M, config.is_float);
    
    if (config.rank == 0) {
        initialize_matrix(input_matrix, &config);
    }
    
    compute_all_pairwise(input_matrix, result_matrix, &config);
    
    if (config.rank == 0) {
        print_results(result_matrix, &config);
        printf("\n");
    }
    
    free(input_matrix);
    free(result_matrix);
    
    MPI_Finalize();
    return 0;
} 