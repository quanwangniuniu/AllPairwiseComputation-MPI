#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include "all_pairwise.h"

/**
 * Tests matrix distribution and gathering
 * @param config Program configuration
 */
void test_matrix_distribution(Config *config) {
    // Allocate test matrices
    void *input_matrix = allocate_matrix(config->N, config->M, config->is_float);
    void *local_matrix = NULL;
    LocalInfo local_info;
    
    // Initialize test matrix in process 0
    if (config->rank == 0) {
        if (config->is_float) {
            float *matrix = (float*)input_matrix;
            for (int i = 0; i < config->N * config->M; i++) {
                matrix[i] = (float)i;
            }
        } else {
            double *matrix = (double*)input_matrix;
            for (int i = 0; i < config->N * config->M; i++) {
                matrix[i] = (double)i;
            }
        }
    }
    
    // Test distribution
    distribute_matrix(input_matrix, &local_matrix, config, &local_info);
    
    // Verify local matrix contents
    if (config->is_float) {
        float *matrix = (float*)local_matrix;
        for (int i = 0; i < local_info.local_rows; i++) {
            for (int j = 0; j < config->M; j++) {
                int expected = (local_info.start_row + i) * config->M + j;
                assert(matrix[i * config->M + j] == (float)expected);
            }
        }
    } else {
        double *matrix = (double*)local_matrix;
        for (int i = 0; i < local_info.local_rows; i++) {
            for (int j = 0; j < config->M; j++) {
                int expected = (local_info.start_row + i) * config->M + j;
                assert(matrix[i * config->M + j] == (double)expected);
            }
        }
    }
    
    // Clean up
    if (config->rank == 0) {
        free(input_matrix);
    }
    free(local_matrix);
    free(local_info.row_counts);
    free(local_info.row_offsets);
}

/**
 * Tests dot product computation
 * @param config Program configuration
 */
void test_dot_product(Config *config) {
    // Create test vectors
    void *vec1 = malloc(config->N * (config->is_float ? sizeof(float) : sizeof(double)));
    void *vec2 = malloc(config->N * (config->is_float ? sizeof(float) : sizeof(double)));
    
    if (config->is_float) {
        float *v1 = (float*)vec1;
        float *v2 = (float*)vec2;
        for (int i = 0; i < config->N; i++) {
            v1[i] = (float)i;
            v2[i] = (float)(i + 1);
        }
        
        float result = dot_product_float(v1, v2, config->N);
        float expected = 0.0f;
        for (int i = 0; i < config->N; i++) {
            expected += v1[i] * v2[i];
        }
        assert(fabs(result - expected) < 1e-6);
    } else {
        double *v1 = (double*)vec1;
        double *v2 = (double*)vec2;
        for (int i = 0; i < config->N; i++) {
            v1[i] = (double)i;
            v2[i] = (double)(i + 1);
        }
        
        double result = dot_product_double(v1, v2, config->N);
        double expected = 0.0;
        for (int i = 0; i < config->N; i++) {
            expected += v1[i] * v2[i];
        }
        assert(fabs(result - expected) < 1e-12);
    }
    
    free(vec1);
    free(vec2);
}

/**
 * Tests distributed computation
 * @param config Program configuration
 */
void test_distributed_computation(Config *config) {
    // Allocate matrices
    void *input_matrix = NULL;
    void *result_matrix = NULL;
    void *sequential_result = NULL;
    
    if (config->rank == 0) {
        input_matrix = allocate_matrix(config->N, config->M, config->is_float);
        result_matrix = allocate_matrix(config->M, config->M, config->is_float);
        sequential_result = allocate_matrix(config->M, config->M, config->is_float);
        
        // Initialize input matrix
        initialize_matrix(input_matrix, config);
        
        // Compute sequential result
        compute_all_pairwise(input_matrix, sequential_result, config);
    }
    
    // Compute distributed result
    compute_all_pairwise_distributed(input_matrix, result_matrix, config);
    
    // Verify results
    if (config->rank == 0) {
        verify_results(sequential_result, result_matrix, config);
        
        // Clean up
        free(input_matrix);
        free(result_matrix);
        free(sequential_result);
    }
}

// Test helper function to initialize a small matrix with known values
void initialize_test_matrix(void *matrix, Config *config) {
    if (config->rank != 0) return;
    
    if (config->is_float) {
        float *fmatrix = (float*)matrix;
        for (int i = 0; i < config->M; i++) {
            for (int j = 0; j < config->N; j++) {
                fmatrix[i * config->N + j] = (float)(i + j + 1);  // Simple pattern for verification
            }
        }
    } else {
        double *dmatrix = (double*)matrix;
        for (int i = 0; i < config->M; i++) {
            for (int j = 0; j < config->N; j++) {
                dmatrix[i * config->N + j] = (double)(i + j + 1);  // Simple pattern for verification
            }
        }
    }
}

// Test helper function to verify results
void verify_test_results(void *result_matrix, Config *config) {
    if (config->rank != 0) return;
    
    printf("Test Results:\n");
    if (config->is_float) {
        float *result = (float*)result_matrix;
        for (int i = 0; i < config->M; i++) {
            for (int j = i; j < config->M; j++) {
                float expected = 0.0f;
                for (int k = 0; k < config->N; k++) {
                    float val1 = (float)(i + k + 1);
                    float val2 = (float)(j + k + 1);
                    expected += val1 * val2;
                }
                float actual = result[i * config->M + j];
                printf("Expected[%d,%d] = %.2f, Actual = %.2f\n", i, j, expected, actual);
                assert(fabs(expected - actual) < 1e-4);
            }
        }
    } else {
        double *result = (double*)result_matrix;
        for (int i = 0; i < config->M; i++) {
            for (int j = i; j < config->M; j++) {
                double expected = 0.0;
                for (int k = 0; k < config->N; k++) {
                    double val1 = (double)(i + k + 1);
                    double val2 = (double)(j + k + 1);
                    expected += val1 * val2;
                }
                double actual = result[i * config->M + j];
                printf("Expected[%d,%d] = %.2f, Actual = %.2f\n", i, j, expected, actual);
                assert(fabs(expected - actual) < 1e-8);
            }
        }
    }
    printf("All test cases passed!\n");
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    Config config;
    MPI_Comm_size(MPI_COMM_WORLD, &config.num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &config.rank);
    
    // Redirect stdout to file for test results
    if (config.rank == 0) {
        freopen("test_results.txt", "a", stdout);
        printf("\nDistributed Computation Tests\n");
        printf("============================\n");
    }
    
    // Test case 1: Small matrix with float
    if (config.rank == 0) {
        printf("\nTest Case 1: Small matrix with float (Distributed)\n");
        printf("-----------------------------------------------\n");
    }
    
    config.M = 3;
    config.N = 2;
    config.is_float = true;
    
    if (config.rank == 0) {
        printf("\nTest Case 1: M=%d, N=%d, type=float\n", config.M, config.N);
    }
    
    // Broadcast configuration
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Allocate and initialize test matrix
    void *input_matrix = NULL;
    void *result_matrix = NULL;
    
    if (config.rank == 0) {
        input_matrix = allocate_matrix(config.M, config.N, config.is_float);
        result_matrix = allocate_matrix(config.M, config.M, config.is_float);
        initialize_test_matrix(input_matrix, &config);
    }
    
    // Run distributed computation
    compute_all_pairwise(input_matrix, result_matrix, &config);
    
    // Verify results
    verify_test_results(result_matrix, &config);
    
    // Clean up
    if (config.rank == 0) {
        free(input_matrix);
        free(result_matrix);
    }
    
    // Test case 2: Small matrix with double
    if (config.rank == 0) {
        printf("\nTest Case 2: Small matrix with double (Distributed)\n");
        printf("-------------------------------------------------\n");
    }
    
    config.M = 3;
    config.N = 2;
    config.is_float = false;
    
    if (config.rank == 0) {
        printf("\nTest Case 2: M=%d, N=%d, type=double\n", config.M, config.N);
    }
    
    // Broadcast configuration
    MPI_Bcast(&config, sizeof(Config), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if (config.rank == 0) {
        input_matrix = allocate_matrix(config.M, config.N, config.is_float);
        result_matrix = allocate_matrix(config.M, config.M, config.is_float);
        initialize_test_matrix(input_matrix, &config);
    }
    
    // Run distributed computation
    compute_all_pairwise(input_matrix, result_matrix, &config);
    
    // Verify results
    verify_test_results(result_matrix, &config);
    
    // Clean up
    if (config.rank == 0) {
        free(input_matrix);
        free(result_matrix);
    }
    
    MPI_Finalize();
    return 0;
} 