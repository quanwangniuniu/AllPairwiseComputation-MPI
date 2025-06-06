#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include "all_pairwise.h"

/**
 * Parses command-line arguments and validates them
 * @param argc Number of arguments
 * @param argv Array of argument strings
 * @param config Pointer to Config structure to store parsed values
 * @return true if arguments are valid, false otherwise
 */
bool parse_arguments(int argc, char *argv[], Config *config) {
    // Check number of arguments
    if (argc != 4) {
        fprintf(stderr, "Error: Incorrect number of arguments (%d provided, 3 required)\n", argc - 1);
        fprintf(stderr, "Usage: %s M N float|double\n", argv[0]);
        fprintf(stderr, "  M: Number of sequences (positive integer)\n");
        fprintf(stderr, "  N: Length of each sequence (positive integer)\n");
        fprintf(stderr, "  Data type: 'float' or 'double'\n");
        return false;
    }

    // Parse M
    char *endptr;
    errno = 0;
    long M = strtol(argv[1], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || M <= 0) {
        fprintf(stderr, "Error: Invalid value for M ('%s')\n", argv[1]);
        fprintf(stderr, "M must be a positive integer\n");
        return false;
    }
    config->M = (int)M;

    // Parse N
    errno = 0;
    long N = strtol(argv[2], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || N <= 0) {
        fprintf(stderr, "Error: Invalid value for N ('%s')\n", argv[2]);
        fprintf(stderr, "N must be a positive integer\n");
        return false;
    }
    config->N = (int)N;

    // Parse data type
    if (strcmp(argv[3], "float") == 0) {
        config->is_float = true;
    } else if (strcmp(argv[3], "double") == 0) {
        config->is_float = false;
    } else {
        fprintf(stderr, "Error: Invalid data type ('%s')\n", argv[3]);
        fprintf(stderr, "Data type must be either 'float' or 'double' (case-sensitive)\n");
        return false;
    }

    // Set student ID for random seed (replace with actual last 4 digits)
    config->student_id = 6556; // Last 4 digits of student ID 540106556

    return true;
}

/**
 * Allocates memory for a matrix of specified size and type
 * @param rows Number of rows
 * @param cols Number of columns
 * @param is_float True for float, false for double
 * @return Pointer to allocated memory
 */
void* allocate_matrix(int rows, int cols, bool is_float) {
    size_t element_size = is_float ? sizeof(float) : sizeof(double);
    void* matrix = calloc(rows * cols, element_size);
    if (!matrix) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return matrix;
}

/**
 * Initializes matrix with random values between 0 and 1
 * @param matrix Pointer to matrix memory
 * @param config Program configuration
 */
void initialize_matrix(void *matrix, Config *config) {
    // Only process 0 initializes the matrix
    if (config->rank != 0) return;

    // Seed random number generator with student ID
    srand(config->student_id);

    if (config->is_float) {
        float *fmatrix = (float*)matrix;
        for (int i = 0; i < config->N * config->M; i++) {
            fmatrix[i] = (float)rand() / RAND_MAX;
        }
    } else {
        double *dmatrix = (double*)matrix;
        for (int i = 0; i < config->N * config->M; i++) {
            dmatrix[i] = (double)rand() / RAND_MAX;
        }
    }
}

/**
 * Starts the timer
 * @param timer Pointer to timing structure
 */
void start_timer(TimingInfo *timer) {
    timer->start_time = MPI_Wtime();
}

/**
 * Stops the timer
 * @param timer Pointer to timing structure
 */
void stop_timer(TimingInfo *timer) {
    timer->end_time = MPI_Wtime();
}

/**
 * Calculates elapsed time
 * @param timer Pointer to timing structure
 * @return Elapsed time in seconds
 */
double get_elapsed_time(TimingInfo *timer) {
    return timer->end_time - timer->start_time;
}

/**
 * Calculates dot product of two float vectors with loop unrolling factor of 4
 * @param vec1 First vector
 * @param vec2 Second vector
 * @param length Length of vectors
 * @return Dot product result
 */
float dot_product_float(const float* vec1, const float* vec2, int length) {
    float sum = 0.0f;
    int i;
    
    // Main loop with unrolling factor of 4
    for (i = 0; i < length - 3; i += 4) {
        sum += vec1[i] * vec2[i] +
               vec1[i+1] * vec2[i+1] +
               vec1[i+2] * vec2[i+2] +
               vec1[i+3] * vec2[i+3];
    }
    
    // Handle remaining elements
    for (; i < length; i++) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

/**
 * Calculates dot product of two double vectors with loop unrolling factor of 4
 * @param vec1 First vector
 * @param vec2 Second vector
 * @param length Length of vectors
 * @return Dot product result
 */
double dot_product_double(const double* vec1, const double* vec2, int length) {
    double sum = 0.0;
    int i;
    
    // Main loop with unrolling factor of 4
    for (i = 0; i < length - 3; i += 4) {
        sum += vec1[i] * vec2[i] +
               vec1[i+1] * vec2[i+1] +
               vec1[i+2] * vec2[i+2] +
               vec1[i+3] * vec2[i+3];
    }
    
    // Handle remaining elements
    for (; i < length; i++) {
        sum += vec1[i] * vec2[i];
    }
    
    return sum;
}

/**
 * Distributes the input matrix among all processes
 * @param input_matrix Full input matrix (only valid in process 0)
 * @param local_matrix Pointer to local matrix portion
 * @param config Program configuration
 * @param local_info Local computation information
 */
void distribute_matrix(void *input_matrix, void **local_matrix, Config *config, LocalInfo *local_info) {
    // Calculate row distribution
    local_info->row_counts = (int*)malloc(config->num_procs * sizeof(int));
    local_info->row_offsets = (int*)malloc(config->num_procs * sizeof(int));
    
    // Calculate rows per process
    int base_rows = config->M / config->num_procs;
    int extra_rows = config->M % config->num_procs;
    
    // Distribute rows
    int offset = 0;
    for (int i = 0; i < config->num_procs; i++) {
        local_info->row_counts[i] = base_rows + (i < extra_rows ? 1 : 0);
        local_info->row_offsets[i] = offset;
        offset += local_info->row_counts[i];
    }
    
    // Set local information
    local_info->local_rows = local_info->row_counts[config->rank];
    local_info->start_row = local_info->row_offsets[config->rank];
    local_info->end_row = local_info->start_row + local_info->local_rows;
    
    // Allocate local matrix
    size_t element_size = config->is_float ? sizeof(float) : sizeof(double);
    *local_matrix = malloc(local_info->local_rows * config->N * element_size);
    
    // Scatter matrix rows
    if (config->is_float) {
        MPI_Scatterv(input_matrix, 
                    local_info->row_counts, 
                    local_info->row_offsets, 
                    MPI_FLOAT,
                    *local_matrix, 
                    local_info->local_rows * config->N,
                    MPI_FLOAT, 
                    0, 
                    MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(input_matrix, 
                    local_info->row_counts, 
                    local_info->row_offsets, 
                    MPI_DOUBLE,
                    *local_matrix, 
                    local_info->local_rows * config->N,
                    MPI_DOUBLE, 
                    0, 
                    MPI_COMM_WORLD);
    }
}

/**
 * Gathers results from all processes
 * @param result_matrix Full result matrix (only valid in process 0)
 * @param local_result Local result portion
 * @param config Program configuration
 * @param local_info Local computation information
 * 
 * Note on Synchronization:
 * MPI_Gatherv is a collective operation that implicitly synchronizes all processes.
 * No explicit barriers are needed because:
 * 1. All processes must call MPI_Gatherv before any can proceed
 * 2. Process 0 cannot modify the result matrix until all data is received
 * 3. Other processes cannot proceed until their data is sent
 */
void gather_results(void *result_matrix, void *local_result, Config *config, LocalInfo *local_info) {
    // Calculate result counts and offsets
    int *result_counts = (int*)malloc(config->num_procs * sizeof(int));
    int *result_offsets = (int*)malloc(config->num_procs * sizeof(int));
    
    // Each process contributes local_rows * M elements to the result
    for (int i = 0; i < config->num_procs; i++) {
        result_counts[i] = local_info->row_counts[i] * config->M;
        result_offsets[i] = local_info->row_offsets[i] * config->M;
    }
    
    // Gather results using MPI_Gatherv (collective operation)
    if (config->is_float) {
        MPI_Gatherv(local_result, 
                   local_info->local_rows * config->M, 
                   MPI_FLOAT,
                   result_matrix, 
                   result_counts, 
                   result_offsets, 
                   MPI_FLOAT,
                   0, 
                   MPI_COMM_WORLD);
        
        // Zero out lower triangle (only in rank 0)
        // This is safe because MPI_Gatherv ensures all data is received
        if (config->rank == 0) {
            float *result = (float*)result_matrix;
            for (int i = 0; i < config->M; i++) {
                for (int j = 0; j < i; j++) {
                    result[i * config->M + j] = 0.0f;
                }
            }
        }
    } else {
        MPI_Gatherv(local_result, 
                   local_info->local_rows * config->M, 
                   MPI_DOUBLE,
                   result_matrix, 
                   result_counts, 
                   result_offsets, 
                   MPI_DOUBLE,
                   0, 
                   MPI_COMM_WORLD);
        
        // Zero out lower triangle (only in rank 0)
        // This is safe because MPI_Gatherv ensures all data is received
        if (config->rank == 0) {
            double *result = (double*)result_matrix;
            for (int i = 0; i < config->M; i++) {
                for (int j = 0; j < i; j++) {
                    result[i * config->M + j] = 0.0;
                }
            }
        }
    }
    
    free(result_counts);
    free(result_offsets);
}

/**
 * Computes all pairwise dot products in distributed mode
 * @param input_matrix Full input matrix (only valid in process 0)
 * @param result_matrix Full result matrix (only valid in process 0)
 * @param config Program configuration
 */
void compute_all_pairwise_distributed(void *input_matrix, void *result_matrix, Config *config) {
    LocalInfo local_info;
    void *local_matrix = NULL;
    void *local_result = NULL;
    
    // Distribute input matrix
    distribute_matrix(input_matrix, &local_matrix, config, &local_info);
    
    // Allocate local result matrix
    size_t element_size = config->is_float ? sizeof(float) : sizeof(double);
    local_result = malloc(local_info.local_rows * config->M * element_size);
    
    // Initialize local result to zero
    memset(local_result, 0, local_info.local_rows * config->M * element_size);
    
    // Allocate communication buffers
    void *send_buffer = malloc(config->N * element_size);
    void *recv_buffer = malloc(config->N * element_size);
    MPI_Request *send_requests = malloc(config->num_procs * sizeof(MPI_Request));
    MPI_Request *recv_requests = malloc(config->num_procs * sizeof(MPI_Request));
    MPI_Status *statuses = malloc(config->num_procs * sizeof(MPI_Status));
    
    // Process each local row
    for (int i = 0; i < local_info.local_rows; i++) {
        int global_i = local_info.start_row + i;
        
        // First compute dot products with local data
        for (int j = global_i; j < local_info.end_row; j++) {
            if (config->is_float) {
                float result = dot_product_float(
                    (float*)local_matrix + i * config->N,
                    (float*)local_matrix + (j - local_info.start_row) * config->N,
                    config->N
                );
                ((float*)local_result)[i * config->M + j] = result;
            } else {
                double result = dot_product_double(
                    (double*)local_matrix + i * config->N,
                    (double*)local_matrix + (j - local_info.start_row) * config->N,
                    config->N
                );
                ((double*)local_result)[i * config->M + j] = result;
            }
        }
        
        // Then handle non-local data using non-blocking communication
        int num_requests = 0;
        
        // Post receives for needed data
        for (int p = 0; p < config->num_procs; p++) {
            if (p != config->rank) {
                int start_j = local_info.row_offsets[p];
                int end_j = start_j + local_info.row_counts[p];
                
                // Only receive if we need data from this process
                if (end_j > global_i && start_j < config->M) {
                    if (config->is_float) {
                        MPI_Irecv(recv_buffer, config->N, MPI_FLOAT, p, 0,
                                MPI_COMM_WORLD, &recv_requests[num_requests++]);
                    } else {
                        MPI_Irecv(recv_buffer, config->N, MPI_DOUBLE, p, 0,
                                MPI_COMM_WORLD, &recv_requests[num_requests++]);
                    }
                }
            }
        }
        
        // Send our row to other processes that need it
        for (int p = 0; p < config->num_procs; p++) {
            if (p != config->rank) {
                int start_j = local_info.row_offsets[p];
                int end_j = start_j + local_info.row_counts[p];
                
                // Only send if the receiving process needs this row
                if (end_j > global_i && start_j < config->M) {
                    // Copy row to send buffer
                    if (config->is_float) {
                        memcpy(send_buffer, (float*)local_matrix + i * config->N,
                               config->N * element_size);
                        MPI_Isend(send_buffer, config->N, MPI_FLOAT, p, 0,
                                MPI_COMM_WORLD, &send_requests[p]);
                    } else {
                        memcpy(send_buffer, (double*)local_matrix + i * config->N,
                               config->N * element_size);
                        MPI_Isend(send_buffer, config->N, MPI_DOUBLE, p, 0,
                                MPI_COMM_WORLD, &send_requests[p]);
                    }
                } else {
                    // Initialize request to MPI_REQUEST_NULL for processes we don't send to
                    send_requests[p] = MPI_REQUEST_NULL;
                }
            }
        }
        
        // Wait for all communication to complete
        if (num_requests > 0) {
            MPI_Waitall(num_requests, recv_requests, statuses);
        }
        
        // Process received data
        for (int p = 0; p < config->num_procs; p++) {
            if (p != config->rank) {
                int start_j = local_info.row_offsets[p];
                int end_j = start_j + local_info.row_counts[p];
                
                if (end_j > global_i && start_j < config->M) {
                    for (int j = MAX(start_j, global_i); j < end_j; j++) {
                        if (config->is_float) {
                            float result = dot_product_float(
                                (float*)local_matrix + i * config->N,
                                (float*)recv_buffer,
                                config->N
                            );
                            ((float*)local_result)[i * config->M + j] = result;
                        } else {
                            double result = dot_product_double(
                                (double*)local_matrix + i * config->N,
                                (double*)recv_buffer,
                                config->N
                            );
                            ((double*)local_result)[i * config->M + j] = result;
                        }
                    }
                }
            }
        }
        
        // Cancel any outstanding sends
        for (int p = 0; p < config->num_procs; p++) {
            if (p != config->rank) {
                MPI_Request_free(&send_requests[p]);
            }
        }
    }
    
    // Gather results
    gather_results(result_matrix, local_result, config, &local_info);
    
    // Clean up
    free(local_matrix);
    free(local_result);
    free(send_buffer);
    free(recv_buffer);
    free(send_requests);
    free(recv_requests);
    free(statuses);
    free(local_info.row_counts);
    free(local_info.row_offsets);
}

/**
 * Computes all pairwise dot products for the input matrix
 * @param input_matrix Input matrix containing sequences
 * @param result_matrix Matrix to store dot product results
 * @param config Program configuration
 */
void compute_all_pairwise(void *input_matrix, void *result_matrix, Config *config) {
    if (config->num_procs > 1) {
        compute_all_pairwise_distributed(input_matrix, result_matrix, config);
    } else {
        // Original sequential computation
        if (config->is_float) {
            float *matrix = (float*)input_matrix;
            float *result = (float*)result_matrix;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    result[i * config->M + j] = dot_product_float(
                        &matrix[i * config->N],
                        &matrix[j * config->N],
                        config->N
                    );
                }
            }
        } else {
            double *matrix = (double*)input_matrix;
            double *result = (double*)result_matrix;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    result[i * config->M + j] = dot_product_double(
                        &matrix[i * config->N],
                        &matrix[j * config->N],
                        config->N
                    );
                }
            }
        }
    }
}

/**
 * Verifies that sequential and parallel results match
 * @param sequential_result Results from sequential computation
 * @param parallel_result Results from parallel computation
 * @param config Program configuration
 */
void verify_results(void *sequential_result, void *parallel_result, Config *config) {
    if (config->rank == 0) {
        bool match = true;
        double max_diff = 0.0;
        int max_diff_i = -1;
        int max_diff_j = -1;
        int mismatch_count = 0;
        const int MAX_MISMATCHES_TO_PRINT = 5;
        
        printf("\nVerifying results...\n");
        
        if (config->is_float) {
            float *seq = (float*)sequential_result;
            float *par = (float*)parallel_result;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    float diff = fabs(seq[i * config->M + j] - par[i * config->M + j]);
                    if (diff > config->float_tolerance) {
                        match = false;
                        if (diff > max_diff) {
                            max_diff = diff;
                            max_diff_i = i;
                            max_diff_j = j;
                        }
                        if (mismatch_count < MAX_MISMATCHES_TO_PRINT) {
                            printf("Mismatch at [%d,%d]: Sequential=%.6f, Parallel=%.6f, Diff=%.6f\n",
                                   i, j, seq[i * config->M + j], par[i * config->M + j], diff);
                        }
                        mismatch_count++;
                    }
                }
            }
        } else {
            double *seq = (double*)sequential_result;
            double *par = (double*)parallel_result;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    double diff = fabs(seq[i * config->M + j] - par[i * config->M + j]);
                    if (diff > config->double_tolerance) {
                        match = false;
                        if (diff > max_diff) {
                            max_diff = diff;
                            max_diff_i = i;
                            max_diff_j = j;
                        }
                        if (mismatch_count < MAX_MISMATCHES_TO_PRINT) {
                            printf("Mismatch at [%d,%d]: Sequential=%.12f, Parallel=%.12f, Diff=%.12f\n",
                                   i, j, seq[i * config->M + j], par[i * config->M + j], diff);
                        }
                        mismatch_count++;
                    }
                }
            }
        }
        
        if (match) {
            printf("Verification successful: Sequential and parallel results match.\n");
        } else {
            printf("\nVerification failed:\n");
            printf("Total mismatches: %d\n", mismatch_count);
            printf("Maximum difference = %e at position [%d,%d]\n", max_diff, max_diff_i, max_diff_j);
            if (mismatch_count > MAX_MISMATCHES_TO_PRINT) {
                printf("(Showing only first %d mismatches)\n", MAX_MISMATCHES_TO_PRINT);
            }
        }
        printf("\n");
    }
}

/**
 * Prints the upper triangle of the result matrix
 * @param result_matrix Matrix containing dot product results
 * @param config Program configuration
 */
void print_results(void *result_matrix, Config *config) {
    if (config->rank == 0) {
        printf("\nResult Matrix (Upper Triangle):\n");
        if (config->is_float) {
            float *result = (float*)result_matrix;
            for (int i = 0; i < config->M; i++) {
                for (int j = 0; j < config->M; j++) {
                    if (j >= i) {
                        printf("%8.2f ", result[i * config->M + j]);
                    } else {
                        printf("         ");
                    }
                }
                printf("\n");
            }
        } else {
            double *result = (double*)result_matrix;
            for (int i = 0; i < config->M; i++) {
                for (int j = 0; j < config->M; j++) {
                    if (j >= i) {
                        printf("%8.2f ", result[i * config->M + j]);
                    } else {
                        printf("         ");
                    }
                }
                printf("\n");
            }
        }
    }
} 