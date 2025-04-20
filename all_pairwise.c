#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
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
        fprintf(stderr, "Error: Incorrect number of arguments. Usage: %s M N float|double\n", argv[0]);
        return false;
    }

    // Parse M
    char *endptr;
    errno = 0;
    long M = strtol(argv[1], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || M <= 0) {
        fprintf(stderr, "Error: M must be a positive integer\n");
        return false;
    }
    config->M = (int)M;

    // Parse N
    errno = 0;
    long N = strtol(argv[2], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || N <= 0) {
        fprintf(stderr, "Error: N must be a positive integer\n");
        return false;
    }
    config->N = (int)N;

    // Parse data type
    if (strcmp(argv[3], "float") == 0) {
        config->is_float = true;
    } else if (strcmp(argv[3], "double") == 0) {
        config->is_float = false;
    } else {
        fprintf(stderr, "Error: Data type must be either 'float' or 'double'\n");
        return false;
    }

    // Set student ID for random seed (replace with actual last 4 digits)
    config->student_id = 1234; // TODO: Replace with actual student ID

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
    int base_rows = config->N / config->num_procs;
    int extra_rows = config->N % config->num_procs;
    
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
    *local_matrix = malloc(local_info->local_rows * config->M * element_size);
    
    // Scatter matrix rows
    if (config->is_float) {
        MPI_Scatterv(input_matrix, local_info->row_counts, local_info->row_offsets,
                    MPI_FLOAT, *local_matrix, local_info->local_rows * config->M,
                    MPI_FLOAT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(input_matrix, local_info->row_counts, local_info->row_offsets,
                    MPI_DOUBLE, *local_matrix, local_info->local_rows * config->M,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

/**
 * Gathers results from all processes
 * @param result_matrix Full result matrix (only valid in process 0)
 * @param local_result Local result portion
 * @param config Program configuration
 * @param local_info Local computation information
 */
void gather_results(void *result_matrix, void *local_result, Config *config, LocalInfo *local_info) {
    // Calculate result counts and offsets
    int *result_counts = (int*)malloc(config->num_procs * sizeof(int));
    int *result_offsets = (int*)malloc(config->num_procs * sizeof(int));
    
    for (int i = 0; i < config->num_procs; i++) {
        result_counts[i] = local_info->row_counts[i] * config->M;
        result_offsets[i] = local_info->row_offsets[i] * config->M;
    }
    
    // Gather results
    if (config->is_float) {
        MPI_Gatherv(local_result, local_info->local_rows * config->M, MPI_FLOAT,
                   result_matrix, result_counts, result_offsets, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local_result, local_info->local_rows * config->M, MPI_DOUBLE,
                   result_matrix, result_counts, result_offsets, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
    }
    
    free(result_counts);
    free(result_offsets);
}

/**
 * Computes all pairwise dot products in distributed manner
 * @param input_matrix Input matrix containing sequences
 * @param result_matrix Matrix to store dot product results
 * @param config Program configuration
 */
void compute_all_pairwise_distributed(void *input_matrix, void *result_matrix, Config *config) {
    LocalInfo local_info;
    void *local_matrix = NULL;
    
    // Distribute matrix among processes
    distribute_matrix(input_matrix, &local_matrix, config, &local_info);
    
    // Allocate local result matrix
    size_t element_size = config->is_float ? sizeof(float) : sizeof(double);
    void *local_result = malloc(local_info.local_rows * config->M * element_size);
    
    // Perform local computations
    if (config->is_float) {
        float *matrix = (float*)local_matrix;
        float *result = (float*)local_result;
        
        for (int i = 0; i < local_info.local_rows; i++) {
            for (int j = 0; j < config->M; j++) {
                result[i * config->M + j] = dot_product_float(
                    &matrix[i * config->M],
                    &matrix[j * config->M],
                    config->N
                );
            }
        }
    } else {
        double *matrix = (double*)local_matrix;
        double *result = (double*)local_result;
        
        for (int i = 0; i < local_info.local_rows; i++) {
            for (int j = 0; j < config->M; j++) {
                result[i * config->M + j] = dot_product_double(
                    &matrix[i * config->M],
                    &matrix[j * config->M],
                    config->N
                );
            }
        }
    }
    
    // Gather results
    gather_results(result_matrix, local_result, config, &local_info);
    
    // Clean up
    free(local_matrix);
    free(local_result);
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
        
        if (config->is_float) {
            float *seq = (float*)sequential_result;
            float *par = (float*)parallel_result;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    float diff = fabs(seq[i * config->M + j] - par[i * config->M + j]);
                    if (diff > 1e-6) {
                        match = false;
                        max_diff = diff > max_diff ? diff : max_diff;
                    }
                }
            }
        } else {
            double *seq = (double*)sequential_result;
            double *par = (double*)parallel_result;
            
            for (int i = 0; i < config->M; i++) {
                for (int j = i; j < config->M; j++) {
                    double diff = fabs(seq[i * config->M + j] - par[i * config->M + j]);
                    if (diff > 1e-12) {
                        match = false;
                        max_diff = diff > max_diff ? diff : max_diff;
                    }
                }
            }
        }
        
        if (match) {
            printf("Verification successful: Sequential and parallel results match.\n");
        } else {
            printf("Verification failed: Maximum difference = %e\n", max_diff);
        }
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