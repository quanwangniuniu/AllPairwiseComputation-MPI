#ifndef ALL_PAIRWISE_H
#define ALL_PAIRWISE_H

#include <mpi.h>
#include <stdbool.h>
#include <time.h>

// Utility macro for finding maximum of two numbers
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Structure to hold program configuration
typedef struct {
    int M;              // Number of sequences
    int N;              // Length of each sequence
    bool is_float;      // True if using float, false if using double
    int num_procs;      // Number of MPI processes
    int rank;           // Current process rank
    int student_id;     // Last 4 digits of student ID for random seed
} Config;

// Structure to hold timing information
typedef struct {
    double start_time;
    double end_time;
    double sequential_time;
    double parallel_time;
} TimingInfo;

// Structure to hold local computation information
typedef struct {
    int local_rows;     // Number of rows assigned to this process
    int start_row;      // Starting row index for this process
    int end_row;        // Ending row index for this process
    int *row_counts;    // Array of row counts for each process
    int *row_offsets;   // Array of row offsets for each process
} LocalInfo;

// Function declarations
bool parse_arguments(int argc, char *argv[], Config *config);
void* allocate_matrix(int rows, int cols, bool is_float);
void initialize_matrix(void *matrix, Config *config);
void compute_all_pairwise(void *input_matrix, void *result_matrix, Config *config);
void compute_all_pairwise_distributed(void *input_matrix, void *result_matrix, Config *config);
void verify_results(void *sequential_result, void *parallel_result, Config *config);
void start_timer(TimingInfo *timer);
void stop_timer(TimingInfo *timer);
double get_elapsed_time(TimingInfo *timer);
void print_results(void *result_matrix, Config *config);
void distribute_matrix(void *input_matrix, void **local_matrix, Config *config, LocalInfo *local_info);
void gather_results(void *result_matrix, void *local_result, Config *config, LocalInfo *local_info);

// Helper functions for dot product computation
float dot_product_float(const float* vec1, const float* vec2, int length);
double dot_product_double(const double* vec1, const double* vec2, int length);

#endif // ALL_PAIRWISE_H 