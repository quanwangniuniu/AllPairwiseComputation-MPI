# All Pairwise Computation - MPI Implementation

> **Copyright © 2024 The University of Sydney. All rights reserved.**
> 
> This is an academic research project conducted at the University of Sydney. Unauthorized use, reproduction, or distribution of this work is strictly prohibited. For more information, please refer to the [ACADEMIC_DISCLAIMER.md](ACADEMIC_DISCLAIMER.md) file.

## Overview
This project implements a distributed-memory algorithm for computing pairwise dot products between sequences using MPI (Message Passing Interface). The implementation supports both sequential and parallel computation modes, with automatic selection based on the number of available processes.

## Problem Description
Given a set of M sequences, each of length N, stored in a two-dimensional matrix of size N × M, we calculate the dot product for every possible pair of sequences (column vectors) in the input matrix. The dot product of two sequences x and y is defined as:

```
x · y = Σ(xₖ * yₖ) for k = 0 to N-1
```

If we allow a vector to pair with itself, there are M(M + 1)/2 such pairs in total. The results are stored in the upper triangle of a symmetric M × M matrix.

## Features
- Support for both single process and multi-process execution
- Efficient distributed-memory algorithm with load balancing
- Support for both float and double precision
- Loop unrolling optimization (factor of 4)
- Comprehensive test suite
- Memory leak detection
- Performance measurement and comparison
- CI/CD integration with GitHub Actions

## Requirements
- C compiler (GCC)
- MPI implementation (OpenMPI)
- CMake (version 3.10 or higher)
- Make

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/AllPairwiseComputation-MPI.git
cd AllPairwiseComputation-MPI
```

2. Create build directory and compile:
```bash
mkdir build
cd build
cmake ..
make
```

## Usage
The program accepts three command-line arguments:
```bash
./all_pairwise M N float|double
```

Where:
- M: Number of sequences (positive integer)
- N: Length of each sequence (positive integer)
- float|double: Data type to use for computation

### Examples
```bash
# Run with float precision
./all_pairwise 5 4 float

# Run with double precision
./all_pairwise 10 8 double

# Run with multiple processes
mpirun -np 4 ./all_pairwise 100 50 float
```

## Implementation Details

### Data Distribution
- The input matrix is distributed among processes in row blocks
- Each process receives approximately N/p rows
- Process 0 serves as the main distributor
- Communication size is limited to ⌈N × M/p⌉ elements per round

### Algorithm
1. Process 0 initializes the input matrix with random values
2. Matrix is distributed to all processes
3. Each process computes its portion of the result matrix
4. Results are gathered back to process 0
5. Process 0 verifies results and prints output

### Optimizations
- Loop unrolling with factor of 4
- Efficient memory allocation and management
- Load balancing across processes
- Minimized communication overhead

## Testing
The project includes comprehensive test suites:

### Unit Tests
```bash
# Run sequential tests
mpirun -np 1 ./test_all_pairwise

# Run distributed tests
mpirun -np 2 ./test_distributed
```

### Performance Tests
The CI/CD pipeline runs performance tests with various matrix sizes:
- Small (5×4)
- Medium (10×8)
- Large (100×50)

### Memory Tests
Valgrind is used to check for memory leaks:
```bash
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes mpirun -np 1 ./all_pairwise 5 4 float
```

## CI/CD Pipeline
The project uses GitHub Actions for continuous integration and deployment. The pipeline:
1. Sets up the build environment
2. Installs dependencies
3. Builds the project
4. Runs tests
5. Performs memory checks
6. Uploads test results as artifacts

## Performance Analysis
The implementation measures and compares:
- Sequential computation time
- Parallel computation time
- Speedup ratio
- Memory usage

## Error Handling
The program includes comprehensive error handling for:
- Invalid command-line arguments
- Memory allocation failures
- MPI communication errors
- Numerical computation errors

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- University of Sydney - COMP5426 Parallel Computing
- OpenMPI development team
- GitHub Actions team 