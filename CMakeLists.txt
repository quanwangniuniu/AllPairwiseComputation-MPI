cmake_minimum_required(VERSION 3.10)
project(AllPairwiseComputation-MPI)

# Find MPI package
find_package(MPI REQUIRED)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add executable
add_executable(all_pairwise main.c all_pairwise.c)

# Add test executables
add_executable(test_all_pairwise test_all_pairwise.c all_pairwise.c)
add_executable(test_distributed test_distributed.c all_pairwise.c)

# Link MPI libraries
target_link_libraries(all_pairwise MPI::MPI_C)
target_link_libraries(test_all_pairwise MPI::MPI_C)
target_link_libraries(test_distributed MPI::MPI_C)

# Add test
enable_testing()
add_test(NAME test_all_pairwise COMMAND mpirun -np 1 test_all_pairwise)
add_test(NAME test_distributed COMMAND mpirun -np 2 test_distributed)
