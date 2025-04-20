# CMake generated Testfile for 
# Source directory: I:/COMP5426/Assignment2/AllPairwiseComputation-MPI
# Build directory: I:/COMP5426/Assignment2/AllPairwiseComputation-MPI/cmake-build-debug
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_all_pairwise "mpirun" "-np" "1" "test_all_pairwise")
set_tests_properties(test_all_pairwise PROPERTIES  _BACKTRACE_TRIPLES "I:/COMP5426/Assignment2/AllPairwiseComputation-MPI/CMakeLists.txt;25;add_test;I:/COMP5426/Assignment2/AllPairwiseComputation-MPI/CMakeLists.txt;0;")
add_test(test_distributed "mpirun" "-np" "2" "test_distributed")
set_tests_properties(test_distributed PROPERTIES  _BACKTRACE_TRIPLES "I:/COMP5426/Assignment2/AllPairwiseComputation-MPI/CMakeLists.txt;26;add_test;I:/COMP5426/Assignment2/AllPairwiseComputation-MPI/CMakeLists.txt;0;")
