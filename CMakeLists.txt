cmake_minimum_required(VERSION 3.16)
project(miniptx LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_library(miniptx_core src/lexer.cpp src/parser.cpp src/ast.cpp src/ir.cpp src/ptx_emitter.cpp)
target_include_directories(miniptx_core PUBLIC include)

add_executable(miniptxc src/main.cpp)
target_link_libraries(miniptxc miniptx_core)

enable_testing()

add_executable(test_lexer tests/test_lexer.cpp)
target_link_libraries(test_lexer miniptx_core)
add_test(NAME lexer COMMAND test_lexer)

add_executable(test_phase1 tests/test_phase1.cpp)
target_link_libraries(test_phase1 miniptx_core)
add_test(NAME phase1 COMMAND test_phase1)

add_executable(test_phase2 tests/test_phase2.cpp)
target_link_libraries(test_phase2 miniptx_core)
add_test(NAME phase2 COMMAND test_phase2)

# Optional: GPU runtime loader (needs CUDA toolkit; build on the DGX)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  add_executable(run_ptx runtime/run_ptx.cpp)
  target_link_libraries(run_ptx CUDA::cuda_driver)
endif()
