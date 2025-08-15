cmake_minimum_required(VERSION 3.15)
project(Cuda VERSION 1.0.0 LANGUAGES CXX CUDA)

# CUDA supports up to C++ 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 13.0
find_package(CUDAToolkit 13.0 EXACT REQUIRED)

# Just doing 50 series for now
set(CMAKE_CUDA_ARCHITECTURES "90")

# C++ optimization flags
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Ox>) # MVSC C++
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)  # Use -O3 for GCC/Clang C++
endif()

# CUDA optimization flags
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O2>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>)

# Expose global code area
include_directories(
    ${CMAKE_SOURCE_DIR}
)

# Subdirectories
add_subdirectory(SimpleFma)
