cmake_minimum_required(VERSION 3.20)
project(TitanCore LANGUAGES CXX CUDA)

# -------------------------------
# 1. C++ Standard & CUDA Architecture
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "80;86;90") # A100/H100 support

# -------------------------------
# 2. Find Dependencies (Torch, CUDA, MPI, NCCL)
# -------------------------------
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)

# NCCL Library
find_library(NCCL_LIBRARY nccl REQUIRED)

message(STATUS "Found LibTorch at ${Torch_DIR}")
message(STATUS "Found CUDA at ${CUDA_TOOLKIT_ROOT_DIR}")

# -------------------------------
# 3. Source Files (Using Core-1/ structure)
# -------------------------------
set(SOURCES
    main.cpp
    Core-1/core/model/gpt.cpp
    Core-1/core/dataloader/dataset.cpp
    Core-1/distributed/nccl.cpp
    Core-1/distributed/fsdp.cpp
    Core-1/core/optimizer/zero.cpp
    Core-1/core/model/kv_cache_paged.cpp
    Core-1/safety/moderation.cpp
    Core-1/logging/audit.cpp
)

# -------------------------------
# 4. Executable
# -------------------------------
add_executable(titancore ${SOURCES})

# -------------------------------
# 5. Include Directories
# -------------------------------
target_include_directories(titancore PRIVATE
    ${CMAKE_SOURCE_DIR}                
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
)

# -------------------------------
# 6. Link Libraries
# -------------------------------
target_link_libraries(titancore 
    ${TORCH_LIBRARIES} 
    ${CUDA_LIBRARIES}
    ${MPI_LIBRARIES}
    ${NCCL_LIBRARY}
)

# -------------------------------
# 7. CUDA & Compiler Properties for 1T Model
# -------------------------------
set_target_properties(titancore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(titancore PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
)
