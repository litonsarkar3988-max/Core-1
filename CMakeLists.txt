cmake_minimum_required(VERSION 3.20)
project(TitanCore LANGUAGES CXX CUDA)

# -------------------------------
# 1. C++ Standard
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------
# 2. Torch / LibTorch Path
# -------------------------------
# ধরো LibTorch system-wide installed
find_package(Torch REQUIRED)
message(STATUS "Found LibTorch at ${Torch_DIR}")

# -------------------------------
# 3. CUDA Support
# -------------------------------
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------
# 4. Source Files
# -------------------------------
file(GLOB_RECURSE CORE_SRC
    "${CMAKE_SOURCE_DIR}/titancore/core/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/core/model/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/core/transformer/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/core/quant/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/audio/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/vision/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/retrieval/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/distributed/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/safety/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/api/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/monitoring/*.cpp"
    "${CMAKE_SOURCE_DIR}/titancore/tools/*.cpp"
)

# -------------------------------
# 5. Executable
# -------------------------------
add_executable(titancore main.cpp ${CORE_SRC})

# -------------------------------
# 6. Include Directories
# -------------------------------
target_include_directories(titancore PRIVATE
    "${CMAKE_SOURCE_DIR}/titancore/core"
    "${CMAKE_SOURCE_DIR}/titancore/core/model"
    "${CMAKE_SOURCE_DIR}/titancore/core/transformer"
    "${CMAKE_SOURCE_DIR}/titancore/core/quant"
    "${CMAKE_SOURCE_DIR}/titancore/audio"
    "${CMAKE_SOURCE_DIR}/titancore/vision"
    "${CMAKE_SOURCE_DIR}/titancore/retrieval"
    "${CMAKE_SOURCE_DIR}/titancore/distributed"
    "${CMAKE_SOURCE_DIR}/titancore/safety"
    "${CMAKE_SOURCE_DIR}/titancore/api"
    "${CMAKE_SOURCE_DIR}/titancore/monitoring"
    "${CMAKE_SOURCE_DIR}/titancore/tools"
)

# -------------------------------
# 7. Link Libraries
# -------------------------------
target_link_libraries(titancore "${TORCH_LIBRARIES}")
set_property(TARGET titancore PROPERTY CXX_STANDARD 17)

# -------------------------------
# 8. CUDA Properties
# -------------------------------
set_target_properties(titancore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
