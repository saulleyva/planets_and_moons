cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Planets_and_Moons)

# -------------------------------------------------
# Configuration
# -------------------------------------------------
set(CMAKE_PREFIX_PATH "C:/libtorch118")
set(Torch_DIR "C:/libtorch118/share/cmake/Torch")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")

# Set CUDA architectures
if(NOT DEFINED CUDA_ARCHITECTURES)
  set(CUDA_ARCHITECTURES 86) # RTX 3070 is Ampere, compute capability 8.6
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# -------------------------------------------------
# Include directories
# -------------------------------------------------
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    src   # For header files in src/
)

# -------------------------------------------------
# Common source files
# -------------------------------------------------
set(COMMON_SRC_FILES
    src/model.cpp
    src/utils.cpp
)

# -------------------------------------------------
# Executable: Planets_and_Moons
# -------------------------------------------------
add_executable(Planets_and_Moons
    src/main.cpp
    ${COMMON_SRC_FILES}
)

target_link_libraries(Planets_and_Moons
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# -------------------------------------------------
# Executable: ProcessImages
# -------------------------------------------------
add_executable(ProcessImages
    scripts/process_images.cpp
)

target_link_libraries(ProcessImages
    ${OpenCV_LIBS}
)

target_include_directories(ProcessImages PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# -------------------------------------------------
# Executable: VisualizePredictions
# -------------------------------------------------
add_executable(VisualizePredictions
    scripts/visualize_predictions.cpp
    ${COMMON_SRC_FILES}
)

target_link_libraries(VisualizePredictions
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# -------------------------------------------------
# Post-build steps for Windows (MSVC)
# -------------------------------------------------
if (MSVC)
    function(copy_torch_dlls target_name)
        file(GLOB TORCH_DLLS "${CMAKE_PREFIX_PATH}/lib/*.dll")
        if(TORCH_DLLS)
            add_custom_command(TARGET ${target_name}
                               POST_BUILD
                               COMMAND ${CMAKE_COMMAND} -E copy_if_different
                               ${TORCH_DLLS}
                               $<TARGET_FILE_DIR:${target_name}>)
        else()
            message(WARNING "Torch DLLs not found at ${CMAKE_PREFIX_PATH}/lib. Please verify the path.")
        endif()
    endfunction()

    copy_torch_dlls(Planets_and_Moons)
    copy_torch_dlls(VisualizePredictions)
endif()
