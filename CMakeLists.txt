cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

if(UNIX)
    # Linux-specific configuration
    message(STATUS "Linux Config")
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
elseif(WIN32)
    message(STATUS "Windows Config")
else()
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CudaRayTracer LANGUAGES C CXX CUDA)

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
endif()

if (NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
endif()

set(CUDAToolkit_VERSION "12.0" CACHE STRING "Minimum version of the CUDA Toolkit")
find_package(CUDAToolkit ${CUDAToolkit_VERSION} REQUIRED)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/vendor/GLFW)

# Glad
set(GLAD_DIR ${CMAKE_SOURCE_DIR}/vendor/Glad)
file(GLOB GLAD_SRC ${GLAD_DIR}/src/*.c)
add_library(glad STATIC ${GLAD_SRC})
target_include_directories(glad PRIVATE ${GLAD_DIR}/include)

# spdlog
set(SPDLOG_BUILD_EXAMPLE OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/vendor/spdlog)

# stb_image
set(STB_DIR ${CMAKE_SOURCE_DIR}/vendor/stb)
file(GLOB STB_SRC ${STB_DIR}/*.cpp)
add_library(stb_image STATIC ${STB_SRC})

# ImGui
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/ImGui)
file(GLOB IMGUI_SRC ${IMGUI_DIR}/*.cpp)
add_library(ImGui STATIC ${IMGUI_SRC})
target_include_directories(ImGui PRIVATE ${IMGUI_DIR})

add_subdirectory(${CMAKE_SOURCE_DIR}/CudaRayTracer)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

