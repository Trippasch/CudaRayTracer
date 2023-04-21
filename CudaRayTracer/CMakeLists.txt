if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found at ${CUDAToolkit_LIBRARY_ROOT}")
    message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA includes: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()

file(GLOB_RECURSE CXX_SRC src/*.cpp src/*.h)
file(GLOB_RECURSE CUDA_SRC src/*.cu src/*.cuh)

add_executable(${PROJECT_NAME} ${CXX_SRC} ${CUDA_SRC} ${CMAKE_SOURCE_DIR}/vendor/ImGui/backends/imgui_impl_glfw.cpp ${CMAKE_SOURCE_DIR}/vendor/ImGui/backends/imgui_impl_opengl3.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 86)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_GENERATE_CODE "arch=compute_86,code=[compute_86,sm_86]")

# Include Directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/GLFW/include ${CMAKE_SOURCE_DIR}/vendor/Glad/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/spdlog/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/stb)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/glm)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/ImGui)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

link_directories(${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(${PROJECT_NAME} glfw glad spdlog stb_image ImGui cudart)

target_compile_definitions(${PROJECT_NAME} PRIVATE "RT_DEBUG" "GLFW_INCLUDE_NONE" "_CRT_SECURE_NO_WARNINGS")

