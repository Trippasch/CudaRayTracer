# CudaRayTracer
CudaRayTracer is a simple real-time brute-force path tracer based on the [Ray Tracing in One Weekend Series](https://raytracing.github.io/) accelerated with [CUDA](https://developer.nvidia.com/cuda-downloads).
## Getting Started
<ins>**1. Download the CUDA toolkit:**</ins>

Start by downloading the CUDA toolkit, if you haven't done that already, from [here](https://developer.nvidia.com/cuda-downloads).

<ins>**2. Downloading the repository:**</ins>

Then clone the repository with:
```
git clone --recurse-submodules git@github.com:Trippasch/CudaRayTracer.git
```
If the repository was cloned non-recursively previously, use ```git submodule update --init``` to clone the necessary submodules.

<ins>**3. Generate Project files:**</ins>

* Premake

For Windows run the [GenerateProjects.bat](https://github.com/Trippasch/CudaRayTracer/blob/master/scripts/windows/GenerateProjects.bat) file. This will generate the visual studio (2022) solution to compile and run the project.

For Linux run the [GenerateProjects.sh](https://github.com/Trippasch/CudaRayTracer/blob/master/scripts/linux/GenerateProjects.sh) file. This will generate makefiles to compile and run the project. The compiler used inside the script is LLVM's clang but you can change it to gcc as well.

* CMake

For Windows run the command: ```cmake -S . -B build``` on the project's root folder to generate the build files.
Then, you can either run the Visual Studio solution to build the project or run ```cmake --build .\build --config=release -j``` to build the application on release mode.

For Linux run the command: ```cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/release/``` and ```cmake --build build/release -j``` to build the application on release mode.

Remember to run the application from the project's root folder to load the correct paths.

<ins>**4. Clean Project files:**</ins>

For Windows run the [CleanProjects.bat](https://github.com/Trippasch/CudaRayTracer/blob/master/scripts/windows/CleanProjects.bat) file.

For Linux run the [CleanProjects.sh](https://github.com/Trippasch/CudaRayTracer/blob/master/scripts/linux/CleanProjects.sh) file.

## Dependencies
The project uses the following dependencies:
  * [CUDA](https://developer.nvidia.com/cuda-downloads) for GPU acceleration.
  * [ImGui](https://github.com/ocornut/imgui) for creating graphical user interfaces (GUIs).
  * [GLM](https://github.com/g-truc/glm) for vector math.
  * [GLFW](https://www.glfw.org/) for creating windows, contexts and surfaces.
  * [Glad](https://glad.dav1d.de/) for generating OpenGL functions.
  * [spdlog](https://github.com/gabime/spdlog) for logging.
  * [stb](https://github.com/nothings/stb) for image loading.
  * [CMake](https://cmake.org/) for building the project.
  * [Premake](https://premake.github.io/) for building the project.
