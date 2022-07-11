project "CudaRayTracer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir "bin/%{cfg.buildcfg}"
    staticruntime "on"

    files {"src/Main.cpp", "src/Core/**.h", "src/Core/**.cpp" }

    includedirs {
        "src/**",
        -- "../vendor/imgui",
        "../vendor/spdlog/include",
        "../vendor/GLFW/include",
        "%{IncludeDir.Glad}",
        "%{IncludeDir.glm}"
    }

    defines {
        "GLFW_INCLUDE_NONE"
    }

    -- Add necessary build customization using standard Premake5
    -- This assumes we have installed Visual Studio integration for CUDA
    -- Here we set it to 11.6
    -- buildcustomizations "BuildCustomizations/CUDA 11.6"
    -- cudaPath "/usr/local/cuda-11.7"

    -- CUDA specific properties
    -- cudaFiles {                                 -- Files compiled by NVCC
    --     "CudaRayTracer/src/Cuda/*.cu"
    -- }

    -- cudaMaxRegCount "32"

    -- cudaCompilerOptions {
    --     "-t0",                                  -- Compile in parallel
    --     "-arch=sm_52", "-gencode=arch=compute_52,code=sm_52", "-gencode=arch=compute_60,code=sm_60",
    --     "-gencode arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
    --     "-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_80,code=sm_80",
    --     "-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_86,code=compute_86"
    -- }

    -- if os.target() == "linux" then
    --     linkoptions {"-L/usr/local/cuda-11.7/lib64 -lcudart"}
    -- end

    targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
    objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

    filter "system:windows"
        systemversion "latest"
        defines { "RT_PLATFORM_WINDOWS" }


        links {
            -- "ImGui",
            "GLFW",
            "Glad",
            "opengl32.lib"
        }

    filter "system:linux"
        systemversion "latest"

        links {
            -- "ImGui",
            "GLFW",
            "Glad",
            "GL",
        }

    filter "configurations:Debug"
        defines { "RT_DEBUG" }
        runtime "Debug"
        symbols "On"

    filter "configurations:Release"
        defines { "RT_RELEASE" }
        runtime "Release"
        optimize "On"
        symbols "On"
        -- cudaFastMath "On"                       -- Enable fast math for release

    filter "configurations:Dist"
        defines { "RT_DIST" }
        runtime "Release"
        optimize "On"
        -- cudaFastMath "On"                       -- Enable fast math for release
        symbols "Off"
