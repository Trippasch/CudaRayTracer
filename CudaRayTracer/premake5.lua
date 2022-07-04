project "CudaRayTracer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    targetdir "bin/%{cfg.buildcfg}"
    staticruntime "on"

    files { "src/**.h", "src/**.cpp" }

    includedirs {
        "../vendor/imgui",
        "../vendor/spdlog/include",
        "../vendor/GLFW/include"
    }


    targetdir ("../bin/" .. outputdir .. "/%{prj.name}")
    objdir ("../bin-int/" .. outputdir .. "/%{prj.name}")

    filter "system:windows"
        systemversion "latest"
        defines { "RT_PLATFORM_WINDOWS" }


        links {
            -- "ImGui",
            "GLFW",
            "opengl32.lib"
        }

    filter "system:linux"
        systemversion "latest"

        links {
            -- "ImGui",
            "GLFW",
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

    filter "configurations:Dist"
        defines { "RT_DIST" }
        runtime "Release"
        optimize "On"
        symbols "Off"
