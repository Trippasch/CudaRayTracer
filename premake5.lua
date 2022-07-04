-- premake5.lua
workspace "CudaRayTracer"
    architecture "x64"
    configurations { "Debug", "Release", "Dist" }
    startproject "CudaRayTracer"

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "CudaRayTracerExternal.lua"