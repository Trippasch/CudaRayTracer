-- Include the premake5 CUDA module
require('premake5-cuda')

-- premake5.lua
workspace('CudaRayTracer')
architecture('x64')
configurations({ 'Debug', 'Release', 'Dist' })
startproject('CudaRayTracer')

outputdir = '%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}'

include('CudaRayTracerExternal.lua')
