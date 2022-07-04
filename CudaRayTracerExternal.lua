-- CudaRayTracerExternal.lua

IncludeDir = {}
IncludeDir["glm"] = "vendor/glm"

group "Dependencies"
    -- include "vendor/imgui"
    include "vendor/GLFW"
group ""

include "CudaRayTracer"
