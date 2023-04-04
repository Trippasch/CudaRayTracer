@echo off

del /S /Q "..\..\CudaRayTracer.sln"
rmdir /S /Q "..\..\bin"
rmdir /S /Q "..\..\bin-int"
del /S /Q "..\..\CudaRayTracer\CudaRayTracer.vcxproj"
del /S /Q "..\..\CudaRayTracer\CudaRayTracer.vcxproj.user"
del /S /Q "..\..\CudaRayTracer\CudaRayTracer.vcxproj.filters"

del /S /Q "..\..\vendor\ImGui\ImGui.vcxproj"
rmdir /S /Q "..\..\vendor\ImGui\bin"
rmdir /S /Q "..\..\vendor\ImGui\bin-int"

del /S /Q "..\..\vendor\Glad\Glad.vcxproj"
del /S /Q "..\..\vendor\Glad\Glad.vcxproj.filters"
rmdir /S /Q "..\..\vendor\Glad\bin"
rmdir /S /Q "..\..\vendor\Glad\bin-int"

del /S /Q "..\..\vendor\GLFW\glfw.vcxproj"
del /S /Q "..\..\vendor\GLFW\glfw.vcxproj.filters"
rmdir /S /Q "..\..\vendor\GLFW\bin"
rmdir /S /Q "..\..\vendor\GLFW\bin-int"

pause
