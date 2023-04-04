@echo off

pushd %~dp0\..\..\
vendor\premake\windows\premake5.exe vs2022
popd

pause
