#!/bin/sh

echo "premake5 gmake2 --cc=gcc"
echo "Generating projects..."
vendor/premake/linux/premake5 gmake2 --cc=gcc

ERRORLEVEL=$?
if [ $ERRORLEVEL -ne 0 ]
then
    echo "Error: "$ERRORLEVEL && exit
fi
