#!/bin/sh

echo "Cleaning projects..."

echo "find . -name 'Makefile' -delete"
find . -name 'Makefile' -delete

echo "rm -rf ./**/bin"
rm -rf ./**/bin 

echo "rm -rf ./**/bin-int"
rm -rf ./**/bin-int

ERRORLEVEL=$?
if [ $ERRORLEVEL -ne 0 ]
then
    echo "Error: "$ERRORLEVEL && exit
fi
