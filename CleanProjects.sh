#!/bin/sh

echo "Cleaning projects..."

echo "find . -name 'Makefile' -delete"
find . -name 'Makefile' -delete
echo "rm -rf bin bin-int"
rm -rf bin bin-int

ERRORLEVEL=$?
if [ $ERRORLEVEL -ne 0 ]
then
    echo "Error: "$ERRORLEVEL && exit
fi
