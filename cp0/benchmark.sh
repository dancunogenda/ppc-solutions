#!/bin/sh

if [ "$1" = "" ]; then
    xx="1 10 100 1000 2000 4000"
else
    xx="1 10 100 1000 2000 4000 6000 8000"
fi
for y in $xx; do
    for x in $xx; do
        ./cp-benchmark $y $x || exit 1
    done
done
