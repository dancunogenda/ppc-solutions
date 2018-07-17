#!/bin/sh

xx="1 10 100 500 1000 1500"
for y in $xx; do
    for x in $xx; do
        ./cp-benchmark $y $x || exit 1
    done
done
