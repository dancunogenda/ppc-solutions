#!/bin/sh

for y in 1 10 100 200 400; do
    for x in 1 10 100 200 400; do
        ./is-benchmark $y $x || exit 1
    done
done
