#!/bin/sh

if [ "$1" = "" ]; then
    xx="100 200 500 1000 2000"
    kk="1 2 5 10"
else
    xx="100 200 500 1000 2000 4000"
    kk="1 2 5 10 20 50 100"
fi

for x in $xx; do
    for k in $kk; do
        ./mf-benchmark $x $x $k || exit 1
    done
done
