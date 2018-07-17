#!/bin/sh

for i in 1 2 4 8; do
    echo "OMP_NUM_THREADS=$i"
    echo
    OMP_NUM_THREADS=$i "$@" || exit 1
    echo
done
echo "DEFAULT"
echo
"$@" || exit 1
echo
