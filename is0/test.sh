#!/bin/bash

./is-test || exit 1

if [ "$1" = "" ]; then
    files="b0 b1"
else
    files="a0 a1 b0 b1"
fi

mkdir -p ../tmp || exit 1
for a in $files; do
    ./pngsegment ../data/$a.png ../tmp/pngsegment-$a-1.png ../tmp/pngsegment-$a-2.png || exit 1
done
cd .. || exit 1
for a in $files; do
    for x in 1 2; do
        common/png-same.sh 1 tmp/pngsegment-$a-$x.png correct/pngsegment-$a-$x.png || exit 1
        rm -f tmp/pngsegment-$a-$x.png
    done
done

