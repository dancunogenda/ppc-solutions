#!/bin/bash

./mf-test || exit 1

files="a0 a1"
mkdir -p ../tmp || exit 1
for a in $files; do
    for w in 1 10; do
        ./pngmf $w ../data/$a.png ../tmp/pngmf-$a-$w-1.png ../tmp/pngmf-$a-$w-2.png || exit 1
    done
done
cd .. || exit 1
for a in $files; do
    for w in 1 10; do
        for x in 1 2; do
            common/png-same.sh 1 tmp/pngmf-$a-$w-$x.png correct/pngmf-$a-$w-$x.png || exit 1
            rm -f tmp/pngmf-$a-$w-$x.png
        done
    done
done

