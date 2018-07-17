#!/bin/bash

error=0.0001
for y in 2 5 10 20 50 100 200 500; do
    for x in 2 5 10 20 50 100 200 500 1000 2000; do
        ./cp-test $error $y $x || exit 1
    done
done
for a in 2 5 10 20 50 100; do
    ./cp-test $error $a 10000 || exit 1
    ./cp-test $error 2000 $a || exit 1
done

files="a0 a1"
mkdir -p ../tmp || exit 1
for a in $files; do
    ./pngcorrelate ../data/$a.png ../tmp/pngcorrelate-$a-1.png ../tmp/pngcorrelate-$a-2.png || exit 1
done
cd .. || exit 1
for a in $files; do
    for x in 1 2; do
        common/png-same.sh 1 tmp/pngcorrelate-$a-$x.png correct/pngcorrelate-$a-$x.png || exit 1
        rm -f tmp/pngcorrelate-$a-$x.png
    done
done

