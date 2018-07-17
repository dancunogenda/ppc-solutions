#!/bin/bash

if [ "$1" = "" ]; then
    for n in {1,10}000000; do
        ./so-test $n || exit 1
    done
else
    for n in {1,10,100,200}000000; do
        ./so-test $n || exit 1
    done
fi
