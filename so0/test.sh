#!/bin/bash

for n in {1,10,100,1000}000; do
    ./so-test $n || exit 1
done
