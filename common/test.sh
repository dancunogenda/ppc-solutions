#!/bin/bash

cd .. || exit 1
mkdir -p tmp || exit 1

for a in 0 1 2; do
    for b in a0 a1; do
        for c in a0 a1; do
            common/pngdiff $a data/$b.png data/$c.png tmp/pngdiff-$a-$b-$c.png > tmp/pngdiff-$a-$b-$c.txt
            echo "$?" > tmp/pngdiff-$a-$b-$c.ret
            test -e tmp/pngdiff-$a-$b-$c.txt || exit 1
            diff correct/pngdiff-$a-$b-$c.txt tmp/pngdiff-$a-$b-$c.txt || exit 1
            diff correct/pngdiff-$a-$b-$c.ret tmp/pngdiff-$a-$b-$c.ret || exit 1
            rm tmp/pngdiff-$a-$b-$c.{png,txt,ret}
        done
    done
done
