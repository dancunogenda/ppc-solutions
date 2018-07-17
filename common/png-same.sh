#!/bin/sh

out=tmp/pngdiff-out
mkdir -p tmp || exit 1
if common/pngdiff "$1" "$2" "$3" /dev/null > "$out"; then
    rm -f "$out"
    exit 0
else
    cat "$out"
    rm -f "$out"
    echo
    echo "$2 and $3 should be identical with resolution $1"
    exit 1
fi
