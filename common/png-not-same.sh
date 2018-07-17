#!/bin/sh

out=tmp/pngdiff-out
mkdir -p tmp || exit 1
if common/pngdiff "$1" "$2" "$3" /dev/null > "$out"; then
    echo "$2 and $3 should not be identical with resolution $1"
    rm -f "$out"
    exit 1
else
    rm -f "$out"
    exit 0
fi
