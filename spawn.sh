#!/bin/bash

if [ $# -lt 2 ]; then
	echo "usage: $0 N CMD [ARGS...]" >&2
	exit 1
fi

n=$1
shift
for i in `seq 1 $n`; do
	"$@" &
	echo $!
done
