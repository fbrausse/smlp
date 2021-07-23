#!/bin/bash
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

usage() {
	echo "usage: $0 [-tv] [FILE]" >&2
	exit 1
}

vars=()
command -v unbuffer >/dev/null
raw=$?
while [ $# -gt 0 ]; do
	case "$1" in
	-r) raw=1 ;;
	-v) vars+=( -v verb=1 ) ;;
	--) shift; break ;;
	-*) usage ;;
	*) break ;;
	esac
	shift
done
if [ $# -gt 1 ]; then
	usage
fi

SCRIPT=$(dirname "$0")/filter-trace.awk

vars+=( -v py=$(dirname "$0")/filter-trace-flt )

if ((raw)); then
	exec awk -F, "${vars[@]}" -f "${SCRIPT}" "$@"
elif [ $# -eq 0 ]; then
	unbuffer -p awk -F, "${vars[@]}" -f "${SCRIPT}" | tr , '\t'
else
	awk -F, "${vars[@]}" -f "${SCRIPT}" "$@" | tr , '\t'
fi
