#!/bin/bash
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

set -e
#set -x

bin=$(realpath "`dirname "$0"`")
export PATH="${bin}:${PATH}"

[[ $# -ge 1 ]] && cd "$1"
${bin}/split-categorical2.py -f RANK data.spec data.csv rank &&
(cd rank0 &&
 ${bin}/split-categorical2.py -f MC data.spec data.csv ch &&
 for c in ch*/; do
	(cd $c &&
	 ${bin}/split-categorical2.py -f Byte data.spec data.csv byte/ &&
	 for b in byte/*/; do
		ln -rs "${bin}/dataset.mk" ${b}/Makefile || break
	 done &&
	 ${bin}/bounds-csv.py < data.csv > bounds.csv
	) || break
 done)
