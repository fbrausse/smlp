#!/bin/bash

set -e -x

loc=$(dirname "${BASH_SOURCE[0]}")	# directory of this script
project=$1				# project source directory
os=$2					# Github runner OS identifier
tgt=$3					# prefix to install packages to

case "$os" in
ubuntu-*)
	dnf -y install \
		wget \
		gcc-toolset-13

	source /opt/rh/gcc-toolset-13/enable

	export R=$tgt
	export HOST=$(uname -m)-pc-linux-gnu

	source $loc/install-gmp.sh install

	source $loc/install-z3.sh install

	;;
*)
	echo "error: OS unknown: $os" >&2
	exit 1
	;;
esac
