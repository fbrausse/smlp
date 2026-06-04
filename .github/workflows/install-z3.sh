#!/bin/bash

set -e -x

source "$(dirname "${BASH_SOURCE[0]}")"/install-gmp.sh

P=z3

eval ${P}_V=${V:-4.8.12}					# version
eval ${P}_F=${F:-$P-\$`echo \${P}_V`.tar.xz}			# source archive name
eval ${P}_W=${W:-$HOME/$P}					# workdir root
eval ${P}_S=${S:-\$`echo ${P}_W`/$P-$P-\$`echo \${P}_V`}	# source dir
eval ${P}_R=${R:-\$`echo ${P}_S`/prefix}			# install prefix

export PYTHON=python3.11
export CPPFLAGS+=" -I$gmp_R/include"
export LDFLAGS+=" -L$gmp_R/lib -Wl,-rpath,$gmp_R/lib"
export PKG_CONFIG_PATH=$gmp_PKG_CONFIG_PATH:$PKG_CONFIG_PATH

run() {
	eval local V=\${${P}_V}
	eval local F=\${${P}_F}
	eval local W=\${${P}_W}
	eval local S=\${${P}_S}
	eval local R=\${${P}_R}
	local T=$1
	local B=build

	mkdir -p $W
	cd $W

	# get
	[[ -f $W/.get ]] || {
		wget -O $F https://github.com/Z3Prover/z3/archive/refs/tags/$P-$V.tar.gz
		touch $W/.get
	}

	# unpack
	[[ -f $W/.unpack ]] || {
		tar xfz $F
		touch $W/.unpack
	}

	cd $S

	# configure
	[[ -f $W/.configure ]] || {
		[ -d $P -a -f $B/Makefile ] || ./configure --gmp --build=$B --prefix=$R
		touch $W/.configure
	}

	# build/install
	[[ -f $W/.$T ]] || {
		sed -i 's/@//' $B/Makefile	# want to see commands
		env LIBRARY_PATH=$gmp_R/lib make -C $B -j`nproc` $T
		touch $W/.$T
	}
}

if [ $# -eq 1 ]; then run $1; fi
