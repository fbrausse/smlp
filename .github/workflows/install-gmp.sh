#!/bin/bash

set -e -x

P=gmp

eval ${P}_V=${V:-6.3.0}					# version
eval ${P}_F=${F:-$P-\$`echo \${P}_V`.tar.xz}		# source archive name
eval ${P}_W=${W:-$HOME/$P}				# workdir root
eval ${P}_S=${S:-\$`echo ${P}_W`/$P-\$`echo \${P}_V`}	# source dir
eval ${P}_R=${R:-\$`echo ${P}_S`/prefix}		# install prefix

#eval ${P}_CPPFLAGS=-I\$`echo ${P}_R`/include
#eval ${P}_LDFLAGS=-L\$`echo ${P}_R`/lib
eval ${P}_PKG_CONFIG_PATH=\$`echo ${P}_R`/lib/pkgconfig

run() {
	eval local V=\${${P}_V}
	eval local F=\${${P}_F}
	eval local W=\${${P}_W}
	eval local S=\${${P}_S}
	eval local R=\${${P}_R}
	local T=$1

	mkdir -p $W
	cd $W

	# get
	[[ -f $W/.get ]] || {
		wget -O $F https://ftp.gnu.org/gnu/gmp/$F
		touch $W/.get
	}

	# unpack
	[[ -f $W/.unpack ]] || {
		tar xfJ $F
		touch $W/.unpack
	}

	cd $S

	# configure
	[[ -f $W/.configure ]] || {
		[ -d $P -a -f Makefile ] || ./configure --enable-cxx --prefix=$R --host=$HOST
		touch $W/.configure
	}

	# build/install
	[[ -f $W/.$T ]] || {
		make -j`nproc` $T
		touch $W/.$T
	}
}

if [ $# -eq 1 ]; then run $1; fi
