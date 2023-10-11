#!/bin/bash

STAGES=( boost kjson kay gmp mpfr flint z3 hdf5 pipdeps smlp )

help() {
	echo "usage: $0 [-OPTS]"
	echo
	echo "Options:"
	echo "  -d      enable debug mode for this script"
	echo "  -h      print this help message"
	echo
	echo "Environment variables by this script to run programs:"
	echo "  CC      path to C compiler, current: '$CC', default: cc"
	echo "  CXX     path to C++ compiler, current: '$CXX', default: c++"
	echo "  NINJA   path to ninja, current: '$NINJA', default: ninja"
	echo "  PIP     path to pip, current: '$PIP', default: pip"
	echo "  PYTHON  path to python, current: '$PYTHON', default: python3"
	echo "  PYTHON_CONFIG  path to python-config utility, current: '$PYTHON_CONFIG', default: auto-detect"
	echo
	echo "Other environment variables may influence the operation of the above programs."
	echo
	echo "This script provides a way for an automated installation of SMLP"
	echo "into a Python virtual environment. Note that this is not the preferred"
	echo "way of installing SMLP, it merely provides a convenience method to get"
	echo "started with SMLP."
	echo
	echo "The preferred way of installing and using SMLP is described in the"
	echo "corresponding README files distributed with the package."
}

error() {
	echo "error: $1" >&2
	exit 1
}

require_sys_programs() {
	local p r
	for p in "$@"; do
		r=$(command -V $p) || error "program '$p' not found"
	done
}

prepare() {
	mkdir -p $TGT/.smlp-quickinstall &&
	return 0
}

get_boost() {
	[[ -f boost_1_82_0.tar.gz ]] ||
	wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz
}

install_boost() {
	local gv pv pc inc
	gv=$($CXX -v |& grep 'gcc version' | awk '{print $3}') &&
	pv=$($PYTHON -V | tr . ' ' | awk '{print $2 "." $3}') &&
	inc=$($PYTHON_CONFIG --includes) &&
	rm -rf boost_1_82_0 &&
	tar xfz boost_1_82_0.tar.gz &&
	cd boost_1_82_0 &&
	./bootstrap.sh --prefix=$TGT --with-libraries=python \
		--with-python=$PYTHON --with-python-version=$pv \
		--with-python-root=$VIRTUAL_ENV &&
	sed -ri "s\
,\busing gcc ;\
,using gcc : $gv : \"$(command -v $CXX)\" : <cflags>\"$inc\" <cxxflags>\"$inc\" ;\
," \
		project-config.jam &&
	./b2 -j`nproc` &&
	./b2 install &&
	cd .. && rm -rf boost_1_82_0
}

get_kjson() {
	[[ -f kjson-v0.2.1.tar.gz ]] ||
	wget -O kjson-v0.2.1.tar.gz https://github.com/fbrausse/kjson/archive/refs/tags/v0.2.1.tar.gz
}

install_kjson() {
	rm -rf kjson-0.2.1 &&
	tar xfz kjson-v0.2.1.tar.gz &&
	cd kjson-0.2.1 &&
	env PATH=/sbin:$PATH make DESTDIR=$TGT install &&
	cd .. && rm -rf kjson-0.2.1
}

get_kay() {
	[[ -f kay.zip ]] ||
	wget -O kay.zip https://github.com/fbrausse/kay/archive/refs/heads/master.zip
}

install_kay() {
	unzip kay.zip &&
	rm -rf kay &&
	mv kay-master kay &&
	cd kay &&
	make DESTDIR=$TGT install
	cd .. && rm -rf kay
}

get_gmp() {
	[[ -f gmp-6.3.0.tar.xz ]] ||
	wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
}

install_gmp() {
	rm -rf gmp-6.3.0 &&
	tar xfJ gmp-6.3.0.tar.xz &&
	cd gmp-6.3.0 &&
	./configure --prefix=$TGT --enable-cxx CC="$CC" CXX="$CXX" &&
	make -j`nproc` &&
	make install &&
	cd .. && rm -rf gmp-6.3.0
}

get_mpfr() {
	[[ -f mpfr-4.2.1.tar.gz ]] ||
	wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.gz
}

install_mpfr() {
	rm -rf mpfr-4.2.1 &&
	tar xfz mpfr-4.2.1.tar.gz &&
	cd mpfr-4.2.1 &&
	./configure --prefix=$TGT --with-gmp=$TGT CC=$CC &&
	make -j`nproc` &&
	make install &&
	cd .. && rm -rf mpfr-4.2.1
}

get_pipdeps() {
	:
}

install_pipdeps() {
	if ! command -v meson >/dev/null; then
		echo "Installing meson using pip..."
		$PIP install meson || error "installing meson failed"
	fi
}

get_flint() {
	[[ -f flint-2.8.5.tar.gz ]] ||
	wget https://flintlib.org/flint-2.8.5.tar.gz
}

install_flint() {
	rm -rf flint-2.8.5 &&
	tar xfz flint-2.8.5.tar.gz &&
	cd flint-2.8.5 &&
	./configure --prefix=$TGT --with-gmp=$TGT --with-mpfr=$TGT --disable-cxx \
		CC=$CC CXX=$CXX LDCONFIG=/sbin/ldconfig \
		CFLAGS="-I$TGT/include -L$TGT/lib64" CXXFLAGS="-I$TGT/include -L$TGT/lib64" &&
	make -j`nproc` &&
	make install &&
	cd .. && rm -rf flint-2.8.5
}

get_z3() {
	[[ -f z3-4.11.2.tar.gz ]] ||
	wget https://github.com/Z3Prover/z3/archive/refs/tags/z3-4.11.2.tar.gz
}

install_z3() {
	rm -rf z3-z3-4.11.2 &&
	tar xfz z3-4.11.2.tar.gz &&
	cd z3-z3-4.11.2 &&
	mkdir build &&
	cd build &&
	env CC=$CC CXX=$CXX CFLAGS="-I$TGT/include" CXXFLAGS="-I$TGT/include" \
		LDFLAGS="-L$TGT/lib64" cmake -G Ninja \
		-DCMAKE_MAKE_PROGRAM="$NINJA" \
		-DCMAKE_INSTALL_PREFIX=$TGT \
		-DZ3_USE_LIB_GMP=yes \
		-DZ3_ENABLE_EXAMPLE_TARGETS=OFF \
		-DZ3_BUILD_DOCUMENTATION=no \
		-DZ3_BUILD_PYTHON_BINDINGS=yes \
		-DZ3_BUILD_JAVA_BINDINGS=no \
		-DPYTHON_EXECUTABLE=$(command -v $PYTHON) \
		.. &&
	$NINJA -j`nproc` &&
	$NINJA install &&
	cd ../.. && rm -rf z3-z3-4.11.2
}

get_hdf5() {
	local MAJOR_P=hdf5-1.14
	local MY_P=hdf5-1.14.2
	[[ -f ${MY_P}.tar.bz2 ]] ||
	wget https://www.hdfgroup.org/ftp/HDF5/releases/${MAJOR_P}/${MY_P}/src/${MY_P}.tar.bz2
}

install_hdf5() {
	rm -rf hdf5-1.14.2 &&
	tar xfj hdf5-1.14.2.tar.bz2 &&
	cd hdf5-1.14.2 &&
	./configure CC=$CC CXX=$CXX --enable-cxx --prefix=$TGT &&
	make -j`nproc` &&
	make install &&
	cd .. && rm -rf hdf5-1.14.2
}

get_smlp() {
	: # nothing to see, please move along
}

install_smlp() {
	local pv lib
	pv=$($PYTHON -V | tr . ' ' | awk '{print $2 "." $3}') &&
	lib=$VIRTUAL_ENV/lib/python$pv/site-packages/smlp/libsmlp$($PYTHON_CONFIG --extension-suffix) &&
	echo "please run inside smlp/utils/poly:"
	echo \
	"env BOOST_ROOT=$TGT PKG_CONFIG_PATH=$TGT/lib64/pkgconfig CC=$CC CXX=$CXX " \
	"meson setup --wipe build -D{kay,kjson,flint,hdf5}-prefix=$TGT --prefix $VIRTUAL_ENV && " \
	"$NINJA -C build install && " \
	"patchelf --set-rpath '\$ORIGIN/../../../../lib64' $lib && " \
	"patchelf --set-rpath '\$ORIGIN/../lib64' $VIRTUAL_ENV/bin/smlp"
}

do_stage() {
	echo "Stage '$1'"

	[[ -f $TGT/.got_$1 ]] || get_$1 || error "getting $1"
	touch $TGT/.got_$1

	[[ -f $TGT/.installed_$1 ]] || install_$1 || error "installing $1"
	touch $TGT/.installed_$1
}

while [ $# -gt 0 ]; do
	case "$1" in
	-d) set -x ;;
	-h) help; exit 0 ;;
	-?)
		error "unknown parameter '$1'"
		;;
	*) break ;;
	esac
	shift
done

if [ $# -ne 0 ]; then
	error "unrecognized additional parameters: $*"
fi

[[ -n "$VIRTUAL_ENV" ]] || error "environment variable VIRTUAL_ENV empty, please activate the python venv to install SMLP to"
TGT=$VIRTUAL_ENV

CC=${CC:-cc}
CXX=${CXX:-c++}
NINJA=${NINJA:-ninja}
PYTHON=${PYTHON:-python3}
PIP=${PIP:-pip}

echo "Settings from environment:"
echo "CC           : $CC"
echo "CXX          : $CXX"
echo "NINJA        : $NINJA"
echo "PYTHON       : $PYTHON"
echo "PIP          : $PIP"

require_sys_programs \
	command test [ [[ mkdir uname tr awk grep env touch rm \
	"$CC" "$CXX" "$NINJA" "$PYTHON" "$PIP"

default_py_cfg() {
	local pp pr pc
	pp=$(command -v $PYTHON) &&
	pr=$(realpath "$pp") &&
	pc=$pr-config &&
	[[ -x "$pc" ]] && echo "$pc"
}

PYTHON_CONFIG=${PYTHON_CONFIG:-$(default_py_cfg)} ||
error "cannot auto-detect python-config tool, please set the PYTHON_CONFIG environment variable"

echo "PYTHON_CONFIG: $PYTHON_CONFIG"

require_sys_programs \
	tar gzip unzip bzip2 xz install make nproc wget cmake \
	"$PYTHON_CONFIG"

[[ "$(uname)" = Linux ]] || error "unsupported OS"

#PIP+=" --python=$(command -v $PYTHON)"
echo -n "Testing integrity of pip installation... "
$PIP check || error "pip installation seems to be broken"

prepare || error "preparing $TGT directory for installation"
for stage in ${STAGES[@]}; do
	(cd $TGT/.smlp-quickinstall && do_stage $stage) ||
	error "stage '$stage' could not be completed"
done
