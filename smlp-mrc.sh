#!/bin/bash
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

usage() {
	printf "\
%s [-OPTS] COMMAND [ARGS...]

Common options [defaults]:
  -d           debug this script
  -h           print this help message
  -k           keep created files / directories on errors
  -j JOBS      run N jobs in parallel; if JOBS is 0 determine via lscpu(1) [1]
  -t TGT       path to target instance directory [derived from basename of SRC
               if given, otherwise '.']

Options used by 'prepare' stage:
  -i SRC       path to source data set ending in .csv
  -s SPEC      path to .spec file describing the target instance

Options used by 'train' stage [defaults replicated from src/defaults.mk]:
  -b BATCH     batch size [32]
  -e EPOCHS    number of epochs to use for training [30]
  -f FILTER    percentile of lowest OBJT values to ignore for training [0]
  -l LAYERS    layer specification [2,1]
  -o OBJT      objective function [RESP]
  -r RESP      response features [delta]
  -s SEED      random number seed for numpy and tensorflow [1234]

Options used by 'search' and 'collect' stages
[defaults replicated from src/defaults.mk]:
  -c COFF      center threshold offset from threshold [0.05]
  -n N         restrict to maximum of N safe regions [100]
  -L TLO       minimum threshold to consider [0.00]
  -H THI       maximum threshold to consider [0.90]

Commands [defaults]:
  run [STAGE]  execute all stages up to STAGE [collect]

Stages (in dependency order):
  prepare      use SRC to setup a fresh TGT instance
  train        train NN models according to prepared TGT instance
  search       determine safety thresholds for TGT instance
  collect      collect shared safe regions

The following variables affect this script and the ones it calls [defaults]:

  MAKE         path to the GNU make utility [gmake]
  PATH         ':'-separated list of paths to search for executables
  PYTHONPATH   ':'-separated list of paths to search for python imports
" "$0" >&2
	exit $1
}

error() {
	printf "error: %s\n" "$*" >&2
}

info() {
	printf "%s\n" "$@" >&2
}

# requires global variables to be set:
#  bin tgt
# usage: stage_01-prepare SRC
#  SRC: path to source.csv.bz2
run_stage_prepare() {
	[[ -f "${src}" ]] || {
		error "SRC '${src}': not a regular file"
		return 2
	}
	[[ -f "${spec}" ]] || {
		error "SPEC '${spec}': not a regular file"
		return 2
	}
	info "creating instance directory '${tgt}'..."
	[[ -e "${tgt}" ]] && {
		error "TGT '${tgt}': already exists, aborting"
		return 3
	}
	mkdir -p "${tgt}" && files+=( "${tgt}" ) &&
	info "linking SRC '${src}' to '${tgt}/data.csv'..." &&
	ln -rs "${src}" "${tgt}"/data.csv && files+=( "${tgt}"/data.csv ) &&
	info "linking SPEC '${spec}' to '${tgt}/data.spec'..." &&
	ln -rs "${spec}" "${tgt}"/data.spec && files+=( "${tgt}"/data.spec ) &&
	info "preparing directory structure split on RANK, CH, Byte" &&
	"${bin}"/mrc-prep-dirs.sh "${tgt}" && files+=( `find "${tgt}"/rank0` `find "${tgt}/rank1"` ) #|| return 4
	for r in "${tgt}"/rank0; do
		ln -rs "${bin}"/mrc-rank.mk "${r}"/Makefile && files+=( "${r}"/Makefile ) ||
		return $?
		for c in "${r}"/ch{0,1}; do
			ln -rs "${bin}"/mrc-ch.mk "${c}"/Makefile && files+=( "${c}"/Makefile ) &&
			cp "${c}/byte/0/data.spec" "${c}/byte" && files+=( "${c}/byte/data.spec" ) ||
			return $?
			for b in "${c}"/byte/{0..7}; do
				rm "${b}/data.spec" &&
				ln -rs "${c}/byte/data.spec" "${b}" ||
				return $?
			done
		done
	done
	ln -rs "${bin}"/mrc-params.mk "${tgt}"/params.mk && files+=( "${tgt}"/params.mk )
}

make() {
	command ${MAKE} -j ${cpus} -C "${tgt}" "${args[@]}" "$@"
}

run_stage_train() {
	make -C rank0 train
}

run_stage_search() {
	make -C rank0 search
}

run_stage_collect() {
	make -C rank0 collect
}

clean_stage() {
	local st=$1
	shift
	local -a files=( "$@" )
	if ((!keep)); then
		[[ $# -eq 0 ]] || rm -vd "${files[@]}"
	fi
}

STAGES=(
	prepare
	train
	search
	collect
)

stage() {
	local st=$1
	[[ -n "${st}" ]] &&
	[[ 0 -le "${st}" ]] &&
	[[ "${st}" -lt ${#STAGES[*]} ]] || {
		error "invalid stage '${st}'; must be in 0,...,`bc <<< ${#STAGES[*]}-1`"
		return 1
	}
	local sdesc=${STAGES[st]}
	local sentinel=${st}-${sdesc}
	shift
	if [ -f "${tgt}/.${sentinel}" ]
	then
		info \
			"directory '${tgt}' appears to already be ${sdesc}'d." \
			"If not, please remove '${tgt}/.${sentinel}' and re-run this command."
	else
		local -a files
		info \
			"executing stage ${st}: ${sdesc}" &&
		run_stage_${sdesc} "$@" &&
		touch "${tgt}/.${sentinel}" &&
		info "successfully executed stage ${st}: ${sdesc}" || {
			local r=$?
			error "running stage ${st}: ${sdesc}"
			clean_stage ${st} "${files[@]}" ||
				error "cleaning stage ${st}: ${sdesc} after error $?; aborting"
			return $r
		}
	fi
}

run() {
	local i
	local st
	if [ $# -eq 0 ]; then
		st=`bc <<< ${#STAGES[*]}-1`
	elif [ $# -eq 1 ]; then
		for i in $(seq 0 `bc <<< ${#STAGES[*]}-1`); do
			[[ "${STAGES[i]}" = "$1" ]] && st=$i && break
		done
		[[ -z "${st}" ]] && {
			error "unknown stage '$1'"
			return 1
		}
	else
		error "command 'run' only accepts 0 or 1 parameters"
		return 1
	fi
	for i in `seq 0 ${st}`; do
		stage $i || {
			local r=$?
			local rem=`seq $((i+1)) ${st} | tr '\n' ,`
			[[ -n "${rem}" ]] &&
			error "skipping stages ${rem} due to previous error $r"
			return $((2+i))
		}
	done
}

tgt_from_file() {
	if [ -z "$1" ]; then echo "."; return 0; fi
	local fname=`basename "$1" .csv`
	[[ -z "`tr -d '[:alnum:]_-' <<< "${fname}"`" ]] || {
		err "FNAME must consist of characters matching '[[:alnum:]_-]'"
		return 1
	}
	sed -r 's#([^/_]+)_#\1/#;s#_([rt]x)$#/\1#' <<< ${fname}	# replace first _ and last by /
}

perror() {
	local opt=$1
	shift
	error "option ${opt} requires parameter $*"
	exit 1
}

args=()
bin="`dirname "$0"`"/src &&
keep=0 &&
cpus=1 &&
while [ $# -gt 0 ]
do
	case "$1" in
	-d) set -x ;;
	-k) keep=1 ;;
	-i) [[ $# -gt 1 ]] && src=$2 && shift || perror -i SRC ;;
	-j) [[ $# -gt 1 ]] && cpus=$2 && shift || perror -j JOBS
		[[ 0 -eq ${cpus} ]] && cpus=`lscpu -b -p | sed 's/#.*//;/^\s*$/d' | wc -l`
		;;
	-s) [[ $# -gt 1 ]] && spec=$2 && shift || perror -s SPEC ;;
	-t) [[ $# -gt 1 ]] && tgt=$2 && shift || perror -t TGT ;;
	# train opts
	-b) [[ $# -gt 1 ]] && args+=( TRAIN_BATCH="$2" ) && shift || perror -b BATCH ;;
	-e) [[ $# -gt 1 ]] && args+=( TRAIN_EPOCHS="$2" ) && shift || perror -e EPOCHS ;;
	-f) [[ $# -gt 1 ]] && args+=( TRAIN_FILTER="$2" ) && shift || perror -f FILTER ;;
	-l) [[ $# -gt 1 ]] && args+=( TRAIN_LAYERS="$2" ) && shift || perror -l LAYERS ;;
	-o) [[ $# -gt 1 ]] && args+=( OBJT="$2" ) && shift || perror -o OBJT ;;
	-r) [[ $# -gt 1 ]] && args+=( RESP="$2" ) && shift || perror -r RESP ;;
	-s) [[ $# -gt 1 ]] && args+=( TRAIN_SEED="$2" ) && shift || perror -s SEED ;;
	# search/collect opts
	-c) [[ $# -gt 1 ]] && args+=( COFF="$2" ) && shift || perror -c COFF ;;
	-n) [[ $# -gt 1 ]] && args+=( N="$2" ) && shift || perror -n N ;;
	-L) [[ $# -gt 1 ]] && args+=( TLO="$2" ) && shift || perror -L TLO ;;
	-H) [[ $# -gt 1 ]] && args+=( THI="$2" ) && shift || perror -H THI ;;
	# general ops
	-h) usage 0 ;;
	-?) usage 1 ;;
	--) shift; break ;;
	*) break ;;
	esac
	shift
done &&
: ${tgt:=`tgt_from_file "${src}"`} || exit $?
: ${MAKE:=gmake}

"$@"
