# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import sys, functools

def log(lvl, *args, **kwargs):
	if lvl <= log.verbosity:
		print(*args, file=sys.stderr, **kwargs)

log.verbosity = 0

def _verbosified(f, n, lvl, *args, **kwargs):
	return f(lvl-n, *args, **kwargs)

def verbosify(f, n=1):
	# functools.partial is compatible with
	# concurrent.futures.ProcessPoolExecutor,
	# as used by, e.g., smlp.mrc.preparea
	return functools.partial(_verbosified, f, n)

def die(code, *args, **kwargs):
	log(0, *args, **kwargs)
	sys.exit(code)

__all__ = [s.__name__ for s in (log, verbosify, die)]
