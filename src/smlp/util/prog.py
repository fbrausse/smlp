
import sys

def log(lvl, *args, **kwargs):
	if lvl <= log.verbosity:
		print(*args, file=sys.stderr, **kwargs)

log.verbosity = 0

def verbosify(f, n=1):
	return lambda lvl, *args, **kwargs: f(lvl-n, *args, **kwargs)

def die(code, *args, **kwargs):
	log(0, *args, **kwargs)
	sys.exit(code)
