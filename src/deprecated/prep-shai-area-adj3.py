#!/usr/bin/env python3

from smlp.mrc.preparea import prep_area
from smlp.util.prog import *
import sys

if __name__ == '__main__':
	if len(sys.argv) != 2 or sys.argv[1] not in ('rx','tx'):
		die(1, 'usage: %s {rx|tx} < DATA.csv > ADJB.csv' % sys.argv[0])

	prep_area(sys.stdin, sys.argv[1] == 'rx', sys.stdout, log=verbosify(log))
