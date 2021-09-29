#!/usr/bin/env python3

import operator as op
import functools as ft
import pandas as pd
import sys, argparse

class MissingAction(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		ns = argparse.Namespace()
		if values == 'fail':
			setattr(ns, 'fail', True)
		elif values.startswith('replace='
		setattr(namespace, self.dest, ns)

def _parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0],
	                            description='Creates an SMLP .spec file '
	                                       +'describing a problem instance '
	                                       +'from a dataset. [default]')
	p.add_argument('data', type=str, help='dataset to describe')
	p.add_argument('-m', type=str, metavar='MISSING',
	               help="how to handle missing entries: 'replace=CONST', "
	                   +"['fail']")
	args = p.parse_args(argv[1:])
	return args

def main(argv=None):
	args = _parse_args(argv)
	data = pd.read_csv(args.data)

if __name__ == '__main__':
	sys.exit(main(sys.argv))
