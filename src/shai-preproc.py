#!/usr/bin/env python3

import pandas as pd

import sys, argparse, json

from smlp.util.prog import log, die
from smlp.mrc.defs import bios_mrc_tx_rx_dataset_meta, prod_cols, spec


def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('product', metavar='PRODUCT', type=str,
	               help='supported: %s' % ', '.join(
				set(prod_cols.keys() | spec.keys())))
	p.add_argument('ty', metavar='TYPE', type=str,
	               help="one of 'rx', 'tx'")
	p.add_argument('-i', '--input', type=str, help='path to input CSV')
	p.add_argument('-o', '--output', type=str, help='path to output CSV')
	p.add_argument('-n', '--no-csv', default=False, action='store_true',
	               help='only output SPEC, do not process CSV')
	p.add_argument('-s', '--spec', type=str, help='path to output .spec')
	p.add_argument('-q', '--quiet', default=False, action='store_true',
	               help='suppress log messages')
	p.add_argument('-v', '--verbose', default=0, action='count',
	               help='increase verbosity')
	args = p.parse_args(argv[1:])
	return args

def main(argv):
	args = parse_args(argv)
	log.verbosity = -1 if args.quiet else args.verbose

	if args.no_csv:
		inp, out = None, None
	else:
		inp = args.input
		if inp is None or inp == '-':
			inp = sys.stdin

		out = args.output
		if out is None or out == '-':
			out = sys.stdout

	specfd = args.spec
	if specfd == '-':
		specfd = sys.stdout
	elif specfd is not None:
		try:
			specfd = open(specfd, 'x')
		except OSError as e:
			die(2, 'error opening SPEC: %s' % e)

	cols, s = bios_mrc_tx_rx_dataset_meta(args.product, args.ty)

	if specfd is not None:
		if s is None:
			raise ValueError('.spec generation for product "%s" '
			                 'not implemented' % args.product)
		json.dump(s, specfd, indent=4)

	if inp is None:
		return None

	log(1, 'extracting cols %s' % cols)

	data = pd.read_csv(inp)
	data['delta'] = data['Up'] - data['Down']
	data = data[cols]
	data.to_csv(out, index=False)

if __name__ == "__main__":
	try:
		sys.exit(main(sys.argv))
	except Exception as e:
		die(2, 'error: %s' % e)
