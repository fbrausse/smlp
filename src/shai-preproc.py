#!/usr/bin/env python3

import pandas as pd

import sys, argparse, json

from smlp.util.prog import log, die

prod_cols = {
	'ICL': {
		'rx': ["Timing","Area","CH","RANK","Byte","LP4_DIMM_RON","CPU_ODT_UP","LP4_SOC_ODT","ICOMP","CTLE_C","CTLE_R","delta"],
		'tx': ["Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN","CPU_RON_UP","LP4_DIMM_ODT_WR","delta"],
	},
	'RKL': {
		'tx_org': ["Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN", "DDR4_RTT_NOM.1","DDR4_RTT_PARK.1", "DDR4_RTT_WR", "CPU_RON_UP", "delta"],
		# just drop the ".1" suffix in two duplicate columns "DDR4_RTT_NOM.1" "DDR4_RTT_PARK.1"
		'tx': ["Timing","Area","CH","RANK","Byte","TXEQ_DEEM","TXEQ_GAIN", "DDR4_RTT_NOM","DDR4_RTT_PARK", "DDR4_RTT_WR", "CPU_RON_UP", "delta"],
		'rx': ["Timing","Area","CH","RANK","Byte", 'DDR4_DIMM_RON', "DDR4_RTT_NOM","DDR4_RTT_PARK", "CPU_ODT_UP","ICOMP","CTLE_C","CTLE_R", "delta"],
	},
	'ADLS': {
		# Area is available in TX and not in RX; not required anyway, dropping from both
		# delta is available in TX and not in RX; recomuting for TX as well as Up - Down
		'tx': ["Timing","MC","RANK","Byte", "DDR4_RTT_NOM_TX", "DDR4_RTT_PARK_TX", "DDR4_RTT_WR", "TXEQ_COEFF0", "TXEQ_COEFF1", "TXEQ_COEFF2", "CPU_RON", "delta"], #"CPU_RON_UP
		# 'DDR4_DIMM_RON', not toggled in recent dataset; might need to add agin to rx_colnms later
		# also in the recent datset CPU_ODT_UP ->CPU_ODT 
		'rx': ["Timing","MC","RANK","Byte","DDR4_RTT_NOM_RX",
		       "DDR4_RTT_PARK_RX", "CPU_ODT","CTLE_EQ","CTLE_C",
		       "CTLE_R","RXBIAS_CTL","RXBIAS_TAIL_CTL","RXBIAS_VREFSEL",
		       "delta"],
	},
}

## FB: my first guess...
#prod_cols['ADLS_new'] = {
#	k: [v.replace('DDR4_RTT_NOM_RX', 'DDR5_RTT_NOM_RD')
##	     .replace('CTLE_EQ', 'RXTAP1')
#	     .replace('DDR4', 'DDR5')
#	    for v in l
#	    if v not in ('CTLE_EQ', 'CTLE_C', 'CTLE_R')
#	   ]
#	for k,l in prod_cols['ADLS'].items()
#}

def shai_params(**params):
	# list concatenation for arbitrarily many lists
	def concat(*lists):
		r = []
		for l in lists:
			r += l
		return r
	# key for length-lexicographic sorting
	def len_lex_key(s):
		return (len(s), s)
	# Shai's lists contain the default as last entry
	def shai_interp_list(l):
		assert len(l) > 0
		return {
			'type': 'knob',
			'safe': l[:-1],
			'default': l[-1],
		}
	def ranges(k):
		if k.startswith('TXEQ_COEFF'):
			return { 'range': 'int', 'rad-abs': 1 }
		else:
			return { 'range': 'float', 'rad-rel': 0.1 }
	def shai_interp(k, v):
		assert type(v) is list
		k = key_rename.get(k, k)
		return (k, dict(concat(shai_interp_list(v).items(), ranges(k).items())))
	#
	tys = ('rx', 'tx')
	key_rename = {
		'SOC_ODT': 'CPU_ODT',
		'SOC_RON': 'CPU_RON',
	}
	def_pre_spec = {
		'Timing': { 'type': 'knob', 'range': 'int', 'rad-abs': 0 },
		'Area'  : { 'type': 'response', 'range': 'float' },
		'MC'    : { 'type': 'categorical', 'range': list(range(2)), 'rad-abs': 0 },
		'RANK'  : { 'type': 'categorical', 'range': list(range(2)), 'rad-abs': 0 },
		'Byte'  : { 'type': 'categorical', 'range': list(range(4)), 'rad-abs': 0 },
	}
	def_resp = {
		'delta' : { 'type': 'response', 'range': 'float' },
	}
	# partition params corresponding to 'rx' and 'tx', respectively,
	# and order them length-lexicographically
	spars = { ty: sorted([k for k in params.keys()
	                      if k.startswith('Parameter_%s' % ty)],
	                     key = len_lex_key)
	          for ty in tys }
	# these should be all, i.e., a proper partition of all our params
	assert sorted(params.keys()) == sorted(concat(*spars.values())), (
		"unknown named arguments not starting with 'Parameter_%s' in: %s" %
		('{%s}' % ','.join(tys) if len(tys) > 1 else str(tys),
		 params.keys()))
	# for 'rx' and 'tx', respectively build a dictionary with feature names
	# as keys and values of the form { 'safe': LIST, 'default': VALUE } as
	# per the lists given in our params;
	# finally concatenate with the defaults
	pre_spec = {
		ty: dict(concat(def_pre_spec.items(),
		                *[[shai_interp(k, v)
		                   for k,v in params[p].items()]
		                  for p in spar],
		                def_resp.items()))
		for ty,spar in spars.items()
	}
	return pre_spec


# copy-paste from Shai's email:
# Goal is to locate good solution for all Memory cards in 4000G2 frequency
# for TX & RX  - Joint parameter is DDR5_RTT_PARK_RX/TX
shai_email_20210802 = dict(

	Parameter_rx_adl_ddr5={
		'RXBIAS_CTL': [8,9,10,11,12,13,14,15,15], #full range 1:15
		'RXBIAS_TAIL_CTL': list(range(4)) + [0],
		'RXBIAS_VREFSEL': list(range(16)) + [13],
		#'SOC_ODT':[15,20,25,30,35,40,45,50,55,60,65,45],
		'SOC_ODT':[40,45,50,55,60,50],
		'RXTAP1': list(range(16)) + [9],
		#'RXTAP2': list(range(32)) + [0],
		# 'RXTAP2': list(range(16)) + [0],
		#'RXTAP3': list(range(16)) + [0],
		# 'RXTAP3': list(range(11)) + [8],
		# 'RXTAP4': list(range(8)) + [0],
	},

	Parameter_rx_ddr5={
	    'DDR5_RTT_NOM_RD': [34,40, 48, 60, 80, 120, 240, 350, 48],
	    #'DDR5_RTT_NOM_RD': [120, 240, 350, 350],
	    #'DDR5_RTT_PARK_RX': [40, 48, 60, 80, 120, 240, 350, 40], #1DPC
	    'DDR5_RTT_PARK_RX': [34,40,48,60, 80, 120,240,60], #2DPC
	    #'DDR5_DIMM_RON': [34, 40, 34],
	   #'DDR5_RON_UP' : [34,40,48,34],
	    'DDR5_RON_UP' : [34,40,34],
	   # 'DDR5_RON_DN' : [34,40,34]
	},

	Parameter_tx_adl_ddr5_4={
	 #'SOC_RON':[20,25,30,35,40,45,50,55,60,65,20], #CPU_RON_UP/DN
	 'SOC_RON':[20,25,30,35,40,20], #CPU_RON_UP/DN
	 'TXEQ_COEFF0': list(range(5)) + [1],#full range 7
	 'TXEQ_COEFF1': list(range(5)) + [1],#full range 7
	 'TXEQ_COEFF2': list(range(5)) + [0]
	 },

	Parameter_tx_ddr5={
	    'DDR5_DFE_TAP1': list(range(81))+[40],
	    'DDR5_DFE_TAP2': list(range(32)) +[15],
	    'DDR5_DFE_TAP3': list(range(26)) +[12],
	    'DDR5_DFE_TAP4': list(range(20)) +[9],
	   # 'DDR5_RTT_NOM_WR': [34, 40, 48, 60, 80, 120, 240, 350, 48],
	    'DDR5_RTT_NOM_WR': [60, 80, 120, 240, 350, 80],
	    #'DDR5_RTT_PARK_TX': [34, 40, 48, 60, 80, 120, 240, 350, 60], #1DPC
	    #'DDR5_RTT_PARK_TX': [48,60, 80, 120, 60], #2DPC
	    'DDR5_RTT_PARK_TX': [40,48,60, 80, 120,240,60], #2DPC
	   # 'DDR5_RTT_WR': [80, 120, 240, 240]
	    'DDR5_RTT_WR': [120, 240, 240]
	}
)

def pre_spec2spec(ps):
	return {
		ty: [{'label': k, **v} for k,v in entries.items()]
		for ty, entries in ps.items()
	}

spec = dict()
spec['ADLS_new'] = pre_spec2spec(shai_params(**shai_email_20210802))

joint = dict()
joint['ADLS_new'] = [('DDR5_RTT_PARK_RX','DDR5_RTT_PARK_TX')]

def bios_mrc_tx_rx_dataset_meta(product, ty):
	cols = prod_cols.get(product)
	s = spec.get(product)

	if cols is None:
		if s is None:
			raise ValueError('product "%s" not supported' % product)
		cols = {
			ty: [f['label'] for f in e]
			for ty,e in s.items()
		}

	if ty not in cols:
		raise ValueError('type "%s" not in supported types: %s' %
			(ty, list(cols.keys())))

	return cols[ty], s[ty] if s is not None else None


__all__ = [s.__name__ for s in (bios_mrc_tx_rx_dataset_meta,spec,)]




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
