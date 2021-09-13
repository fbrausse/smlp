#!/usr/bin/env python3

import pandas as pd
import sys

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
			'range': 'float',
		}
	#
	tys = ('rx', 'tx')
	key_rename = {
		'SOC_ODT': 'CPU_ODT',
		'SOC_RON': 'CPU_RON',
	}
	def_pre_spec = {
		'Timing': { 'type': 'knob', 'range': 'int' },
		'Area'  : { 'type': 'response', 'range': 'float' },
		'MC'    : { 'type': 'categorical', 'range': [0,1] },
		'RANK'  : { 'type': 'categorical', 'range': [0,1] },
		'Byte'  : { 'type': 'categorical', 'range': list(range(4)) },
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
		                *[[(key_rename.get(k, k),
		                    shai_interp_list(v))
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

#def export_bios_mrc_tx_rx_datasets(data, cols, which, out=sys.stdout):
#	pass

def log(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def die(code, *args, **kwargs):
	log(*args, **kwargs)
	sys.exit(code)

def usage(argv):
	return '\n'.join([
'usage: %s PRODUCT TYPE CSV [OUT]' % argv[0],
'',
'Supported PRODUCT values: %s' % ', '.join(list(prod_cols.keys())),
	])

def main(argv):
	if len(argv) < 4 or len(argv) > 5:
		die(1, usage(argv))

	product = sys.argv[1]
	cols = prod_cols.get(product)
	if cols is None:
		s = spec.get(product)
		if s is None:
			die(2, 'error: product "%s" not supported' % product)
		cols = {
			ty: [f['label'] for f in e]
			for ty,e in s.items()
		}

	which = sys.argv[2]
	if which not in cols:
		die(3, 'error: type "%s" not in supported types: %s' %
		       (which, list(cols.keys())))

	log('extracting cols %s' % cols[which])

	data = pd.read_csv(sys.argv[3])
	data['delta'] = data['Up'] - data['Down']
	data = data[cols[which]]
	with open(sys.argv[4], 'w') if len(sys.argv) >= 5 else sys.stdout as f:
		data.to_csv(f, index=False)

if __name__ == "__main__":
	sys.exit(main(sys.argv))
