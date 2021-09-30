#!/usr/bin/env python3

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import NamedTuple, Set

from smlp.util import const

def f_weigh(x, sigma, x0):
	return 1/(1+np.exp(-2*sigma*(x-x0)))

#def inv_f_weigh_002(y, x0):
#	return 25 * np.log(-y/(y-1)) + x0

rx_delta_a = 2.346
rx_time_a  = 4.882
tx_delta_a = 15.6
tx_time_a  = rx_time_a

class Tform:
	def __init__(self, is_rx, timing_col='Timing', delta_col='delta',
	             trad_col='trad', obj_col='area'):
		if is_rx:
			self.delta_a, self.time_a = rx_delta_a, rx_time_a
		else:
			self.delta_a, self.time_a = tx_delta_a, tx_time_a
		self.timing_col = timing_col
		self.delta_col  = delta_col
		self.trad_col   = trad_col
		self.obj_col    = obj_col

	def fd(self, d):
		return f_weigh(d, 0.02 / self.delta_a, 100)

	def ft(self, t):
		return f_weigh(t, 0.02 / self.time_a, 100)

	def f(self, t, d):
		return self.ft(t) * self.fd(d)

	def obj(self, tw, t):
		wd = tw[self.delta_col]
		wd.index = tw[self.timing_col]
		delta = min(wd)
		#delta_area = delta * len(wd)
		return self.f(t, delta)

	def time_win_sz(self, tw):
		x = tw[self.timing_col]
		return max(x) - min(x) + 1

	def ff(self, v, time_window_rad):
		w = pd.DataFrame(columns=[self.trad_col,self.obj_col], index=v.index)
		for i in range(len(v)):
			r = v.iloc[i]
			tw = v[abs(v[self.timing_col] - r[self.timing_col]) <= time_window_rad]
			w.iloc[i,1] = self.obj(tw, self.time_win_sz(tw))
			w.iloc[i,0] = time_window_rad
		w[v.columns] = v
		return w

	def tform(self, kv, time_window_radii, log):
		g = kv[1]
		log(1, 'tform %s' % (kv[0],))
		v = g.sort_values(by=[self.timing_col], ascending=True)
		return pd.DataFrame().append([self.ff(v, rad) for rad in time_window_radii])


class DataDesc(NamedTuple):
	timing_col : str
	delta_col : str

	other_output_cols : Set[str]

	@property
	def output_cols(self) -> Set[str]:
		return {self.delta_col, *self.other_output_cols}

def data_desc_from_spec(timing_col, delta_col, spec : list):
	assert all(c in (s['label'] for s in spec)
	           for c in (timing_col, delta_col))
	return DataDesc(timing_col, delta_col,
	                {s['label'] for s in spec
	                 if s['type'] == 'response' and s['label'] != delta_col})

class TradDataDesc(NamedTuple):
	timing_col : str
	trad_col : str
	obj_col : str
	other_output_cols : Set[str]

	@property
	def output_cols(self) -> Set[str]:
		return {self.obj_col, *self.other_output_cols}

shai_data_desc = DataDesc('Timing', 'delta', {'Area'})

def trad_from_data_desc(desc : DataDesc, trad_col='trad', obj_col='area'):
	return TradDataDesc(desc.timing_col, trad_col, obj_col, desc.other_output_cols)

def prep_area(inp, is_rx : bool, out, log=const(None), max_workers=None,
              mp_context=None, desc : DataDesc = shai_data_desc,
              trad_col='trad', obj_col='area',
              time_window_radii=[100 + i for i in [-30,-20,-10,0,10,20,30]]
             ) -> TradDataDesc:

	#if is_rx: # rx
	#	delta_a, time_a = rx_delta_a, rx_time_a
	#else: # tx
	#	delta_a, time_a = tx_delta_a, tx_time_a
	#
	## delta
	#fd = lambda d: f_weigh(d, 0.02 / delta_a, 100)
	#ft = lambda t: f_weigh(t, 0.02 / time_a, 100)
	##inv_ft = lambda y: inv_f_weigh_002(y, 19)
	##inv_fd = lambda y: inv_f_weigh_002(y, 16)
	#f = lambda t, d: ft(t) * fd(d)
	#
	##def inv_f_d(t):
	##	return lambda y: inv_ft(y / ft(t))
	#
	#def obj(tw, t):
	#	wd = tw['delta']
	#	wd.index = tw['Timing']
	#	delta = min(wd)
	#	#delta_area = delta * len(wd)
	#	return f(t, delta)
	#
	#def time_win_sz(tw):
	#	x = tw['Timing']
	#	return max(x) - min(x) + 1
	#
	#def ff(v, time_window_rad):
	#	w = pd.DataFrame(columns=['trad','area'], index=v.index)
	#	for i in range(len(v)):
	#		r = v.iloc[i]
	#		tw = v[abs(v['Timing'] - r['Timing']) <= time_window_rad]
	#		w.iloc[i,1] = obj(tw, time_win_sz(tw))
	#		w.iloc[i,0] = time_window_rad
	#	w[v.columns] = v
	#	return w

	#def tform(kv):
	#	g = kv[1]
	#	log(1, 'tform %s' % (kv[0],))
	#	v = g.sort_values(by=['Timing'], ascending=True)
	#	return pd.DataFrame().append([ff(v, rad) for rad in time_window_radii])

	data = pd.read_csv(inp)
	data = data.drop_duplicates()

	timing_col = desc.timing_col
	delta_col = desc.delta_col
	outputs = desc.output_cols

	assert timing_col not in outputs
	assert delta_col in outputs

	assert timing_col in data.columns
	assert all(o in data.columns for o in outputs)

	assert trad_col not in data.columns
	assert obj_col not in data.columns

	grid_cols = [c for c in data.columns if c not in (timing_col, *outputs)]
	log(1, 'grid defined by columns %s' % grid_cols)
	grid = data.groupby(grid_cols)

	tform = Tform(is_rx, timing_col=timing_col, delta_col=delta_col,
	              trad_col=trad_col, obj_col=obj_col)

	with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as ex:
		wrhdr = True
		fut2res = [ex.submit(tform.tform, kv, time_window_radii, log=log) for kv in grid]
		for fut in as_completed(fut2res):
			fut.result().to_csv(out, index=False, header=wrhdr)
			wrhdr = False
	# pd.concat([tform(kv) for kv in grid]).write_csv(sys.stdout, index=False)

	return trad_from_data_desc(desc, trad_col, obj_col)

if __name__ == '__main__':
	import sys
	from smlp.util.prog import die, log, verbosify

	a = sys.argv

	try:
		is_rx = {'rx': True, 'tx': False}[a[1]]
		max_workers = int(a[2]) if len(a) > 2 else None
	except:
		die(1, 'usage: %s {rx|tx} [N_PROC] < DATA.csv > ADJB.csv' % a[0])

	prep_area(sys.stdin, is_rx, sys.stdout, log=verbosify(log),
	          max_workers=max_workers)
