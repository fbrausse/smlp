#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import concurrent.futures

def f_weigh(x, sigma, x0):
	return 1/(1+np.exp(-2*sigma*(x-x0)))

#def inv_f_weigh_002(y, x0):
#	return 25 * np.log(-y/(y-1)) + x0

rx_delta_a = 2.346
rx_time_a  = 4.882
tx_delta_a = 15.6
tx_time_a  = rx_time_a

if True: # rx
	delta_a, time_a = rx_delta_a, rx_time_a
else: # tx
	delta_a, time_a = tx_delta_a, tx_time_a


# delta
fd = lambda d: f_weigh(d, 0.02 / delta_a, 100)
ft = lambda t: f_weigh(t, 0.02 / time_a, 100)
#inv_ft = lambda y: inv_f_weigh_002(y, 19)
#inv_fd = lambda y: inv_f_weigh_002(y, 16)
f = lambda t, d: ft(t) * fd(d)

#def inv_f_d(t):
#	return lambda y: inv_ft(y / ft(t))

def obj(tw, t):
	wd = tw['delta']
	wd.index = tw['Timing']
	delta = min(wd)
	#delta_area = delta * len(wd)
	return f(t, delta)

def time_win_sz(tw):
	x = tw['Timing']
	return max(x) - min(x) + 1

time_window_radii = [100 + i for i in [-30,-20,-10,0,10,20,30]]

def tform(kv):
	g = kv[1]
	print('tform %s' % (kv[0],), file=sys.stderr)
	v = g.sort_values(by=['Timing'], ascending=True)

	def ff(time_window_rad):
		w = pd.DataFrame(columns=['trad','area'], index=v.index)
		for i in range(len(v)):
			r = v.iloc[i]
			tw = v[abs(v['Timing'] - r['Timing']) <= time_window_rad]
			w.iloc[i,1] = obj(tw, time_win_sz(tw))
			w.iloc[i,0] = time_window_rad
		w[v.columns] = v
		return w

	return pd.DataFrame().append([ff(rad) for rad in time_window_radii])

data_path = sys.argv[1]
data = pd.read_csv(data_path)
data = data.drop_duplicates()

grid_cols = [c for c in data.columns if c not in ('Timing','Area','delta')]
grid = data.groupby(grid_cols)

with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
	wrhdr = True
	fut2res = [ex.submit(tform, kv) for kv in grid]
	for fut in concurrent.futures.as_completed(fut2res):
		fut.result().to_csv(sys.stdout, index=False, header=wrhdr)
		wrhdr = False
# pd.concat([tform(kv) for kv in grid]).write_csv(sys.stdout, index=False)
