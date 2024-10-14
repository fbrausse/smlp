#!/usr/bin/env python3

import pandas as pd
import openpyxl as ox
import json

def ws2df(ws):
	from itertools import islice
	data = ws.values
	cols = next(data)[1:]
	data = list(data)
	idx = [r[0] for r in data]
	data = (islice(r, 1, None) for r in data)
	df = pd.DataFrame(data, index=idx, columns=cols)
	return df

wb = ox.load_workbook('via_results.xlsx')
df = pd.concat([ws2df(ws) for ws in wb])

# drop the constant column called "pad" from the dataset as well as all_scores
df = df.drop('pad', axis=1).drop('all_scores', axis=1)

# rename problematic columns (they need to form valid Python identifiers)
# for the specification of the objective
df = df.rename(columns={
	col: col.replace(' ', '_')
	        .replace('+', '_')
	        .replace('@', '_')
	        .replace('.', '_')
	for col in df.columns
})

# TODO: make 'stackup' unordered categorical: pandas.get_dummies()

# write .spec file for training/solving
def spec_feat(name):
	r = { 'label': name, }
	if name == 'stackup':
		r['type'] = 'input'
		r['range'] = 'int'
	elif name.startswith('RL__'):
		r['type'] = 'response'
		r['range'] = 'float'
	else:
		r['type'] = 'knob'
		r['range'] = 'int'
		r['rad-rel'] = 0.05
	if name == 'st_X':
		r['grid'] = list(range(60, 88+1, 4))
	return r

# order is important: same as columns in data set
spec = [spec_feat(col) for col in df.columns]

with open('via_results.spec', 'w') as f:
	json.dump(spec, f)

# Create sub-directories a/, b/, c/, ... for each response and store a Makefile
# describing this subdirectory and referring to 'common.mk' in this directory
import os

dirs=[]

for i,resp in enumerate(filter(lambda x: x['type'] == 'response', spec)):
	# create a subdirectory, link the data.csv
	d = chr(ord('a')+i)
	dirs.append(d)
	os.makedirs(d, exist_ok=True)
	try:
		os.symlink('../via_results.csv', d + '/data.csv')
	except FileExistsError as e:
		pass
	# and put a Makefile delegating to top-level common.mk
	with open(d + '/Makefile', 'w') as f:
		print('OBJ = %s' % (resp['label'],), file=f)
		print('include ../common.mk', file=f)

	# the solver maximizes, so negate this objective
	df[resp['label']] = -df[resp['label']]

# write .csv file for training/solving
df.to_csv('via_results.csv', index=False)

with open('Makefile', 'w') as f:
	print("""
train search:
	$(MAKE) -C a $@
	$(MAKE) -C b $@
	$(MAKE) -C c $@
	$(MAKE) -C d $@
	$(MAKE) -C e $@
""", file=f)

print("Done preparing subdirectories %s, now run 'make train && make search'" % (dirs,))
print("Note: the solver works on negated objectives - it maximizes, so results will need negation as well")
print()
print("Results will be in each sub-directory in 'search.safe.csv':")
print("col 'center_obj' holds the objective value at the center of the region")
print("col 'thresh' contains the max-threshold found for stable regions normalized to [0,1]")
