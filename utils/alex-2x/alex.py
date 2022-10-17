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

wb = ox.load_workbook('new.xlsx')
df = ws2df(wb["via#1_All_Results"])

# drop the constant column called "pad" from the dataset as well as all_scores
assert all(df['pad'] == df['pad'][0])
df = df.drop('pad', axis=1) # column B, all 16

# rename problematic columns (they need to form valid Python identifiers)
# for the specification of the objective
def prep_name(name):
	return (name.replace(' ', '_')
	            .replace('+', '_')
	            .replace('@', '_')
	            .replace('.', '_'))

df = df.rename(columns={
	col: prep_name(col)
	for col in df.columns
})

# TODO: make 'stackup' unordered categorical: pandas.get_dummies()

#B := 'pad'
#J == (D-C-B)/2 -- B == (D-C-2J)
#J == (E-B)/2   -- B == (E-2J)
# -> drop 'E' besides 'pad' = 'B'
df.drop('E', axis=1)

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
		r['range'] = 'float'
		r['rad-abs'] = 0.5
	if r['type'] != 'response':
		for k in df[name]:
			assert str(int(k)) == str(k), (name, k)
	if name == 'H':
		r['safe'] = list(range(60, 88+1, 4)) # key 'safe': [60,64,68,...,88]
	elif name in "CJDE":
		r['safe'] = list(range(min(df[name]), max(df[name])+1))
	return r

# order is important: same as columns in data set
spec = [spec_feat(col) for col in df.columns]

with open('new.spec', 'w') as f:
	json.dump(spec, f)

# Create sub-directories a/, b/, c/, ... for each response and store a Makefile
# describing this subdirectory and referring to 'common.mk' in this directory
import os

dirs=[]

resps = filter(lambda x: x['type'] == 'response', spec)

for i,resp in enumerate(resps):
	# create a subdirectory, link the data.csv
	d = chr(ord('a')+i)
	dirs.append(d)
	os.makedirs(d, exist_ok=True)
	try:
		os.symlink('../new.csv', d + '/data.csv')
	except FileExistsError as e:
		pass
	# and put a Makefile delegating to top-level common.mk
	with open(d + '/Makefile', 'w') as f:
		print('OBJ = %s' % (resp['label'],), file=f)
		print('include ../common.mk', file=f)

	# the solver maximizes, so negate this objective
	df[resp['label']] = -df[resp['label']]

# TODO: add new objective: 1/(1/||obj_i - RL_i_delta||^2)

# write .csv file for training/solving
df.to_csv('new.csv', index=False)

alpha = {
	'RL @8e+09Hz' : 27.5,
	'RL @1.6e+10Hz': 22.5,
	'RL @2.4e+10Hz': 17.5,
	'RL @3.2e+10Hz': 15,
	'RL @4e+10Hz': 15
}

for k in alpha:
	assert prep_name(k) in df.columns, (k, prep_name(k))

with open('Makefile', 'w') as f:
	print('ALPHA = And(%s)' %
		', '.join(['%s > %s' % (prep_name(k), v) for k,v in alpha.items()]),
		file=f)
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
