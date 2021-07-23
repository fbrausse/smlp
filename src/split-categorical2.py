#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import json, sys, itertools, os, argparse

import pandas as pd

def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('-f', '--features', type=str,
	               help='comma-separated list of categorical features to split on')
	p.add_argument('spec', metavar='SPEC')
	p.add_argument('data', metavar='DATA')
	p.add_argument('outprefix', metavar='OUTPREFIX')
	return p.parse_args(argv[1:])

args = parse_args(sys.argv)

with open(args.spec, 'r') as f:
	spec = json.load(f)

data = pd.read_csv(args.data).drop_duplicates()
if args.features is None:
	features = [s['label'] for s in spec if s['type'] == 'categorical']
else:
	features = args.features.split(',')
	assert all(any(s['label'] == f and s['type'] == 'categorical' for s in spec)
	           for f in features)
cati = [i for i in range(len(spec)) if spec[i]["label"] in features]
cati.sort()
out=[]
#dir, pre = os.path.split(os.path.realpath(sys.argv[3]))
dir, pre = os.path.split(args.outprefix)
os.makedirs(os.path.realpath(dir), exist_ok=True)
for idx, catv in zip(itertools.count(), itertools.product(*[spec[i]["range"] for i in cati])):
	d = os.path.join(dir, pre + str(idx))
	sp = [s for s in spec if s['label'] not in features]
	os.mkdir(d)
	cats = { spec[i]['label']: v for i,v in zip(cati, catv) }
	data.query(' & '.join(['%s == %s' % (k,v) for k,v in cats.items()]))[
		[s['label'] for s in sp]
	].to_csv(os.path.join(d, 'data.csv'), index=False)
	with open(os.path.join(d, 'data.spec'), 'w') as f:
		json.dump(sp, f)
	with open(os.path.join(d, 'categorical'), 'w') as f:
		json.dump(cats, f)
	with open(os.path.join(d, 'params.mk'), 'w') as f:
		for it in cats.items():
			print('%s = %s' % it, file=f)
		print('include %s' % os.path.join('$(me)',
		                                  os.path.relpath(os.curdir,
		                                                  start=d),
		                                  'params.mk'),
		      file=f)
