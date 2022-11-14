#!/usr/bin/env python3

import json, csv
import sys, os
from fractions import Fraction

def denorm(bnd, v):
	Min = bnd['min']
	Max = bnd['max']
	return Min + (Max - Min) * v

if __name__ == '__main__':
	if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
		gp = sys.argv[1] + '/model_gen_data_12.json'
		cp = sys.argv[1] + '/search.safe.csv'
	elif len(sys.argv) == 3:
		gp = sys.argv[1]
		cp = sys.argv[2]
	else:
		print('''usage: %s { DIR | GEN CSV }

Outputs the transformed CSV to standard output.

If given, DIR must contain the files 'data_bounds_data_12.json',
'model_gen_data_12.json' and 'search.safe.csv'.

Otherwise, a JSON file GEN containing a record with entries

  "obj-bounds": { "min": value, "max": value },
  "pp": { "response": "min-max" }

and the CSV to transform from the arguments are used.''', file=sys.stderr)
		sys.exit(1)

	with open(gp) as f:
		gen = json.load(f)
		assert gen['pp']['response'] == 'min-max'
		ob  = gen['obj-bounds']

	with open(cp) as f:
		w = csv.writer(sys.stdout, dialect='unix', quoting=csv.QUOTE_MINIMAL)
		for i,rec in enumerate(csv.reader(f, dialect='unix')):
			if i == 0:
				hdrs = rec
				row = rec
			else:
				row = []
				for l,v in zip(hdrs,rec):
					v = Fraction(v)
					v = v.numerator if v.denominator == 1 else float(v)
					if l in ('center_obj','thresh'):
						v = denorm(ob, v)
					row.append(v)
			w.writerow(row)
