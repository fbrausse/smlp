#!/usr/bin/env python3

import openpyxl as ox

wb = ox.load_workbook('formula_EH_16.xlsx')
ws = wb['formula_EH_16']

it = iter(ws.values)
hdr = next(it)
assert hdr == ('Term', 'Coefficient')
icp = next(it)
assert icp[0] == 'Intercept'
with open('infix.term', 'x') as f:
	print(icp[1], file=f)
	for term, coeff in it:
		print('+%s*%s' % (coeff, term), file=f)
