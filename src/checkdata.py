#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import ctypes as C, os

class CD:
	class _lib(C.CDLL):
		def __init__(self,
		             path=os.path.join(os.path.dirname(__loader__.path),
		                               '..','lib','libcheck-data.so')):
			class create_data(C.Structure):
				pass
			super().__init__(path)
			P = C.POINTER
			for k,v in {
				'create' : (P(create_data), (   C.c_char_p ,
				                                C.c_char_p ,
				                                C.c_char_p ,
				                              P(C.c_float ),
				                                C.c_size_t ,
				                              P(C.c_char_p),
				                            )),
				'destroy': (None, (P(create_data),
				                  )),
				'check'  : (C.c_size_t, (P(create_data),
				                         P(C.c_size_t ),
				                         P(C.c_float  ),
				                           C.c_double  ,
				                           C.c_void_p  ,
				                           C.c_void_p  ,
				                        )),
			}.items():
				f = getattr(self, 'check_data_' + k)
				f.restype, f.argtypes = v

	def __init__(self, objective, spec_path, data_path, safe_labels,
	             bnds=None, lib=None):
		if lib is None:
			lib = CD._lib()
		if bnds is not None:
			bnds = (C.c_float * len(bnds))(*bnds)
		self._lib = lib
		n = len(safe_labels)
		lbls = [l.encode() for l in safe_labels]
		self._cdp = lib.check_data_create(objective.encode(),
		                                  spec_path.encode(),
		                                  data_path.encode(),
		                                  bnds, n,
		                                  (C.c_char_p * n)(*lbls))
		if not self._cdp:
			raise ValueError()

	def __del__(self):
		if self._cdp:
			self._lib.check_data_destroy(self._cdp)

	# returns a pair (F,O) of the numbers of data samples in ball around
	# safe_values failing (F) and satisfying (O) the threshold condition
	def check(self, safe_values, threshold):
		n_ok = C.c_size_t()
		n = len(safe_values)
		n_fail = self._lib.check_data_check(self._cdp, C.byref(n_ok),
		                                    (C.c_float*n)(*safe_values),
		                                    threshold, None, None)
		return n_fail, n_ok.value

if __name__ == "__main__":
	with open('test-safe.csv', 'r') as f:
		import csv
		rd = csv.reader(f)
		safe_labels = next(rd)
		c = CD('delta', 'data.spec', 'data.csv', safe_labels)
		print('fail,ok')
		for r in rd:
			print('%d,%d' % c.check([float(v) for v in r], .9))
