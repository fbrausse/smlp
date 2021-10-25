
import ctypes as C, os
from typing import Sequence

import numpy as np
import numpy.ctypeslib as npct

from smlp.mrc.shai import speced

class _cats(C.Structure):
	_fields_ = list({
		'data'  : C.POINTER(C.c_char_p),
		'concat': C.c_char_p,
		'n'     : C.c_size_t,
	}.items())

class smlp_value(C.Union):
	_fields_ = list({
		'c': C.c_size_t,
		'i': C.c_longlong, # actually, intmax_t
		'd': C.c_double,
	}.items())

class _safe(C.Structure):
	_fields_ = list({
		'data': C.POINTER(smlp_value),
		'n'   : C.c_size_t
	}.items())

class _rad(C.Union):
	_fields_ = list({
		'abs': smlp_value,
		'rel': C.c_double,
	}.items())

class smlp_spec_entry(C.Structure):
	_fields_ = [
		('dtype'            , C.c_uint, 2),
		('purpose'          , C.c_uint, 2),
		('radius_type'      , C.c_uint, 2),
		('has_default_value', C.c_uint, 1),
	] + list({
		'label': C.c_char_p,
		'cats' : _cats,
		'safe' : _safe,
		'default_value': smlp_value,
		'rad'  : _rad,
	}.items())

class smlp_spec(C.Structure):
	_fields_ = list({
		'cols'    : C.POINTER(smlp_spec_entry),
		'n'       : C.c_size_t,
	}.items())

class smlp_mrc_shai(C.Structure):
	_fields_ = list({
		'shai'       : C.c_void_p,
		'error'      : C.c_char_p,
		'np_cols'    : C.POINTER(C.py_object), # np.array
		'np_idcs'    : C.py_object,
		'width'      : C.c_size_t,
		'const_width': C.c_size_t,
		'spec'       : C.POINTER(smlp_spec),
	}.items())

class smlp_mrc_shai_params(C.Structure):
	_fields_ = list({
		'csv_in_path'   : C.c_char_p,
		'spec_in_path'  : C.c_char_p,
		'spec_out_path' : C.c_char_p,
		'timing_lbl'    : C.c_char_p,
		'delta_lbl'     : C.c_char_p,
		'v1'            : C.c_char_p,
	}.items())

def c_spec_to_py(spec):
	def entry(e):
		r = {}
		r['label'] = e.label.decode()
		dt = {0: lambda v: int(v.i),
		      1: lambda v: float(v.d),
		      2: lambda v: int(v.c),
		     }[e.dtype]
		if e.dtype == 2:
			assert e.purpose == 0
		r['type'], r['range'] = (
			('categorical',
			 [{0: int, 1: float, 2: int}[e.dtype](e.cats.data[i].decode())
			  for i in range(e.cats.n)])
			if e.dtype == 2 else
			({0: 'knob', 1: 'input', 2: 'response'}[e.purpose],
			 {0: 'int', 1: 'float'}[e.dtype])
		)
		if e.radius_type == 1:
			r['rad-abs'] = dt(e.rad.abs)
		elif e.radius_type == 2:
			r['rad-rel'] = e.rad.rel
		elif e.radius_type == 3:
			r['rad-abs'] = 0
		if e.safe.n > 0:
			r['safe'] = [dt(e.safe.data[i]) for i in range(e.safe.n)]
		if e.has_default_value:
			r['default'] = dt(e.default_value)
		return r
	return [entry(spec.cols[i]) for i in range(spec.n)]


def _init_lib(dir_ = os.path.join(os.path.dirname(__loader__.path),'..','lib')):
	lib = npct.load_library('libshai-prep', dir_)
	for k,v in {
		'init': (C.c_int, (C.POINTER(smlp_mrc_shai),
				   C.POINTER(smlp_mrc_shai_params),
				   C.c_uint)),
		'fini': (None   , (C.POINTER(smlp_mrc_shai),)),
		'rad' : (None   , (C.POINTER(smlp_mrc_shai),
				   C.c_int)),
	}.items():
		f = getattr(lib, 'smlp_mrc_shai_prep_' + k)
		f.restype, f.argtypes = v
	lib.verbosity = C.c_int.in_dll(lib, 'verbosity')
	return lib

lib = _init_lib()

class SP:
	def __init__(self, csv_in_path : str, spec_in_path : str,
	             timing_lbl : str, delta_lbl : str,
	             spec_out_path : str = None, v1 : str = None,
	             verbosity : int = None):
		self._sh = smlp_mrc_shai()

		if verbosity is not None:
			lib.verbosity.value = verbosity

		par = smlp_mrc_shai_params(csv_in_path.encode(),
		                           spec_in_path.encode(),
		                           spec_out_path.encode() if spec_out_path is not None else None,
		                           timing_lbl.encode(),
		                           delta_lbl.encode(),
		                           v1.encode() if v1 is not None else None)

		r = lib.smlp_mrc_shai_prep_init(C.byref(self._sh),
		                                C.byref(par),
		                                0)
		if r < 0:
			raise OSError(os.strerror(-r))
		elif r > 0:
			raise ValueError(self._sh.error.decode())

		self._spec = c_spec_to_py(self._sh.spec[0])

	def __del__(self):
		lib.smlp_mrc_shai_prep_fini(C.byref(self._sh))

	def _trad(self, trad):
		lib.smlp_mrc_shai_prep_rad(C.byref(self._sh), trad)

	@property
	def spec(self):
		return self._spec

	@property
	def const_width(self):
		return self._sh.const_width

	@property
	def width(self):
		return self._sh.width

	@property
	def height(self):
		return len(self._sh.np_idcs)

	def dataframe(self, /, trad = None, const = False):
		import pandas as pd
		w = self.const_width if const else self.width
		if trad is not None:
			self._trad(trad)
		data = pd.DataFrame({self.spec[i]['label']: self._sh.np_cols[i]
		                     for i in range(w)}, copy=False)
		return data.iloc[self._sh.np_idcs]

	def speced(self, trad):
		return speced(self.dataframe(trad), self.spec)


if __name__ == '__main__':
	#csv_in = 'test.csv'
	csv_in = '../../problems/shai/9/1/rx-pp.csv'
	sp = SP(csv_in, 'shai-9-1-rx-pp.spec', 'Timing', 'delta', verbosity=2)
	# idcs = C.pythonapi.Py_IncRef(sp._sh.np_idcs)
