
from ..util import const, log, die
from ..train import nn_main
from .defs  import shai_data_desc

from common import DataFileInstance

from fractions import Fraction, Decimal
from typing    import Union, List, Tuple, NamedTuple
from copy      import copy, deepcopy

import json

import pandas as pd

class Dimension:
	spec_entry : dict

	def __init__(self, spec_entry):
		self.spec_entry = spec_entry

	def __getattr__(self, name):
		if name == 'spec_entry':
			return super().__getattr__(name)
		else:
			try:
				return self.spec_entry[name]
			except KeyError as e:
				raise AttributeError("'%s' object has no attribute '%s'"
				                     % (type(self).__name__, name)
				                    ) from e

	def __setattr__(self, name, value):
		if name == 'spec_entry':
			super().__setattr__(name, value)
		else:
			self.spec_entry[name] = value

	def __delattr__(self, name):
		if name == 'spec_entry':
			raise NotImplementedError
		else:
			del self.spec_entry[name]

	# Required for pandas
	def __hash__(self):
		return str(self).__hash__()

	# Required for pandas
	def __eq__(self, other):
		return str(self) == other

	def __str__(self):
		return self.label

	def __repr__(self):
		return '%s(%s)' % (type(self).__name__, repr(self.spec_entry))

	def assert_compat(self, other):
		a, b = self.type, other.type
		assert a == b, "'type' not compatible: %s != %s" % (a, b)
		a, b = self.range, other.range
		assert a == b, "'range' not compatible: %s != %s" % (a, b)
		a, b = getattr(self, 'rad-rel', None), getattr(other, 'rad-rel', None)
		assert a == b, "'rad-rel' not compatible: %s != %s" % (a, b)
		a, b = getattr(self, 'rad-abs', None), getattr(other, 'rad-abs', None)
		assert a == b, "'rad-abs' not compatible: %s != %s" % (a, b)
		a, b = getattr(self, 'default', None), getattr(other, 'default', None)
		assert a == b, "'default' not compatible: %s != %s" % (a, b)
		a, b = getattr(self, 'safe', None), getattr(other, 'safe', None)
		assert a == b, "'safe' not compatible: %s != %s" % (a, b)


class JSON_DecimalEncoder(json.JSONEncoder):
	def encode(self, o):
		if isinstance(o, Decimal):
			return str(o)
		else:
			return super().encode(o)


class Speced:
	def __init__(self, data : pd.DataFrame):
		assert type(data) is pd.DataFrame
		for c in data.columns:
			assert type(c) is Dimension, (
				"column '%s' of wrong type: %s != Dimension"
				% (c, type(c)))
		self._data = data

	@property
	def data(self) -> pd.DataFrame:
		return self._data

	@property
	def spec(self) -> list:
		return [c.spec_entry for c in self.data]

	@property
	def dim(self) -> int:
		return len(self.spec)

	def from_subset(self, data):
		return Speced(data)

	def drop(self, feat):
		return Speced(self.data.drop(feat, axis=1))

	def restrict(self, feat, value):
		assert feat in self.data
		return Speced(self.data[self.data[feat] == value])

	def fix(self, feat, value):
		assert feat in self.data
		return self.restrict(feat, value).drop(feat)

	def relabel(self, old, new : str):
		c = deepcopy(self.data[old].name)
		c.label = new
		return Speced(self.data.rename(columns={old: c}))

	@property
	def input_features(self) -> List[Dimension]:
		return [c for c in self.data if c not in self.response_features]

	@property
	def response_features(self) -> List[Dimension]:
		return [c for c in self.data if c.type == 'response']

	@property
	def dependent(self):
		return Speced(self.data[self.response_features])

	@property
	def independent(self):
		return Speced(self.data[self.input_features])

	def store(self, data_out, spec_out, *, openmode='x', json_enc_cls=JSON_DecimalEncoder):
		if isinstance(spec_out, str):
			with open(spec_out, openmode) as f:
				return to_csv(data_out, f, openmode=openmode,
				              json_enc_cls=json_enc_cls)

		import json
		try:
			# spec is more important, try to store it first
			json.dump(self.spec, spec_out, indent=4, cls=json_enc_cls)
		finally:
			self.data.to_csv(data_out, index=False, mode=openmode)

def _explode_cat(sp, col):
	b = sp.data[col]
	assert b.dtype.is_dtype('category')
	for c in b.dtype.categories:
		dsc = sp.fix(col, c)
		for r in dsc.response_features:
			dsc = dsc.relabel(r.label, '%s_%s' % (r.label, c))
		yield dsc

def explode_cat(sp, col):
	gen = _explode_cat(sp, col)
	c = next(gen)
	try:
		while True:
			dsc = next(gen)
			# TODO: merge c.desc and dsc.desc somehow?
			c = Speced(pd.merge(c.data, dsc.data, how='outer'))
	except StopIteration:
		pass
	return c


class ShaiData(Speced):

	def __init__(self, data, desc : NamedTuple):
		assert desc is not None
		for f,c in zip(desc._fields, desc):
			if c is not None:
				if isinstance(c, str):
					assert c in data, "description's '%s' (%s) not in data" % (c,f)
				else:
					for d in c:
						assert d in data, "description's '%s' (in %s) not in data" % (d,f)
		super().__init__(data)
		self._desc = desc

	@property
	def desc(self):
		return self._desc

	def from_subset(self, data):
		return ShaiData(data, self.desc)

	def drop(self, feat):
		return ShaiData(super().drop(feat).data,
		                self.desc.drop(feat))

	def restrict(self, feat, value):
		return ShaiData(super().restrict(feat, value).data, self.desc)

	# fix is OK from Speced

	def relabel(self, old, new : str):
		return ShaiData(super().relabel(old, new).data,
		                self.desc.relabel(old, new))


def load_spec(path : str, floats=Decimal):
	import json
	with open(path) as f:
		spec = json.load(f, parse_float=floats)
	assert type(spec) is list
	for i,s in enumerate(spec):
		assert 'label' in s and type(s['label']) is str, (
			"'label' missing or of wrong type in item %d" % i)
		assert 'type' in s and s['type'] in ('knob', 'input', 'response', 'categorical'), (
			"'type' missing or has wrong value in item %d labelled '%s'" % (i, s['label']))
		assert 'range' in s and (
			(type(s['range']) is list) if s['type'] == 'categorical'
			else (s['range'] in ('int','float'))), (
			"'range' missing or has wrong value in item %d labelled '%s'" % (i, s['label']))
	return spec


def speced(data : Union[pd.DataFrame, str], spec : Union[list, str],
           cls : type = Speced):
	if isinstance(data, str):
		data = pd.read_csv(data)

	if isinstance(spec, str):
		spec = load_spec(spec)

	assert len([c for c in data]) == len(spec)
	for i,(c,s) in enumerate(zip(data, spec)):
		assert c == s['label'], (
			"column %d labels differ: data '%s' != '%s' in spec"
			% (i, c, s['label']))

	def interp(ser, sp):
		if sp['type'] == 'categorical':
			return ser.astype('category')
		else:
			return ser

	return cls(pd.DataFrame({ Dimension(s): interp(data[c], s)
	                          for c,s in zip(data, spec) }))


def lookup_joint(rx, tx, joint):
	assert len(set(r for r,t in joint)) == len(joint), (
		'duplicate joint RX features')
	assert len(set(t for r,t in joint)) == len(joint), (
		'duplicate joint TX features')
	rl = list(rx.data.keys())
	tl = list(tx.data.keys())
	for rt in joint:
		r,t = rt
		assert r in rx.data, 'joint feature %s not in RX' % r
		assert t in tx.data, 'joint feature %s not in TX' % t
		r,t = rl[rl.index(r)], tl[tl.index(t)]
		assert r.type != 'response'
		assert t.type != 'response'
		yield r, t

def init_joint(rx_data, rx_spec, tx_data, tx_spec,
               joint : List[Tuple[str,str]], force : bool,
               log = log, cls = Speced):
	log(1, 'loading RX...')
	rx = speced(rx_data, rx_spec, cls=cls)
	log(1, 'loading TX...')
	tx = speced(tx_data, tx_spec, cls=cls)

	joint = list(lookup_joint(rx, tx, joint))

	for r,t in joint:
		try:
			r.assert_compat(t)
		except AssertionError as e:
			if force:
				log(0, 'warning: joint %s=%s: %s' % (r, t, e))
			else:
				raise

	return rx, tx, joint

"""
rx, tx, joint = shai.init_joint('rx-pp.csv','rx-pp.spec','tx-pp.csv','tx-pp.spec',
                                [('DDR5_RTT_PARK_RX','DDR5_RTT_PARK_TX')
                                ,('MC','MC'),('RANK','RANK'),('Byte','Byte')
                                ],True,
                                cls=lambda df: shai.ShaiData(df, preparea.shai_data_desc))
"""

eye_w_col = Dimension({'label': 'eye_w', 'type': 'response'})
eye_h_col = Dimension({'label': 'eye_h', 'type': 'response'})

#trad_col = Dimension({'label': 'trad', 'type': 'knob', 'range': 'float', 'rad-abs': 0})
#obj_col = Dimension({'label': 'area', 'type': 'response'})


#def _preparea(ds, is_rx, log, max_workers):
#	from smlp.mrc import preparea
#	from smlp.util import verbosify
#
#	for r in ds.desc.other_output_cols:
#		ds = ds.drop(r)
#
#	b = ds.data[ds.desc.byte_col]
#	assert b.dtype.is_dtype('category')
#	for c in b.dtype.categories:
#		dsc = ds.fix(ds.desc.byte_col, c)
#		dsc = ShaiData(pd.concat(preparea.prep_area(dsc.data, is_rx, log, max_workers,
#		                                            trad_col=trad_col, obj_col=obj_col,
#		                                            desc = dsc.desc)
#		                        ), dsc.desc)
#		#dsc = dsc.drop(desc.delta_col)
#		for r in dsc.response_features:
#			dsc = dsc.relabel(r.label, '%s_%s' % (r.label, c))
#		yield dsc
#
#def preparea(ds, is_rx, log, max_workers):
#	gen = _preparea(ds, is_rx, log, max_workers)
#	c = next(gen)
#	try:
#		while True:
#			dsc = next(gen)
#			# TODO: merge c.desc and dsc.desc somehow?
#			c = Speced(pd.merge(c.data, dsc.data, how='outer'))
#	except StopIteration:
#		pass
#	return c

# requires access to desc.rank_col, desc.channel_col, desc.byte_col,
#                    desc.timing_col, desc.delta_col, desc.other_output_cols
def prepare(ds : ShaiData, is_rx : bool, log, max_workers : int) -> Speced:
	"""
		Transforms the annotated MRC instance `ds` (either RX or TX)
		into a `Speced` instance with additional features 'trad' and
		'area' by fixing RANK=0, MC=0 and dropping irrelevant outputs
		like 'Area'.
	"""

	from smlp.mrc import preparea

	# fix 'RANK'=0, 'MC'=0
	ds = ds.fix(ds.desc.rank_col, 0).fix(ds.desc.channel_col, 0)

	# drop 'Area'
	for r in ds.desc.other_output_cols:
		ds = ds.drop(r)

	# compute non-linear 'area' for default set of 'trad'
	ds = ShaiData(#pd.concat(preparea.prep_area(ds.data, is_rx, log,
	              #                             max_workers,
	              #                             trad_col = trad_col,
	              #                             obj_col = obj_col,
	              #                             desc = ds.desc)
	              #         ),
	              preparea.v2(ds.data, preparea.DEF_TIME_WINDOW_RADII,
	                          ds.desc, eye_w = eye_w_col, eye_h = eye_h_col,
	                          max_workers=max_workers
	                         ),
	              ds.desc)

	ds = ds.drop(ds.desc.delta_col)

	from pprint import pprint
	import sys
	pprint([c.label for c in ds.data.columns], stream=sys.stderr)

	# transform 'delta' and 'area' every value i of 'Byte'
	# into 'delta_i' and 'area_i' to get rid of 'Byte'
	return explode_cat(ds, ds.desc.byte_col)


def parse_args(argv):
	import argparse

	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('rx_data', type=str, help='path to RX-PP.csv')
	p.add_argument('rx_spec', type=str, help='path to RX-PP.spec')
	p.add_argument('tx_data', type=str, help='path to TX-PP.csv')
	p.add_argument('tx_spec', type=str, help='path to TX-PP.spec')
	p.add_argument('-j', '--joint', type=str, default='', help=''
	               'comma-separated list of the form '
	               'RXCOL1=TXCOL1[,RXCOL2=TXCOL2[,...]] '
	               'specifying the joint features between RX and TX')
	p.add_argument('-f', '--force', default=False, action='store_true',
	               help='ignore incompatibilities between joint features')
	p.add_argument('-n', '--max_workers', default=None, type=int, help=''
	               'number of parallel jobs to use for preparea '
	               '[default: #cpus]')
	p.add_argument('-q', '--quiet', default=False, action='store_true',
	               help='suppress log messages')
	p.add_argument('-o', '--outdir', metavar='DIR', type=str, help=
	               'path to (non-existent) working directory to prepare')
	p.add_argument('-v', '--verbose', default=0, action='count',
	               help='increase verbosity')
	args = p.parse_args(argv[1:])

	joint = []
	if len(args.joint) > 0:
		try:
			cols = args.joint.split(',')
			if len(cols[-1]) == 0:
				cols = cols[:-1]
			for e in cols:
				r,t = e.split('=')
				joint.append((r,t))
		except ValueError:
			die(1, 'error: list in invalid format for parameter --joint: %s'
			       % args.joint)
	args.joint = joint

	return args


if __name__ == '__main__':
	import sys, os, json
	args = parse_args(sys.argv)
	log.verbosity = args.verbose

	rx, tx, joint = init_joint(args.rx_data, args.rx_spec,
	                           args.tx_data, args.tx_spec,
	                           args.joint, force=args.force, log=log,
	                           cls=lambda df: ShaiData(df, shai_data_desc))

	rx = prepare(rx, True, log, args.max_workers)
	tx = prepare(tx, False, log, args.max_workers)

	if args.outdir is not None:
		if os.path.exists(args.outdir):
			assert os.path.isdir(args.outdir), (
				"error: output DIR path '%s' exists and "
				"is not a directory" % args.outdir)
			assert len(os.listdir(args.outdir)) == 0, (
				"error: output DIR '%s' exists and is not empty"
				% args.outdir)
		os.mkdir(args.outdir)
		with open(os.path.join(args.outdir, 'joint'), 'x') as f:
			json.dump(joint, f, indent=4)

		model={}

		for sp, name in [(rx, 'rx'), (tx, 'tx')]:
			wd_prefix = os.path.join(args.outdir, name)
			inst = DataFileInstance(wd_prefix)

			# store the data
			sp.store(inst.data_fname, wd_prefix + '.spec')

			# train NN
			model[name] = nn_main(inst, sp.response_features,
			                      sp.input_features, data=sp,
			                      layers_spec='2,1', seed=1234,
			                      epochs=30, batch_size=32)
