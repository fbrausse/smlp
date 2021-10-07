
import pandas as pd

from fractions import Fraction, Decimal
from typing    import Union, List, Tuple, NamedTuple
from smlp.util import const, log, die
from copy      import copy, deepcopy

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


class Speced:
	def __init__(self, data : pd.DataFrame):
		assert type(data) is pd.DataFrame
		for c in data.columns:
			assert type(c) is Dimension, "column %s of wrong type: %s != Dimension" % (c, type(c))
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
	def response_features(self) -> List[Dimension]:
		return [c for c in self.data if c.type == 'response']

	@property
	def dependent(self):
		return Speced(self.data[self.response_features])

	@property
	def independent(self):
		return Speced(self.data[[c for c in self.data
		                           if c not in self.response_features]])

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

def speced(data : Union[pd.DataFrame, str], spec : Union[list, str],
           cls : type = Speced):
	if type(data) is str:
		data = pd.read_csv(data)

	if type(spec) is str:
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


def prepare(rx : Speced, tx : Speced, joint : List[Tuple[str,str]]):
	pass


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
                                ],True)
"""

trad_col = Dimension({'label': 'trad', 'type': 'knob', 'range': 'float', 'rad-abs': 0})
obj_col = Dimension({'label': 'area', 'type': 'response'})

def _preparea(ds, is_rx, log, max_workers):
	from smlp.mrc import preparea
	from smlp.util import verbosify

	for r in ds.desc.other_output_cols:
		ds = ds.drop(r)

	b = ds.data[ds.desc.byte_col]
	assert b.dtype.is_dtype('category')
	for c in b.dtype.categories:
		dsc = ds.fix(ds.desc.byte_col, c)
		dsc = ShaiData(pd.concat(preparea.prep_area(dsc.data, is_rx, log, max_workers,
		                                            trad_col=trad_col, obj_col=obj_col,
		                                            desc = dsc.desc)
		                        ), dsc.desc)
		#dsc = dsc.drop(desc.delta_col)
		for r in dsc.response_features:
			dsc = dsc.relabel(r.label, '%s_%s' % (r.label, c))
		yield dsc

def preparea(ds, is_rx, log, max_workers):
	gen = _preparea(ds, is_rx, log, max_workers)
	c = next(gen)
	try:
		while True:
			dsc = next(gen)
			# TODO: merge c.desc and dsc.desc somehow?
			c = Speced(pd.merge(c.data, dsc.data, how='outer'))
	except StopIteration:
		pass
	return c


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


def main(argv):
	args = parse_args(argv)

	rx, tx, joint = init_joint(args.rx_data, args.rx_spec,
	                           args.tx_data, args.tx_spec,
	                           args.joint, force=args.force, log=log)

	rx = rx.fix('RANK', 0).fix('MC', 0)
	#rx.drop('Area')
	tx = tx.fix('RANK', 0).fix('MC', 0)
	#tx.drop('Area')

	from smlp.util import verbosify

	rxpp = preparea(rx.data, True, verbosify(log),
	                max_workers=args.max_workers,
	                trad_col=trad_col, obj_col=obj_col)
	txpp = preparea(tx.data, False, verbosify(log),
	                max_workers=args.max_workers,
	                trad_col=trad_col, obj_col=obj_col)

	return rx, tx, joint, rxpp, txpp


if __name__ == '__main__':
	import sys
	main(sys.argv)
