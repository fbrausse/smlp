
from enum import Enum, unique
from typing import NamedTuple, Sequence, List

@unique
class Interp(Enum):
	Config   = 'knob'
	Input    = 'input'
	Response = 'response'

@unique
class Type(Enum):
	Int  = 'int'
	Cat  = 'categorical'
	Real = 'float'

class FeatureNT(NamedTuple):
	label : str
	type  : Interp
	range : Type

class BaseSpec(List[FeatureNT]):
	def __init__(self, fs : Sequence[FeatureNT], file_path = None):
		super().__init__(fs)
		self.file_path = file_path

def base_feature(nt : FeatureNT) -> dict:
	maps = {
		'label': lambda x  : x,
		'type' : lambda ty : ty.value,
		'range': lambda rng: rng.value,
	}
	return { k: maps.get(k, lambda x: x)(v) for k,v in zip(nt._fields, nt) }

__all__ = [s.__name__ for s in (Interp, Type, FeatureNT, BaseSpec
                               ,base_feature
                               )]
