#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.


import os, datetime, sys, json
from fractions import Fraction
from collections import OrderedDict
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
import numpy as np


        
NP2PY = {
    np.int64: int,
    np.float64: float,
}

def np2py(o, lenient=False):
    if lenient:
        ty = NP2PY.get(type(o), type(o))
    else:
        ty = NP2PY[type(o)]
    return ty(o)

class np_JSONEncoder(json.JSONEncoder):
    def default(self, o):
        return NP2PY.get(type(o), super().default)(o)

# function to be applied to args options that are intended to be Boolean
# but are specified as strings
def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def timed(f, desc=None, log=lambda *args: print(*args, file=sys.stderr)):
    now = datetime.datetime.now()
    r = f()
    t = (datetime.datetime.now() - now).total_seconds()
    if desc is not None:
        log(desc, 'took', t, 'sec')
        return r
    return r, t

# extract names of system/desigh inputs from interface spec file
def get_input_names(spec):
    return [s['label'] for s in spec if s['type'] != 'response']


def get_radii(spec, center):
    abs_radii = []
    for s,c in zip(spec, center):
        if s['type'] == 'categorical':
            abs_radii.append(0)
            continue
        if 'rad-rel' in s and c != 0:
            w = s['rad-rel']
            abs_radii.append(Fraction(w) * abs(c))
        else:
            try:
                w = s['rad-abs']
            except KeyError:
                w = s['rad-rel']
            abs_radii.append(w)
    return abs_radii

def scaler_from_bounds(spec, bnds):
    sc = MinMaxScaler()
    mmin = []
    mmax = []
    for s in spec:
        b = bnds[s['label']]
        mmin.append(b['min'])
        mmax.append(b['max'])
    sc.fit((mmin, mmax))
    return sc

def io_scalers(spec, gen, bnds):
    si = scaler_from_bounds([s for s in spec
                             if s['type'] in ('categorical', 'knob', 'input')],
                            bnds)
    so = scaler_from_bounds([s for s in spec
                             if s['type'] == 'response'
                             and s['label'] in gen['response']],
                            bnds)
    return si, so

def obj_range(gen, bnds):
    r = gen['response']
    for resp in r:
        if gen['objective'] == resp:
            so = scaler_from_bounds([{'label': resp}], bnds)
            return so.data_min_[0], so.data_max_[0]
    assert len(r) == 2
    sOu = None
    sOl = None
    for i in range(2):
        if gen['objective'] == ("%s-%s" % (r[i],r[1-i])):
            u = bnds[r[i]]
            l = bnds[r[1-i]]
            sOl = u['min']-l['max']
            sOu = u['max']-l['min']
            break
    assert sOl is not None
    assert sOu is not None
    return sOl, sOu

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    @property
    def range(self):
        return self.max - self.min
    def norm(self, x):
        return (x - self.min) / self.range
    def denorm(self, y):
        return y * self.range + self.min

class Id:
    def __init__(self):
        pass
    def norm(self, x):
        return x
    def denorm(self, y):
        return y

SCALER_TYPE = {
    'min-max': lambda b: MinMax(b['min'], b['max']),
    None     : lambda b: Id(),
}

def input_scaler(gen, b):
    return SCALER_TYPE[gen['pp'].get('features')](b)

def response_scaler(gen, b):
    return SCALER_TYPE[gen['pp'].get('response')](b)

def response_scalers(gen, bnds):
    return [response_scaler(gen, bnds[r]) for r in gen['response']]


class SolverTimeoutError(Exception):
    pass

# intersection of two lists, preserves the order in the first list but is not efficient
def list_intersection(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of hvm_utils.list_intersection() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of hvm_utils.list_intersection() is not a list')
    lst = [value for value in lst1 if value in lst2]
    #print('lst', lst)
    return(lst)

# Subtraction of two lists. All occurrences of elements of list2 will be dropped from list1.
# The resulting list will have repetitions if list1 has multiple occurrences of an element not 
# in list2; this behavior is different from function list_intersection_set() below.
def list_subtraction(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of hvm_utils.list_subtraction() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of hvm_utils.list_subtraction() is not a list')
    lst = [value for value in lst1 if value not in lst2]
    return lst


# intersection of two lists thru sets (more efficient but need not preserve the order in lst1).
# All occurrences of elements of list2 will be dropped from list1, and the resulting list will
# not have repeated occurrences of an element,unlike function list_intersection_set() above.
def list_intersection_set(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of hvm_utils.list_intersection_set() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of hvm_utils.list_intersection_set() is not a list')
    lst = list(set.intersection(set(lst1), set(lst2)))
    #print('lst', lst)
    return(lst)



# subtraction of two lists; order in the returned list might be differnt from the prder in lst1
def list_subtraction_set(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of hvm_utils.list_subtraction_set() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of hvm_utils.list_subtraction_set() is not a list')
    #print('input lst1', lst1, 'lst2', lst2)
    lst = list(set(lst1).difference(set(lst2)))
    #print('return lst', lst)
    return lst


# subtraction of two lists
def list_is_subset(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of hvm_utils.list_is_subset() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of hvm_utils.list_is_subset() is not a list')
    for e in lst1:
        if not e in lst2:
            return False
    return True

# unique elements of a list lst if we do not need to preserve the order 
def list_unique_unordered(lst):
    if not isinstance(lst, list) :
        raise Exception('Argument lst of hvm_utils.list_unique_unordered() is not a list')
    return list(set(lst))

# unique elements of a list lst if we do need to preserve the order
def list_unique_ordered(lst):
    if not isinstance(lst, list) :
        raise Exception('Argument lst of hvm_utils.list_unique_ordered() is not a list')
    d = {}
    for x in lst:
        d[x] = 1
    return list(d.keys())

# union of the lists in a list of lists (argument list_of_lists).
# just a concatenation, withot changing the prder or dropping duplicates
def lists_union(list_of_lists):
    return [x for y in list_of_lists for x in y]

# drop duplicate elements, preserve the original order of the remaining 
# elements (with respect to the first occurrence)
def list_remove_duplicates(lst):
    return list(OrderedDict.fromkeys(lst))

# Union (concatenation) of elements of lists in list_of_lists, then droping duplicates.
# Elements from earlier lists in list_of_lists appear first in the resulting list, and 
# duplicates are dropped so that the order of elements is not affected.
def lists_union_order_preserving_without_duplicates(list_of_lists):
    return list_remove_duplicates(lists_union(list_of_lists))