#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.


import os, datetime, sys, json
from fractions import Fraction
from collections import OrderedDict
from pandas import DataFrame, concat
from pandas.api.types import is_object_dtype
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import ast
import builtins

        
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

def str_to_float_list(value):
    return [float(x) for x in value.split(',')]

def str_to_str_list(value):
    return [x for x in value.split(',')]

def timed(f, desc=None, log=lambda *args: print(*args, file=sys.stderr)):
    now = datetime.datetime.now()
    r = f()
    t = (datetime.datetime.now() - now).total_seconds()
    if desc is not None:
        log(desc, 'took', t, 'sec')
        return r
    return r, t

'''
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
'''
# intersection of two lists, preserves the order in the first list but is not efficient
def list_intersection(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of utils.list_intersection() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of utils.list_intersection() is not a list')
    lst = [value for value in lst1 if value in lst2]
    #print('lst', lst)
    return(lst)

# Subtraction of two lists. All occurrences of elements of list2 will be dropped from list1.
# The resulting list will have repetitions if list1 has multiple occurrences of an element not 
# in list2; this behavior is different from function list_intersection_set() below.
def list_subtraction(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of utils.list_subtraction() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of utils.list_subtraction() is not a list')
    lst = [value for value in lst1 if value not in lst2]
    return lst


# intersection of two lists thru sets (more efficient but need not preserve the order in lst1).
# All occurrences of elements of list2 will be dropped from list1, and the resulting list will
# not have repeated occurrences of an element,unlike function list_intersection_set() above.
def list_intersection_set(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of utils.list_intersection_set() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of utils.list_intersection_set() is not a list')
    lst = list(set.intersection(set(lst1), set(lst2)))
    #print('lst', lst)
    return(lst)



# subtraction of two lists; order in the returned list might be differnt from the prder in lst1
def list_subtraction_set(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of utils.list_subtraction_set() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of utils.list_subtraction_set() is not a list')
    #print('input lst1', lst1, 'lst2', lst2)
    lst = list(set(lst1).difference(set(lst2)))
    #print('return lst', lst)
    return lst


# subtraction of two lists
def list_is_subset(lst1, lst2):
    if not isinstance(lst1, list) :
        raise Exception('Argument lst1 of utils.list_is_subset() is not a list')
    if not isinstance(lst2, list):
        raise Exception('Argument lst2 of utils.list_is_subset() is not a list')
    for e in lst1:
        if not e in lst2:
            return False
    return True

# unique elements of a list lst if we do not need to preserve the order; also, the order 
# (and the result) might not be deterministic
def list_unique_unordered(lst):
    if not isinstance(lst, list) :
        raise Exception('Argument lst of utils.list_unique_unordered() is not a list')
    return list(set(lst))

# unique elements of a list lst if we do need to preserve the order
def list_unique_ordered(lst):
    if not isinstance(lst, list) :
        raise Exception('Argument lst of utils.list_unique_ordered() is not a list')
    d = {}
    for x in lst:
        d[x] = 1
    return list(d.keys())

# union of the lists in a list of lists (argument list_of_lists).
# just a concatenation, withot changing the order or dropping duplicates
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

# determaine whether pandas dataframe column is numeric using numpy's mp.number type
def pd_df_col_is_numeric(df, col_name):
    res = np.issubdtype(df[col_name].dtype, np.number)
    if res:
        assert df[col_name].dtype in [int, float]
    return res

def pd_df_col_is_categorical(df, col_name):
    return is_object_dtype(df[col_name])

# given an algo name like dt and the hyper parameter dictionary param_dict  
# this function returns a modified dictionary obtained from param_dictby by 
# adding algo name to the parameter name and its abbriviated name in param_dict.
def param_dict_with_algo_name(param_dict, algo):
    #print('param_dict', param_dict)
    result_dict = {}
    for k, v in param_dict.items():
        v_updated = v.copy()
        v_updated['abbr'] = algo + '_' + v['abbr']
        result_dict[algo + '_' + k] = v_updated
    return result_dict

# Whether a response is numeric or binary. Numeric responses are defined as ones
# that have float values or int values besides 0 and 1. Responses with values 0 or 1 
# are defined as binary, It is expected that if response vas of of type object/string
# with 1 or 2 values then these value have been renames to 1 and 0 based of definition
# of which one of these two strings is positive and which one is nagative, and based
# on smlp postive and negative values declared by user (where by default 1 is positive 
# and 0 is negative)
def get_response_type(df:DataFrame, resp_name:str):
    assert resp_name in df.columns.tolist()
    #print('df.shape', df.shape)
    if df.shape[1] > 1:
        resp_type = df.dtypes[resp_name]
    else:
        resp_type = df.dtypes[0]
    #print('response dtype', resp_type)
    resp_vals = set(df[resp_name].tolist()); #print('resp_vals', resp_vals, resp_vals.issubset({0,1}))
    if resp_type == int and resp_vals.issubset({0,1}):
        return "classification"
    elif resp_type == float:
        return "regression"
    else:
        raise Exception('Classification vs regression mode cannot be determined for response ' + str(resp_name))
                              

def rows_dict_to_df(rows_dict, colnames, index=True):
    df = None
    for rowname, row in rows_dict.items():
        if index:
            row_df = DataFrame([row], index=[rowname], columns=colnames);  #print('row_df\n', row_df)
        else:
            row_df = DataFrame([row], columns=colnames);  #print('row_df\n', row_df)
         
        if df is None:
            df = row_df
        else:
            df = concat([df, row_df], axis=0)
    return df

def cast_type(obj, tp):
    #print('obj', obj, 'source tp', type(obj), 'target tp', tp)
    if tp == int:
        res = int(obj)
    elif tp == float:
        res = float(obj)
    elif tp == str:
        res = str(obj)
    else:
        raise Exception('Unsupported type ' + str(tp) + ' in function cast_type')
    #print('res', type(res), res)
    if tp in [int, float]:
        if np.isnan(res) or res is None:
            raise Exception('Casting of ' + str(obj) + ' to type ' + str(tp) + ' failed')
    return res


# This function is copied from 
# https://stackoverflow.com/questions/68390248/ast-get-the-list-of-only-the-variable-names-in-an-expression
# It computes variable names in python expression. This function does not properly cover all constructs 
# allowed within general python expressions; e.g., loops (loop iteration variables will be returned
# among the leaf variables in the expression).
def get_expression_variables(expression):
    tree = ast.parse(expression)
    variables = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            variables.append(node.id)
    return [v for v in set(variables) if v not in vars(builtins)]

'''
code to control caching dynamiccally
import functools
import smlp

def conditional_cache(use_cache):
    def decorator(func):
        if use_cache:
            return functools.cache(func)
        else:
            return func
    return decorator

class MyClass:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache

    @property
    def smlp_true(self):
        decorator = conditional_cache(self.use_cache)
        return decorator(lambda: smlp.true)()

    @property
    def smlp_false(self):
        decorator = conditional_cache(self.use_cache)
        return decorator(lambda: smlp.false)()

# Example usage
# Enable caching
my_instance = MyClass(use_cache=True)
result_true = my_instance.smlp_true
result_false = my_instance.smlp_false

# Disable caching
my_instance.use_cache = False
# Now, the properties will not use cache

'''

