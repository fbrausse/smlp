import os
import json
from fractions import Fraction

from smlp_py.smlp_utils import get_expression_variables, list_unique_unordered


# spec file API; in addition, gives access to query constraints' expressions
# and expressions of assertions, queries, optimization objectives specified 
# through command line or through other files.
class SmlpSpec:
    def __init__(self):
        self.spec = None
        self.version = None
        self._alpha_dict = None
        self._eta_dict = None
        self._theta_dict = None
        self._domain_dict = None
        self._asrt_exprs = None
        self._quer_exprs = None
        self._objv_exprs = None
        self._alpha_global_expr = None
        self._beta_global_expr = None 
        
        
        # feilds in sec file defining spec of each variable
        self._SPEC_DICTIONARY_VERSION = 'version'
        self._SPEC_DICTIONARY_SPEC = 'spec'
        
        self._SPEC_VARIABLE_LABEL = None
        self._SPEC_VARIABLE_TYPE = None
        self._SPEC_INPUT_TAG = None
        self._SPEC_KNOB_TAG = None
        self._SPEC_RANGE_INTEGER = 'int'
        self._SPEC_RANGE_REAL = 'float'
        self._SPEC_INPUTS_BOUNDS = None
        self._SPEC_KNOBS_GRID = None
        self._SPEC_KNOBS_ABSOLUTE_RADIUS = None
        self._SPEC_KNOBS_RELATIVE_RADIUS = None
        self._SPEC_VARIABLE_RANGE = None
        self._SPEC_DOMAIN_RANGE_TAG = 'range'
        self._SPEC_DOMAIN_INTERVAL_TAG = 'interval'
        

    def set_spec_tokens(self):
        if self.version == '1.1':
            self._SPEC_VARIABLE_LABEL = 'label'
            self._SPEC_VARIABLE_TYPE = 'type'
            self._SPEC_INPUT_TAG = 'input'
            self._SPEC_KNOB_TAG = 'knob'
            self._SPEC_INPUTS_BOUNDS = 'bounds'
            self._SPEC_KNOBS_GRID = 'grid'
            self._SPEC_KNOBS_ABSOLUTE_RADIUS = 'rad-abs'
            self._SPEC_KNOBS_RELATIVE_RADIUS = 'rad-rel'
            self._SPEC_VARIABLE_RANGE = 'range'
        else:
            raise Exception('Spec version ' + str(self.version) + ' is not supported')
    
    @property
    def get_spec_integer_tag(self):
        return self._SPEC_RANGE_INTEGER
    
    @property
    def get_spec_real_tag(self):
        return self._SPEC_RANGE_REAL
    
    @property
    def get_spec_range_tag(self):
        return self._SPEC_DOMAIN_RANGE_TAG
    
    @property
    def get_spec_interval_tag(self):
        return self._SPEC_DOMAIN_INTERVAL_TAG
    
    # sanity checks on declarations in the spec file
    def sanity_check_spec(self):
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if self._SPEC_VARIABLE_LABEL not in var_spec.keys():
                raise Exception('A variable does not have the label (name) declared in spec file')
            if self._SPEC_VARIABLE_TYPE not in var_spec.keys():
                raise Exception('Variable ' + str(var_spec[self._SPEC_VARIABLE_LABEL] + 
                    ' does not have type declared in spec file'))
            if self._SPEC_VARIABLE_RANGE not in var_spec.keys():
                raise Exception('Variable ' + str(var_spec[self._SPEC_VARIABLE_LABEL] + 
                    ' does not have range declared in spec file'))
            if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                if not var_spec[self._SPEC_VARIABLE_TYPE] in [self._SPEC_INPUT_TAG, self._SPEC_KNOB_TAG]:
                    raise Exception('Domain intervals (bounds) are only supported for free inputs and knobs')
        
    def set_spec_file(self, spec_file):
        if spec_file is None:
            return
        if spec_file is not None:
            if not os.path.isfile(spec_file+'.spec'):
                raise Exception('Spec file ' + str(spec_file) + '.spec' + ' does not exist')
            with open(spec_file+'.spec', 'r') as sf:
                spec_dict = json.load(sf, parse_float=Fraction)
            assert isinstance(spec_dict, dict)
            
            assert set(spec_dict.keys()) == {self._SPEC_DICTIONARY_VERSION, self._SPEC_DICTIONARY_SPEC}
            self.spec = spec_dict[self._SPEC_DICTIONARY_SPEC]; 
            self.version = spec_dict[self._SPEC_DICTIONARY_VERSION]
            print('self.spec', self.spec); print('self.version', self.version)
            self.set_spec_tokens()
            self.sanity_check_spec()

    def set_spec_global_alpha_exprs(self, alph_expr):
        #print('setting alph_expr', alph_expr)
        self._alpha_global_expr = alph_expr
        
    def set_spec_global_beta_exprs(self, beta_expr):
        self._beta_global_expr = beta_expr
        
    def set_spec_asrt_exprs(self, asrt_exprs):
        self._asrt_exprs = asrt_exprs
        
    def set_spec_objv_exprs(self, objv_exprs):
        self._objv_exprs = objv_exprs
        
    def set_spec_quer_exprs(self, quer_exprs):
        self._quer_exprs = quer_exprs
    
    # Create dictionary that maps input variables (free inputs and knobs) to respective
    # domains specified as a closed interval. Other constraints on domains of inputs can
    # be specified using command line option -alpha
    @property
    def get_spec_alpha_bounds_dict(self):
        if self._alpha_dict is not None:
            return self._alpha_dict
        
        alpha_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                assert var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_INPUT_TAG or \
                    var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_KNOB_TAG
                mn = var_spec[self._SPEC_INPUTS_BOUNDS][0]
                mx = var_spec[self._SPEC_INPUTS_BOUNDS][1]
                alpha_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {
                    'min': var_spec[self._SPEC_INPUTS_BOUNDS][0], 
                    'max':var_spec[self._SPEC_INPUTS_BOUNDS][1]}
        print('alpha_dict', alpha_dict)
        
        self._alpha_dict = alpha_dict
        return alpha_dict
     
    # Compute dictionary that maps knobs to respective value grids in the spec file
    @property
    def get_spec_eta_grids_dict(self):
        if self._eta_dict is not None:
            return self._eta_dict
        
        eta_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if var_spec[self._SPEC_VARIABLE_TYPE] != self._SPEC_KNOB_TAG:
                continue
            if self._SPEC_KNOBS_GRID not in var_spec.keys():
                continue
            eta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_KNOBS_GRID]
        print('eta_dict', eta_dict)
        
        self._eta_dict = eta_dict
        return eta_dict
    
    # Compute dictionary that maps knobs to theta radii specified in spec file
    @property
    def get_spec_theta_radii_dict(self):
        if self._theta_dict is not None:
            return self._theta_dict
        
        theta_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if var_spec[self._SPEC_VARIABLE_TYPE] != self._SPEC_KNOB_TAG:
                continue
            if self._SPEC_KNOBS_ABSOLUTE_RADIUS not in var_spec.keys() and \
                self._SPEC_KNOBS_RELATIVE_RADIUS not in var_spec.keys():
                continue
            if self._SPEC_KNOBS_ABSOLUTE_RADIUS in var_spec.keys() and \
                self._SPEC_KNOBS_RELATIVE_RADIUS in var_spec.keys():
                raise Exception('Both absolute and relative radii are specified for variable ' + str(var_spec))
            assert self._SPEC_KNOBS_ABSOLUTE_RADIUS in var_spec.keys() or \
                self._SPEC_KNOBS_RELATIVE_RADIUS in var_spec.keys()
            if self._SPEC_KNOBS_ABSOLUTE_RADIUS in var_spec.keys():
                theta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {
                    self._SPEC_KNOBS_ABSOLUTE_RADIUS: var_spec[self._SPEC_KNOBS_ABSOLUTE_RADIUS], 
                    self._SPEC_KNOBS_RELATIVE_RADIUS: None}
            if self._SPEC_KNOBS_RELATIVE_RADIUS in var_spec.keys():
                theta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {self._SPEC_KNOBS_ABSOLUTE_RADIUS: None, 
                    self._SPEC_KNOBS_RELATIVE_RADIUS: var_spec[self._SPEC_KNOBS_RELATIVE_RADIUS]}
        #print('theta_dict', theta_dict)
        
        self._theta_dict = theta_dict
        return theta_dict   
    
    # Compute dictionary that maps variables to their domains (lower and upper bounds).
    # Domain bounds are usually defined for inputs, but upper or lower bounds may in general be
    # defined on model outputs (responses) to be served as constraints during model exploration,
    # therefore there is no sanity check to ensure the bounds are defined for inputs only.
    @property
    def get_spec_domain_dict(self):
        if self._domain_dict is not None:
            return self._domain_dict
        
        self._domain_dict = {}
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            var_bounds = var_spec[self._SPEC_INPUTS_BOUNDS] if self._SPEC_INPUTS_BOUNDS in var_spec.keys() else None
            var_range = var_spec[self._SPEC_VARIABLE_RANGE]
            if not var_range in [self._SPEC_RANGE_INTEGER, self._SPEC_RANGE_REAL]:
                raise Exception('Unsupported variable range (type) ' + var_spec[self._SPEC_VARIABLE_RANGE])
            self._domain_dict[var_spec['label']] = {'range': var_range, 'interval': var_bounds}
        print('self._domain_dict', self._domain_dict)
        return self._domain_dict
    
    # Compute variables in model exploration constraints -- constraints on model interface
    # (inputs that can be knobs or free inputs, and outputs), assertions, queries, optimization
    # objectives. Some of these constraints are specified through a spec file, some through
    # command line. This function is used to make sure that variables in these constraints are
    # used during model training (and not dropped during data processing or feature selection),
    # therefore the model exploration constraints are well defined on the model interface.
    def get_spec_constraint_vars(self):
        constraints_vars = []
        #print('spec self', self._alpha_global_expr, self._beta_global_expr, self._asrt_exprs)
        if self._alpha_global_expr is not None:
            alph_vars = get_expression_variables(self._alpha_global_expr); #print('alph_vars', alph_vars)
            constraints_vars = constraints_vars + alph_vars
        if self._beta_global_expr is not None:
            beta_vars = get_expression_variables(self._beta_global_expr); #print('beta_vars', beta_vars)
            constraints_vars = constraints_vars + beta_vars
        if self._objv_exprs is not None:
            for objv_expr in self._objv_exprs:
                objv_vars = get_expression_variables(objv_expr); #print('objv_expr', objv_expr)
                constraints_vars = constraints_vars + objv_vars
        if self._asrt_exprs is not None:
            for asrt_expr in self._asrt_exprs:
                asrt_vars = get_expression_variables(asrt_expr); #print('asrt_vars', asrt_vars)
                constraints_vars = constraints_vars + asrt_vars
        if self._quer_exprs is not None:
            for quer_expr in self._quer_exprs:
                quer_vars = get_expression_variables(quer_expr); #print('quer_expr', quer_expr)
                constraints_vars = constraints_vars + quer_vars
        
        return list_unique_unordered(constraints_vars)