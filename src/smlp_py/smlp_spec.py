import os
import json
from fractions import Fraction

# spec file API
class SmlpSpec:
    def __init__(self):
        self.spec = None
        self.version = None
        
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
        
    def sanity_check_spec(self):
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if self._SPEC_VARIABLE_LABEL not in var_spec.keys():
                raise Exception('A variable does not have the label (name) defined in spec file')
            if self._SPEC_VARIABLE_TYPE not in var_spec.keys():
                raise Exception('Variable ' + str(var_spec[self._SPEC_VARIABLE_LABEL] + 
                    ' does not have type defined in spec file'))
        
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

    @property
    def get_spec_alpha_bounds_dict(self):
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
        return alpha_dict
            
    @property
    def get_spec_eta_grids_dict(self):
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
        return eta_dict
    
    @property
    def get_spec_theta_radii_dict(self):
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
        print('theta_dict', theta_dict)
        return theta_dict   
    
    @property
    def get_spec_domain_dict(self):
        domain_dict = {}
        for var_spec in self.spec:
            print('var_spec', var_spec)
            if var_spec[self._SPEC_VARIABLE_RANGE] == self._SPEC_RANGE_REAL:
                if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                    domain_dict[var_spec['label']] = {'interval': var_spec[self._SPEC_INPUTS_BOUNDS], 
                        self._SPEC_KNOBS_GRID: None}
                else:
                    domain_dict[var_spec['label']] = {'interval': [], self._SPEC_KNOBS_GRID: None}
            elif var_spec[self._SPEC_VARIABLE_RANGE] == self._SPEC_RANGE_INTEGER:
                if self._SPEC_KNOBS_GRID in var_spec.keys():
                    domain_dict[var_spec['label']] = {'interval': None, self._SPEC_KNOBS_GRID: 
                        var_spec[self._SPEC_KNOBS_GRID]}
                else:
                    domain_dict[var_spec['label']] = {'interval': None, self._SPEC_KNOBS_GRID: []}
            else:
                raise Exception('Unsupported variable range ' + str(var_spec[self._SPEC_VARIABLE_RANGE]) + 
                    ' in the spec: value must be ' + self._SPEC_RANGE_REAL + ' or ' + self._SPEC_RANGE_INTEGER)
        print('domain_dict', domain_dict); 
        return domain_dict