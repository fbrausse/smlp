import os
import json
from fractions import Fraction

from smlp_py.smlp_utils import get_expression_variables, list_unique_unordered, np_JSONEncoder


# spec file API; in addition, gives access to query constraints' expressions and expressions of assertions, queries,
# optimization objectives specified through command line or through other files.
class SmlpSpec:
    def __init__(self):
        self._spec_logger = None       # logger
        self.spec = None               # the full spec including version, interface spec, as well as global constraints
                                       #     alpha, beta, assertions, queries, optimization/tuning objectives
        self.version = None            # version of spec format, to support backward compatibility when changing the format
        self._alpha_ranges_dict = None # variable ranges defined using the self._SPEC_INPUTS_BOUNDS field in spec file
        self._eta_dict = None          # control (knob) input variable grid of allowed values (centre point values)
        self._eta_ranges_dict = None   # 
        self._theta_dict = None        # stability radii for knob input variables
        self._domain_dict = None       # type and range of the interface variables (mainly free inputs and knobs), defined
                                       #     using spec field self._SPEC_INPUTS_BOUNDS and supporting integer and real types
                                       #     self._SPEC_RANGE_INTEGER, self._SPEC_RANGE_REAL
        self.radius_absolute = None
        self.radius_relative = None
        self.delta_absolute = None
        self.delta_relative = None
        
        self._DEF_DELTA_RELATIVE = 0.01 
        self._DEF_DELTA_ABSOLUTE = 0.0 
        self._DEF_ALPHA = None
        self._DEF_BETA = None
        self._DEF_ETA = None 
        self._DEF_RADIUS_RELATIVE = None
        self._DEF_RADIUS_ABSOLUTE = None
        
        self.spec_params_dict = {
            'spec': {'abbr':'spec', 'default':None, 'type':str,
                'help':'Name of spec file including full path, must be provided [default None]'}, 
            'delta_relative': {'abbr':'delta_rel', 'default':self._DEF_DELTA_RELATIVE, 'type':float, 
                'help':'exclude (1+DELTA)*radius region for non-grid components ' +
                    '[default: {}]'.format(str(self._DEF_DELTA_RELATIVE))},
            'delta_absolute': {'abbr':'delta_abs', 'default':self._DEF_DELTA_ABSOLUTE, 'type':float, 
                'help':'exclude (1+DELTA)*radius region for non-grid components ' +
                    '[default: {}]'.format(str(self._DEF_DELTA_ABSOLUTE))},
            'radius_relative': {'abbr':'rad_rel', 'default':self._DEF_RADIUS_RELATIVE, 'type':float, 
                'help':'Relative radius, in terms of percentage of the value of the knob to which it applies ' +
                    'to compute the absolute radius to be used in theta (stability) constraint. Overrides relative ' +
                    'radius value specified in the spec file [default: {}]'.format(str(self._DEF_RADIUS_RELATIVE))},
            'radius_absolute': {'abbr':'rad_abs', 'default':self._DEF_RADIUS_ABSOLUTE, 'type':float, 
                'help':'Absolute value of radius to be used in theta (stability) constraint. Override relative ' +
                    'radius value specified in the spec file [default: {}]'.format(str(self._DEF_RADIUS_ABSOLUTE))},
            'alpha': {'abbr':'alpha', 'default':self._DEF_ALPHA, 'type':str, 
                'help':'constraints on model inputs (free inputs or configuration knobs) ' +
                    '[default: {}]'.format(str(self._DEF_ALPHA))},
            'beta': {'abbr':'beta', 'default':self._DEF_BETA, 'type':str, 
                'help':'constraints on model outputs, relevant for "optimize" mode only ' +
                    '(when selecting model configuration that are safe and near-optimal) ' +
                    '[default: {}]'.format(str(self._DEF_BETA))},
            'eta': {'abbr':'eta', 'default':self._DEF_ETA, 'type':str, 
                'help':'global constraints on/accross knobs that define legal configurations of knobs ' +
                    'during search for optimal configurations in "optimize" and "optsyn" modes ' +
                    '[default: {}]'.format(str(self._DEF_ETA))}
        }

        # assertions -- specified through command line (_cmdl_), spec file (_spec_), and final definition of assertions
        # self._asrt_dict, currently obtained by overriding spec definitions (if any) with command line definitions (if any).
        self._asrt_cmdl_dict = None
        self._asrt_spec_dict = None
        self._asrt_dict = None
        
        # queries -- specified through command line (_cmdl_), spec file (_spec_), and final definition of queries
        # self._quer_dict, currently obtained by overriding spec definitions (if any) with command line definitions (if any).
        self._quer_cmdl_dict = None
        self._quer_spec_dict = None
        self._quer_dict = None
        
        # objectives -- specified through command line (_cmdl_), spec file (_spec_), and final definition of objectives
        # self._objv_dict, currently obtained by overriding spec definitions (if any) with command line definitions (if any).
        self._objv_cmdl_dict = None
        self._objv_spec_dict = None
        self._objv_dict = None
        
        # alpha -- global constraints on inputs only (free inputs and / or knobs), cmdl, spec and final versions 
        # defined as self._alpha_global_expr by overriding cmdl spec definition self._alpha_cmdl_expr (if any) 
        # with command line definition self._alpha_cmdl_expr (if any).
        self._alpha_global_expr = None
        self._alpha_cmdl_expr = None
        self._alpha_spec_expr = None
        
        # beta -- global constraints on interface -- inputs (free inputs and / or knobs), and outputs (responses) of
        # the model, to be satisfied during optimization and tuning tasks (not relevant for verification and querying).
        # defined as self._beta_global_expr by overriding cmdl spec definition self._beta_cmdl_expr (if any) 
        # with command line definition self._beta_cmdl_expr (if any).
        self._beta_global_expr = None 
        self._beta_cmdl_expr = None
        self._beta_spec_expr = None
        
        # eta -- global constraints on knobs (in addition to eta constraints derived from knob ranges and grids).
        self._eta_global_expr = None
        
        # witnesses to queries, for certify mode; supposed to include assignements to inputs and knobs
        self._witn_dict = None
        
        # configurations to assertions, for verify mode; supposed to include assignements to knobs only (not inputs)
        self._config_dict = None
        
        # feilds in spec file defining specification of each variable
        self._SPEC_DICTIONARY_VERSION = 'version'
        self._SPEC_DICTIONARY_SPEC = 'spec'
        self._SPEC_DICTIONARY_ASSERTIONS = 'assertions'
        self._SPEC_DICTIONARY_QUERIES = 'queries'
        self._SPEC_DICTIONARY_OBJECTIVES = 'objectives'
        self._SPEC_DICTIONARY_WITNESSES = 'witnesses'
        self._SPEC_DICTIONARY_CONFIGS = 'configurations'
        self._SPEC_DICTIONARY_ALPHA = 'alpha' # global constraints on inputs -- knobs and free inputs
        self._SPEC_DICTIONARY_BETA = 'beta' # global interface costraints to be valid during optimization and tuning
        self._SPEC_DICTIONARY_ETA = 'eta' # global constraints on knobs (during candidate search)
        self._SPEC_DICTIONARY_SYSTEM_TAG = 'system' 
        
        self._SPEC_VARIABLE_LABEL = None
        self._SPEC_VARIABLE_TYPE = None
        self._SPEC_INPUT_TAG = None
        self._SPEC_KNOB_TAG = None
        self._SPEC_OUTPUT_TAG = 'response'
        self._SPEC_RANGE_INTEGER = 'int'
        self._SPEC_RANGE_REAL = 'float'
        self._SPEC_INPUTS_BOUNDS = None
        self._SPEC_KNOBS_GRID = None
        self._SPEC_KNOBS_ABSOLUTE_RADIUS = None
        self._SPEC_KNOBS_RELATIVE_RADIUS = None
        self._SPEC_VARIABLE_RANGE = None
        self._SPEC_DICTIONARY_SYSTEM = None
        
        # tracks changes ib spec file token names per spec version
        self._spec_tokens_dict = {
            'SPEC_VARIABLE_LABEL' : {'1.1': 'label', '1.2': 'label'},
            'SPEC_VARIABLE_TYPE' : {'1.1': 'type', '1.2': 'interface'},
            'SPEC_INPUT_TAG' : {'1.1': 'input', '1.2': 'input'},
            'SPEC_KNOB_TAG' : {'1.1': 'knob', '1.2': 'knob'},
            'SPEC_OUTPUT_TAG' : {'1.1': 'response', '1.2': 'output'},
            'SPEC_INPUTS_BOUNDS' : {'1.1': 'bounds', '1.2': 'range'},
            'SPEC_KNOBS_GRID' : {'1.1': 'grid', '1.2': 'grid'},
            'SPEC_KNOBS_ABSOLUTE_RADIUS' : {'1.1': 'rad-abs', '1.2': 'rad-abs'},
            'SPEC_KNOBS_RELATIVE_RADIUS' : {'1.1': 'rad-rel', '1.2': 'rad-rel'},
            'SPEC_VARIABLE_RANGE' : {'1.1': 'range', '1.2': 'type'},
            'SPEC_RANGE_REAL' : {'1.1': 'float', '1.2': 'real'},
            'SPEC_DICTIONARY_SPEC' : {'1.1': 'spec', '1.2': 'variables'}
        }
        
    # used by external scripts to set spec version while generating a spec file
    def set_spec_version(self, version:str):
        self.version = version
    
        
    # tokens used in spec file per version -- might change from version to version, required for backward compatibility.
    def set_spec_tokens(self):
        if self.version == '1.1':
            self._SPEC_VARIABLE_LABEL = 'label'
            self._SPEC_VARIABLE_TYPE = 'type'
            self._SPEC_INPUT_TAG = 'input'
            self._SPEC_KNOB_TAG = 'knob'
            self._SPEC_OUTPUT_TAG = 'response'
            self._SPEC_INPUTS_BOUNDS = 'bounds'
            self._SPEC_KNOBS_GRID = 'grid'
            self._SPEC_KNOBS_ABSOLUTE_RADIUS = 'rad-abs'
            self._SPEC_KNOBS_RELATIVE_RADIUS = 'rad-rel'
            self._SPEC_VARIABLE_RANGE = 'range'
            self._SPEC_RANGE_REAL = 'float'
            self._SPEC_DICTIONARY_SPEC = 'spec'
        elif self.version == '1.2':
            self._SPEC_VARIABLE_LABEL = 'label'
            self._SPEC_VARIABLE_TYPE = 'interface'
            self._SPEC_INPUT_TAG = 'input'
            self._SPEC_KNOB_TAG = 'knob'
            self._SPEC_OUTPUT_TAG = 'output'
            self._SPEC_INPUTS_BOUNDS = 'range'
            self._SPEC_KNOBS_GRID = 'grid'
            self._SPEC_KNOBS_ABSOLUTE_RADIUS = 'rad-abs'
            self._SPEC_KNOBS_RELATIVE_RADIUS = 'rad-rel'
            self._SPEC_KNOBS_ABSOLUTE_DELTA = 'delta-abs'
            self._SPEC_KNOBS_RELATIVE_DELTA = 'delta-rel'
            self._SPEC_VARIABLE_RANGE = 'type'
            self._SPEC_RANGE_REAL = 'real'
            self._SPEC_DICTIONARY_SPEC = 'variables'
        else:
            raise Exception('Spec version ' + str(self.version) + ' is not supported')
    
    def set_logger(self, logger):
        self._spec_logger = logger 
    
    @property
    def get_spec_integer_tag(self):
        return self._SPEC_RANGE_INTEGER
    
    @property
    def get_spec_real_tag(self):
        return self._SPEC_RANGE_REAL
        
    # sanity checks on declarations in the spec file
    def sanity_check_spec(self):
        # eta global and grid constraints can only be defined on knobs
        eta_expr = self.get_spec_global_eta_expr
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
            if var_spec[self._SPEC_VARIABLE_TYPE] != self._SPEC_KNOB_TAG:
                if self._SPEC_KNOBS_GRID in var_spec.keys():
                    raise Exception('ETA grid constraint can only be defined for knobs')
                if self._SPEC_KNOBS_ABSOLUTE_RADIUS in var_spec.keys() or  \
                    self._SPEC_KNOBS_RELATIVE_RADIUS in var_spec.keys():
                    raise Exception('THETA radius can only be defined for knobs')
                if eta_expr is not None:
                    if var_spec[self._SPEC_VARIABLE_LABEL] in get_expression_variables(eta_expr):
                        self._spec_logger.error('ETA constraint ' + str(eta_expr) + 
                            ' contains a non-knob variable ' + str() + '; aborting...')
                        raise Exception('ETA global constraint can only contain knobs')
        
        # sanity check witnesses: query names in witnesses specification must be among query names
        if self.get_spec_witn_dict is not None and self.get_spec_quer_exprs_dict is not None:
            witn_queries = set(self.get_spec_witn_dict.keys())
            queries = set(self.get_spec_quer_exprs_dict)
            if not witn_queries.issubset(queries):
                raise Exception('Query names ' + str(witn_queries.difference(queries)) + 
                    ' used for specifying witnesses are not specified among the queries ' + str(queries))
                
        # sanity check knob cinfigs: assertion names in knob config specification must be among  assertion names
        if self.get_spec_config_dict is not None and self.get_spec_asrt_exprs_dict is not None:
            config_asserts = set(self.get_spec_config_dict.keys())
            asserts = set(self.get_spec_asrt_exprs_dict)
            if not config_asserts.issubset(asserts):
                raise Exception('Assertion names ' + str(config_asserts.difference(asserts)) + 
                    ' used for specifying knob configurations are not specified among the assertions ' + str(asserts))
    
    # upgrade spec file to a new version: from version source_version to version target_version.
    # write the new spec file in the same directory where the spec_file was located.
    def upgrade_spec(self, spec_file, source_version, target_version):
        if not os.path.isfile(spec_file): #+'.spec'
            raise Exception('Spec file ' + str(spec_file) + ' does not exist') #'.spec' + 
        with open(spec_file, 'r') as sf: #+'.spec'
            spec_dict = json.load(sf) #, parse_float=Fraction
        spec_new_dict = {}
        #print('varoiables ?', self._SPEC_DICTIONARY_SPEC)
        #print('spec tokens dict vals', self._spec_tokens_dict.values())
        def upgrade_token(token):
            for version_vals in self._spec_tokens_dict.values():
                #print('version_vals', version_vals)
                token_val_source = version_vals[source_version]; #print('token_val_source', token_val_source)
                token_val_target = version_vals[target_version]; #print('token_val_target', token_val_target)
                if token_val_source == token:
                    #print('match', token)
                    return token_val_target
        
        for k,v in spec_dict.items():
            #print('k', k, 'v', v)
            if k == self._SPEC_DICTIONARY_VERSION:
                spec_new_dict[self._SPEC_DICTIONARY_VERSION] = target_version
            elif k == self._SPEC_DICTIONARY_SPEC:
                vars = []
                for var_spec in spec_dict[self._SPEC_DICTIONARY_SPEC]:
                    var_spec_new = {}
                    for var_spec_token, var_spec_val in var_spec.items():
                        #print('var_spec_token', var_spec_token, 'var_spec_val', var_spec_val)
                        var_spec_token_target = upgrade_token(var_spec_token)
                        var_spec_val_target = upgrade_token(var_spec_val)
                        if var_spec_val_target is None:
                            var_spec_val_target = var_spec_val
                        #print('replace', var_spec_token, 'with', var_spec_token_target)
                        #print('replace', var_spec_val, 'with', var_spec_val_target)
                        var_spec_new[var_spec_token_target] = var_spec_val_target
                    #print('var_spec_new', var_spec_new)
                    vars.append(var_spec_new)
                spec_new_dict[self._spec_tokens_dict['SPEC_DICTIONARY_SPEC'][target_version]] = vars
            else:
                spec_new_dict[k] = v
        #print('spec_new_dict', spec_new_dict); 
        with open(spec_file+'_v{}.spec'.format('1.2'), 'w') as f:
            json.dump(spec_new_dict, f, indent='\t', cls=np_JSONEncoder)
        return spec_new_dict
                    
        
    def set_spec_file(self, spec_file):
        if spec_file is None:
            return
        if spec_file is not None:
            if not os.path.isfile(spec_file): #+'.spec'
                raise Exception('Spec file ' + str(spec_file) + ' does not exist') #'.spec' + 
            with open(spec_file, 'r') as sf: #+'.spec'
                spec_dict = json.load(sf, parse_float=Fraction)
            #print('spec_dict loaded\n', spec_dict)
            assert isinstance(spec_dict, dict)
            self.set_spec_version(spec_dict[self._SPEC_DICTIONARY_VERSION])
            self.set_spec_tokens()
            assert self._SPEC_DICTIONARY_VERSION in spec_dict.keys()
            assert self._SPEC_DICTIONARY_SPEC in spec_dict.keys()
            #print(set(spec_dict.keys()))
            assert (set(spec_dict.keys())).issubset({self._SPEC_DICTIONARY_VERSION, self._SPEC_DICTIONARY_SPEC,
                self._SPEC_DICTIONARY_ASSERTIONS, self._SPEC_DICTIONARY_OBJECTIVES, self._SPEC_DICTIONARY_QUERIES,
                self._SPEC_DICTIONARY_ALPHA, self._SPEC_DICTIONARY_BETA, self._SPEC_DICTIONARY_ETA, 
                self._SPEC_DICTIONARY_SYSTEM_TAG, self._SPEC_DICTIONARY_WITNESSES, self._SPEC_DICTIONARY_CONFIGS})
            self.spec_dict = spec_dict
            self.spec = spec_dict[self._SPEC_DICTIONARY_SPEC]; 
            self.version = spec_dict[self._SPEC_DICTIONARY_VERSION]
            self._spec_logger.info('Model exploration specification:\n' + str(self.spec_dict))
            #self._spec_logger.info(json.stringif(self.spec_dict, ensure_ascii=False, indent='\t', cls=np_JSONEncoder)) #parse_float=Fraction
            #self.set_spec_tokens()
            self.sanity_check_spec()
            #print(self._SPEC_DICTIONARY_SPEC); 
            #self.upgrade_spec(spec_file, '1.1', '1.2')

    # Override relative and absolute radii values supplied in the spec file
    def set_radii(self, rad_abs, rad_rel):
        self.radius_relative = rad_rel
        self.radius_absolute = rad_abs
    
    # Override relative and absolute radii values supplied in the spec file
    def set_deltas(self, rad_abs, rad_rel):
        self.delta_relative = rad_rel
        self.delta_absolute = rad_abs
        
    # file specifying a witness per query
    #def set_spec_witness_file(self, witness_file):
    #    self._witness_file = witness_file
    
    # API to compute thelist of responses in spec
    @property
    def get_spec_responses(self):
        return [var_spec[self._SPEC_VARIABLE_LABEL] for var_spec in self.spec if 
            var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_OUTPUT_TAG]
    
    # API to compute the list of features in spec: free inputs and control inputs (knobs)
    @property
    def get_spec_features(self):
        return [var_spec[self._SPEC_VARIABLE_LABEL] for var_spec in self.spec if 
            var_spec[self._SPEC_VARIABLE_TYPE] in [self._SPEC_INPUT_TAG, self._SPEC_KNOB_TAG]]
    
    # API to compute list of knobs in spec
    @property
    def get_spec_knobs(self):
        return [var_spec[self._SPEC_VARIABLE_LABEL] for var_spec in self.spec if 
            var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_KNOB_TAG]
    
    # API to compute the list of inputs in spec
    @property
    def get_spec_inputs(self):
        return [var_spec[self._SPEC_VARIABLE_LABEL] for var_spec in self.spec if 
            var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_INPUT_TAG]
    
    # anonymize system/model interface names -- knobs, inputs, outputs.
    # currently is used to anonymize trace logs for debugging.
    @property
    def get_anonymized_interface(self):
        def anomym_dict(var, name_pref):
            return dict(zip(sorted(var), [name_pref + str(i) for i in range(len(var))]))
        anonymize_dict = {'knobs':{}, 'inputs':{}, 'outputs':{}}
        anonymize_dict['knobs'] = anomym_dict(self.get_spec_knobs, 'p')
        anonymize_dict['inputs'] = anomym_dict(self.get_spec_inputs, 'x')
        anonymize_dict['outputs'] = anomym_dict(self.get_spec_responses, 'y')
        return anonymize_dict
        
    # API to get definition of the original system (that SMLP intends to model with ML).
    # If provided, it is a string that correponds to python expression of the system's funcion.
    # The feild is not mondatory (we do not always know or want to use definition of the 
    # original system), and the function returns None if system is not specified in the spec.
    @property
    def get_spec_system(self):
        if self._SPEC_DICTIONARY_SYSTEM_TAG not in self.spec_dict.keys():
            return None
        assert isinstance(self.spec_dict[self._SPEC_DICTIONARY_SYSTEM_TAG], dict)
        self._SPEC_DICTIONARY_SYSTEM = self.spec_dict[self._SPEC_DICTIONARY_SYSTEM_TAG]
        return self._SPEC_DICTIONARY_SYSTEM
    
    # Sanity check of spec in verification mode - all knobs must be asigned fixed value,
    # spec must contain assertions and should not contain beta constraints.
    # In verification mode the knobs are allowed (to consider verification with stability)
    # but value of each knob should eveluate to a constant bsed on the knob constraints
    # defined in spec file. As part of methodology (and to make things simple), we require
    # constant assignement to knobs to be defined through singleton grid or through range
    # that only contains one value (that is, max and min bounds must equeal, e.g., [5,5]).
    # it is possible to assign fixed values to knobs using global eta expressions, say using
    # firmula knob1=5 and knob2=3, which assigns values 5 and 3 to knobs knob1 and knob2,
    # but we want to avoid extra check for global eta constrainst required to understand 
    # whether all knobs are forced to exacly one value, thus the above methodology convention.
    # Constant values to knobs can also be specified in the spec file using 'configurations' 
    # field. The first two sanity checks in this function must be performed anyway, while  
    # the rest only when "configurations" field is not specified in the spec file. 
    def sanity_check_verification_spec(self):
        if self.get_spec_asrt_exprs_dict is None:
            raise Exception('Assertions are not specified in "verify" mode: aborting...')
        if self.get_spec_beta_global_expr is not None:
            raise Exception('Beta constraints are not supported in "verify" mode: aborting...')
        spec_knobs = self.get_spec_knobs
        non_constant_knobs = []
        knobs_without_range_and_grid = []
        witness = {}
        for var_spec in self.spec:
            if not var_spec[self._SPEC_VARIABLE_LABEL] in spec_knobs:
                continue
            #print('knob spec', var_spec)
            if self._SPEC_KNOBS_GRID in var_spec.keys():
                if len(var_spec[self._SPEC_KNOBS_GRID]) == 0:
                    raise Exception('Knob' + str(self._SPEC_VARIABLE_LABEL) + ' has an empty grid')
                if len(var_spec[self._SPEC_KNOBS_GRID]) > 1:
                    non_constant_knobs.append(var_spec[self._SPEC_VARIABLE_LABEL])
                else:
                    witness[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_KNOBS_GRID][0]
                continue
            elif self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                if var_spec[self._SPEC_INPUTS_BOUNDS][0] != var_spec[self._SPEC_INPUTS_BOUNDS][1]:
                    non_constant_knobs.append(var_spec[self._SPEC_VARIABLE_LABEL])
                else:
                    #var_spec[self._SPEC_KNOBS_GRID] = [var_spec[self._SPEC_INPUTS_BOUNDS][0]]
                    witness[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_INPUTS_BOUNDS][0]
                continue
            else:
                knobs_without_range_and_grid.append(var_spec[self._SPEC_VARIABLE_LABEL])
        
        if len(non_constant_knobs) > 0:
            raise Exception('Knobs ' + str(non_constant_knobs) + ' are not assigned constant values as part of specification, in "verify" mode: aborting...')
        if len(knobs_without_range_and_grid) > 0:
            raise Exception('Knobs ' + str(knobs_without_range_and_grid) + ' have neither ranges nor grids specified, in "varify" mode: aborting...')
        return witness
            
    # Similar to sanity check for spec file in verify mode.
    # Knob values in this case can be provided is spec file using
    # the "witnesses" field instead of the "configurations" field.
    def sanity_check_certification_spec(self):
        #print(self.get_spec_witn_dict)
        if self.get_spec_witn_dict is not None:
            return self.get_spec_witn_dict
        non_constant_knobs_inputs = []
        knobs_without_range_and_grid = []
        witness = {}
        for var_spec in self.spec:
            if not var_spec[self._SPEC_VARIABLE_LABEL] in self.get_spec_knobs +  self.get_spec_inputs:
                continue
            #print('knob spec', var_spec)
            if self._SPEC_KNOBS_GRID in var_spec.keys():
                if len(var_spec[self._SPEC_KNOBS_GRID]) == 0:
                    raise Exception('Knob' + str(self._SPEC_VARIABLE_LABEL) + ' has an empty grid')
                if len(var_spec[self._SPEC_KNOBS_GRID]) > 1:
                    non_constant_knobs_inputs.append(var_spec[self._SPEC_VARIABLE_LABEL])
                else:
                    witness[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_KNOBS_GRID][0]
                continue
            elif self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                if var_spec[self._SPEC_INPUTS_BOUNDS][0] != var_spec[self._SPEC_INPUTS_BOUNDS][1]:
                    non_constant_knobs_inputs.append(var_spec[self._SPEC_VARIABLE_LABEL])
                else:
                    #var_spec[self._SPEC_KNOBS_GRID] = [var_spec[self._SPEC_INPUTS_BOUNDS][0]]
                    witness[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_INPUTS_BOUNDS][0]
                continue
            else:
                knobs_without_range_and_grid.append(var_spec[self._SPEC_VARIABLE_LABEL])
        
        if len(non_constant_knobs_inputs) > 0:
            raise Exception('Knobs/Inputs ' + str(non_constant_knobs_inputs) + ' are not assigned constant values as part of specification, in "certify" mode: aborting...')
        if len(knobs_without_range_and_grid) > 0:
            raise Exception('Knobs ' + str(knobs_without_range_and_grid) + ' have neither ranges nor grids specified, in "certify" mode: aborting...')
        return witness
    
    # API to extract from spec a global alpha constraint defined using feild "alpha"
    @property
    def get_spec_alpha_global_expr(self):
        if self._alpha_cmdl_expr is not None:
            alpha_expr = self._alpha_cmdl_expr
            #print('get_spec_alpha_global_expr  1', alpha_expr)
        elif self._alpha_spec_expr is not None:
            alpha_expr = self._alpha_spec_expr
            #print('get_spec_alpha_global_expr  2', alpha_expr)
        elif self._SPEC_DICTIONARY_ALPHA in self.spec_dict.keys():
            assert isinstance(self.spec_dict[self._SPEC_DICTIONARY_ALPHA], str)
            alpha_expr = self.spec_dict[self._SPEC_DICTIONARY_ALPHA]
            self._alpha_spec_expr = alpha_expr
            self._alpha_global_expr = alpha_expr
            #print('get_spec_alpha_global_expr  3', alpha_expr)
        else:
            #print('get_spec_alpha_global_expr  4  -- None')
            assert self._alpha_cmdl_expr is None
            assert self._alpha_spec_expr is None
            assert self._alpha_global_expr is None
            alpha_expr = None
            
        #print('alpha_expr', alpha_expr, 'cmdl', self._alpha_cmdl_expr, 'spec', self._alpha_spec_expr, 'glbl', self._alpha_global_expr)
        return alpha_expr

    # API to extract from spec a global beta constraint defind using feild "beta"
    @property
    def get_spec_beta_global_expr(self):
        if self._beta_cmdl_expr is not None:
            beta_expr = self._beta_cmdl_expr
        elif self._beta_spec_expr is not None:
            beta_expr = self._beta_spec_expr
        elif self._SPEC_DICTIONARY_BETA in self.spec_dict.keys():
            assert isinstance(self.spec_dict[self._SPEC_DICTIONARY_BETA], str)
            beta_expr = self.spec_dict[self._SPEC_DICTIONARY_BETA]
            self._beta_spec_expr = beta_expr
            self._beta_global_expr = beta_expr
        else:
            assert self._beta_cmdl_expr is None
            assert self._beta_spec_expr is None
            assert self._beta_global_expr is None
            beta_expr = None
            
        #print('beta_expr', beta_expr, 'cmdl', self._beta_cmdl_expr, 'spec', self._beta_spec_expr, 'glbl', self._beta_global_expr)
        return beta_expr

    def get_cmdl_assertions(self, arg_asrt_names, arg_asrt_exprs, cmdl_cond_sep):
        if arg_asrt_exprs is None:
            return None, None, None
        else:
            asrt_exprs = arg_asrt_exprs.split(cmdl_cond_sep)
            if arg_asrt_names is not None:
                asrt_names = arg_asrt_names.split(',')
            else:
                asrt_names = ['asrt_'+str(i) for i in enumerate(len(asrt_exprs))];
        assert asrt_names is not None and asrt_exprs is not None
        assert len(asrt_names) == len(asrt_exprs); 
        #print('asrt_names', asrt_names); print('asrt_exprs', asrt_exprs)
        self._asrt_cmdl_dict = dict(zip(asrt_names, asrt_exprs))
        self._asrt_dict = self._asrt_cmdl_dict
        return dict(zip(asrt_names, asrt_exprs)), asrt_names, asrt_exprs
    
    def get_cmdl_queries(self, arg_query_names, arg_query_exprs, cmdl_cond_sep):
        if arg_query_exprs is None:
            return None, None, None
        else:
            query_exprs = arg_query_exprs.split(cmdl_cond_sep)
            if arg_query_names is not None:
                query_names = arg_query_names.split(',')
            else:
                query_names = ['query_'+str(i) for i in enumerate(len(query_exprs))];
        assert query_names is not None and query_exprs is not None
        assert len(query_names) == len(query_exprs); 
        #print('query_names', query_names); print('query_exprs', query_exprs)
        self._quer_cmdl_dict = dict(zip(query_names, query_exprs))
        self._quer_dict = self._quer_cmdl_dict
        return self._quer_dict, query_names, query_exprs
    
    # When smlp mode is optimize, objectives must be defined. If they are not provided, the default is to use
    # the reponses as objectives, and the names of objectives are names of the responses prefixed bu 'objv_'.
    def get_cmdl_objectives(self, arg_objv_names, arg_objv_exprs, resp_names, cmdl_cond_sep):
        if arg_objv_exprs is None:
            return None, None, None
            #objv_exprs = resp_names
            #objv_names = ['objv_' + resp_name for resp_name in resp_names]
        else:
            objv_exprs = arg_objv_exprs.split(cmdl_cond_sep)
            if arg_objv_names is not None:
                objv_names = arg_objv_names.split(',')
            else:
                objv_names = ['objv_'+str(i) for i in enumerate(len(objv_exprs))];
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs); 
        #print('objv_names', objv_names); print('objv_exprs', objv_exprs)
        self._objv_cmdl_dict = dict(zip(objv_names, objv_exprs))
        self._objv_dict = self._objv_cmdl_dict
        return self._objv_dict, objv_names, objv_exprs

    @property
    def get_spec_asrt_exprs_dict(self):
        if self._asrt_cmdl_dict is not None:
            return self._asrt_cmdl_dict
        if self._SPEC_DICTIONARY_ASSERTIONS in self.spec_dict.keys():
            self._asrt_spec_dict = self.spec_dict[self._SPEC_DICTIONARY_ASSERTIONS]
            self._asrt_dict = self._asrt_spec_dict
            return self._asrt_dict
        else:
            return None
    
    @property
    def get_spec_quer_exprs_dict(self):
        if self._quer_cmdl_dict is not None:
            return self._quer_cmdl_dict
        if self._SPEC_DICTIONARY_QUERIES in self.spec_dict.keys():
            self._spec_quer_exprs = list(self.spec_dict[self._SPEC_DICTIONARY_QUERIES].values())
            self._quer_spec_dict = self.spec_dict[self._SPEC_DICTIONARY_QUERIES]
            self._quer_dict = self._quer_spec_dict
            return self._quer_dict
        else:
            return None
    
    @property
    def get_spec_objv_exprs_dict(self):
        if self._objv_cmdl_dict is not None:
            return self._objv_cmdl_dict
        if self._SPEC_DICTIONARY_OBJECTIVES in self.spec_dict.keys():
            self._spec_objv_exprs = list(self.spec_dict[self._SPEC_DICTIONARY_OBJECTIVES].values())
            self._objv_spec_dict = self.spec_dict[self._SPEC_DICTIONARY_OBJECTIVES]
            self._objv_dict = self._objv_spec_dict
            return self._objv_dict 
        else:
            return None
    
    # witnesses (assignments to knobs and inputs) that are inspected in mode "certify"
    @property
    def get_spec_witn_dict(self):
        #print('self.spec_dict.keys()', self.spec_dict.keys(), self._SPEC_DICTIONARY_WITNESSES)
        if self._SPEC_DICTIONARY_WITNESSES in self.spec_dict.keys():
            self._witn_dict = self.spec_dict[self._SPEC_DICTIONARY_WITNESSES]
        else:
            self._witn_dict = None  
        return self._witn_dict
    
    # configurations that are inspected in mode "verify"
    @property
    def get_spec_config_dict(self):
        #print('self.spec_dict.keys()', self.spec_dict.keys(), self._SPEC_DICTIONARY_WITNESSES)
        if self._SPEC_DICTIONARY_CONFIGS in self.spec_dict.keys():
            self._config_dict = self.spec_dict[self._SPEC_DICTIONARY_CONFIGS]
        else:
            self._config_dict = None  
        return self._config_dict
    
    def get_spec_component_exprs(self, alph_cmdl, beta_cmdl, delta_abs_cmdl, delta_rel_cmdl, asrt_names_cmdl, asrt_exprs_cmdl,
             quer_names_cmdl, quer_exprs_cmdl, objv_names_cmdl, objv_exprs_cmdl, resp_names, cmdl_cond_sep):
        assert self.spec is not None
        # alpha
        if self._alpha_cmdl_expr is None and alph_cmdl is not None:
            self._alpha_cmdl_expr = alph_cmdl
            self._alpha_global_expr = alph_cmdl
        alph_expr = self.get_spec_alpha_global_expr; #print('alph_expr', alph_expr)
        
        # beta
        if self._beta_cmdl_expr is None and beta_cmdl is not None:
            self._beta_cmdl_expr = beta_cmdl
            self._beta_global_expr = beta_cmdl
        beta_expr = self.get_spec_beta_global_expr; #print('beta_expr', beta_expr)

        # theta radii
        theta_radii_dict = self.get_spec_theta_radii_dict
        
        # delta for theta applied for lemma generation 
        delta_dict = self.get_spec_delta_dict(delta_abs_cmdl, delta_rel_cmdl)
        
        # assertions
        asrt_expr_dict, asrt_names, asrt_exprs = self.get_cmdl_assertions(asrt_names_cmdl, asrt_exprs_cmdl, cmdl_cond_sep)
        asrt_expr_dict = self.get_spec_asrt_exprs_dict; #print('asrt_expr_dict', asrt_expr_dict)
        asrt_names = list(asrt_expr_dict.keys()) if asrt_expr_dict is not None else None
        asrt_exprs = list(asrt_expr_dict.values()) if asrt_expr_dict is not None else None
        
        # queries
        quer_expr_dict, query_names, query_exprs = self.get_cmdl_queries(quer_names_cmdl, quer_exprs_cmdl, cmdl_cond_sep)
        #print('quer_expr_dict', quer_expr_dict); print('query_names, query_exprs', query_names, query_exprs)
        quer_expr_dict = self.get_spec_quer_exprs_dict; #print('quer_expr_dict 2', quer_expr_dict)
        quer_names = list(quer_expr_dict.keys()) if quer_expr_dict is not None else None
        quer_exprs = list(quer_expr_dict.values()) if quer_expr_dict is not None else None

        # optimization objectives
        objv_expr_dict, objv_names, objv_exprs = self.get_cmdl_objectives(objv_names_cmdl, objv_exprs_cmdl, resp_names, cmdl_cond_sep)
        objv_expr_dict = self.get_spec_objv_exprs_dict; #print('objv_expr_dict', objv_expr_dict)
        objv_names = list(objv_expr_dict.keys()) if objv_expr_dict is not None else None
        objv_exprs = list(objv_expr_dict.values()) if objv_expr_dict is not None else None
        
        # witnesses (value assignements to both knobs and inputs)
        witn_dict = self.get_spec_witn_dict
        
        # knob configurations (value assignements to knobs only)
        config_dict = self.get_spec_config_dict
        
        self._spec_logger.info('Computed spec global constraint expressions:')
        self._spec_logger.info('Global alpha : ' + str(alph_expr))
        self._spec_logger.info('Global beta  : ' + str(beta_expr))
        self._spec_logger.info('Radii  theta : ' + str(theta_radii_dict))
        self._spec_logger.info('Delta  const : ' + str(delta_dict))
        if asrt_expr_dict is not None:
            for n, e in asrt_expr_dict.items():
                self._spec_logger.info('Assertion ' + str(n) + ': ' + str(e))
        if quer_expr_dict is not None:
            for n, e in quer_expr_dict.items():
                self._spec_logger.info('Query ' + str(n) + ': ' + str(e))
        if witn_dict is not None:
            if quer_expr_dict is not None:
                for n, w in quer_expr_dict.items():
                    if n in witn_dict.keys():
                        self._spec_logger.info('Witness to query ' + str(n) + '\n' + str(witn_dict[n]))
        if config_dict is not None:
            if asrt_expr_dict is not None:
                for n, w in asrt_expr_dict.items():
                    if n in config_dict.keys():
                        self._spec_logger.info('Configuration for assertion ' + str(n) + ':\n' + str(config_dict[n]))
        if objv_expr_dict is not None:
            for n, e in objv_expr_dict.items():
                self._spec_logger.info('Objective ' + str(n) + ': ' + str(e))
        
        system = self.get_spec_system
        assert system == self._SPEC_DICTIONARY_SYSTEM
        if self._SPEC_DICTIONARY_SYSTEM is not None:
            self._spec_logger.info('Original system : ' + str(system))
        
        self.sanity_check_spec()
        return (alph_expr, beta_expr, theta_radii_dict, delta_dict, asrt_names, asrt_exprs, 
            quer_names, quer_exprs, config_dict, witn_dict, objv_names, objv_exprs, system)
    
        
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
                if self._SPEC_KNOBS_GRID in var_spec.keys():
                    raise Exception('ETA constraint can only be defined for knobs')
                continue
            if self._SPEC_KNOBS_GRID not in var_spec.keys():
                continue
            eta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = var_spec[self._SPEC_KNOBS_GRID]
        
        #print('eta_dict', eta_dict)
        self._eta_dict = eta_dict
        self._spec_logger.info('Knob grids (eta): ' + str(self._eta_dict))
        return eta_dict
    
    # Compute dictionary that maps knobs to theta radii specified in spec file
    @property
    def get_spec_theta_radii_dict(self):
        if self._theta_dict is not None:
            return self._theta_dict
        
        assert self.radius_absolute is None or self.radius_relative is None
        theta_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if var_spec[self._SPEC_VARIABLE_TYPE] != self._SPEC_KNOB_TAG:
                continue
            
            # first try to get radii values from command line, if they were specified in command line
            if self.radius_relative is not None and self._SPEC_KNOBS_RELATIVE_RADIUS in var_spec.keys():
                #print('self.radius_relative', self.radius_relative)
                theta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {self._SPEC_KNOBS_ABSOLUTE_RADIUS: None, 
                    self._SPEC_KNOBS_RELATIVE_RADIUS: self.radius_relative}
                continue
            if self.radius_absolute is not None and self._SPEC_KNOBS_ABSOLUTE_RADIUS in var_spec.keys():
                theta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {
                    self._SPEC_KNOBS_ABSOLUTE_RADIUS: var_spec[self.radius_absolute], 
                    self._SPEC_KNOBS_RELATIVE_RADIUS: None}
            
            # extract radii values from spec file
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
    
    # relative and absolute delta constant used for lemma generation in optimization procedures
    def get_spec_delta_dict(self, delta_abs, delta_rel):
        return {'delta_abs': delta_abs, 'delta_rel': delta_rel}
        
    # Compute dictionary that maps variables to their domains (lower and upper bounds and grids).
    # Variable domains are used by SMLP C++ backand to define variable types and ranges (bounds).
    # Currently we only use variable types (real/float, int, set) to declare to solvers, and ranges
    # of variables are used in alpha and eta constraints (currenlty not in beta constraints). 
    # Domains are usually defined for inputs and knobs, but upper or lower bounds may in general can
    # be defined on model outputs (responses) to be served as constraints during model exploration,
    # but since they are currently used in beta constraints, we impose a sanity check that to ensure 
    # that the bounds are defined for inputs and knobe only. Types are declared for outputts as well.
    @property
    def get_spec_domain_dict(self):
        if self._domain_dict is not None:
            return self._domain_dict
        
        self._domain_dict = {}
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            # TODO !!! ranges on outputs should become part of beta???
            if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                assert var_spec[self._SPEC_VARIABLE_TYPE] in [self._SPEC_INPUT_TAG, self._SPEC_KNOB_TAG]
            #if var_spec[self._SPEC_VARIABLE_TYPE] not in [self._SPEC_INPUT_TAG, self._SPEC_KNOB_TAG]:
            #    continue
            var_bounds = var_spec[self._SPEC_INPUTS_BOUNDS] if self._SPEC_INPUTS_BOUNDS in var_spec.keys() else None
            var_range = var_spec[self._SPEC_VARIABLE_RANGE]
            if not var_range in [self._SPEC_RANGE_INTEGER, self._SPEC_RANGE_REAL]:
                raise Exception('Unsupported variable range (type) ' + var_spec[self._SPEC_VARIABLE_RANGE])
            self._domain_dict[var_spec['label']] = {'range': var_range, 'interval': var_bounds}
            
        #print('self._domain_dict', self._domain_dict)
        self._spec_logger.info('Variable domains (alpha): ' + str(self._domain_dict))
        return self._domain_dict
    
    # Create dictionary that maps inputs and knobs to respective domains specified as a closed interval.
    # Other constraints on domains of inputs can be specified using command line option -alpha.
    @property
    def get_spec_alpha_bounds_dict(self):
        if self._alpha_ranges_dict is not None:
            return self._alpha_ranges_dict
        
        alpha_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                assert var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_INPUT_TAG or \
                    var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_KNOB_TAG
                if  var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_KNOB_TAG:
                    continue 
                mn = var_spec[self._SPEC_INPUTS_BOUNDS][0]
                mx = var_spec[self._SPEC_INPUTS_BOUNDS][1]
                if mn is not None and mx is not None:
                    if mx < mn:
                        range_str = '['+ str(mn) + ',' + str(mx) + ']'
                        raise Exception('The range ' + str(range_str) + ' of variable ' + 
                            str(var_spec[self._SPEC_VARIABLE_LABEL]) + ' is not specified correcly')
                alpha_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {
                    'min': var_spec[self._SPEC_INPUTS_BOUNDS][0], 
                    'max':var_spec[self._SPEC_INPUTS_BOUNDS][1]}
        #print('alpha_dict', alpha_dict)
        
        self._alpha_ranges_dict = alpha_dict
        self._spec_logger.info('Input bounds (alpha): ' + str(self._alpha_ranges_dict))
        return alpha_dict
    
    # Create dictionary that maps knobs to respective ranges specified as a closed interval.
    @property
    def get_spec_eta_bounds_dict(self):
        if self._eta_ranges_dict is not None:
            return self._eta_ranges_dict
        
        eta_dict = {}
        #print('self.spec', self.spec)
        for var_spec in self.spec:
            #print('var_spec', var_spec)
            if self._SPEC_INPUTS_BOUNDS in var_spec.keys():
                assert var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_INPUT_TAG or \
                    var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_KNOB_TAG
                if  var_spec[self._SPEC_VARIABLE_TYPE] == self._SPEC_INPUT_TAG:
                    continue
                mn = var_spec[self._SPEC_INPUTS_BOUNDS][0]
                mx = var_spec[self._SPEC_INPUTS_BOUNDS][1]
                if mn is not None and mx is not None:
                    if mx < mn:
                        range_str = '['+ str(mn) + ',' + str(mx) + ']'
                        raise Exception('The range ' + str(range_str) + ' of variable ' + 
                            str(var_spec[self._SPEC_VARIABLE_LABEL]) + ' is not specified correcly')
                eta_dict[var_spec[self._SPEC_VARIABLE_LABEL]] = {
                    'min': var_spec[self._SPEC_INPUTS_BOUNDS][0], 
                    'max':var_spec[self._SPEC_INPUTS_BOUNDS][1]}
        #print('eta_dict', eta_dict)
        
        self._eta_ranges_dict = eta_dict
        self._spec_logger.info('Knob bounds (eta): ' + str(self._eta_ranges_dict))
        return self._eta_ranges_dict
    
    # access definition of global eta constraint
    @property
    def get_spec_global_eta_expr(self):
        if self._SPEC_DICTIONARY_ETA in self.spec_dict.keys():
            self._eta_global_expr = self.spec_dict[self._SPEC_DICTIONARY_ETA]
            return self._eta_global_expr
        else:
            return None
    
    # Compute variables in model exploration constraints -- constraints on model interface
    # (inputs that can be knobs or free inputs, and outputs), assertions, queries, optimization
    # objectives. Some of these constraints are specified through a spec file, some through
    # command line. This function is used to make sure that variables in these constraints are
    # used during model training (and not dropped during data processing or feature selection),
    # therefore the model exploration constraints are well defined on the model interface.
    def get_spec_constraint_vars(self):
        constraints_vars = []
        #print('spec self', self._alpha_global_expr, self._beta_global_expr)
        if self._alpha_global_expr is not None:
            alph_vars = get_expression_variables(self.get_spec_alpha_global_expr); #print('alph_vars', alph_vars)
            constraints_vars = constraints_vars + alph_vars
        if self._beta_global_expr is not None:
            beta_vars = get_expression_variables(self._beta_global_expr); #print('beta_vars', beta_vars)
            constraints_vars = constraints_vars + beta_vars
        if self._eta_global_expr is not None:
            eta_vars = get_expression_variables(self._eta_global_expr); #print('eta_vars', beta_vars)
            constraints_vars = constraints_vars + eta_vars
        if self._objv_dict is not None:
            for objv_expr in self._objv_dict.values():
                objv_vars = get_expression_variables(objv_expr); #print('objv_expr', objv_expr)
                constraints_vars = constraints_vars + objv_vars
        if self._asrt_dict is not None:
            for asrt_expr in self._asrt_dict.values():
                asrt_vars = get_expression_variables(asrt_expr); #print('asrt_vars', asrt_vars)
                constraints_vars = constraints_vars + asrt_vars
        if self._quer_dict is not None:
            for quer_expr in self._quer_dict.values():
                quer_vars = get_expression_variables(quer_expr); #print('quer_expr', quer_expr)
                constraints_vars = constraints_vars + quer_vars

        if self._witn_dict is not None:
            for witn in self._witn_dict.values():
                #print('witn.keys()', witn.keys())
                constraints_vars = constraints_vars + list(witn.keys())
                
        if self._config_dict is not None:
            for config in self._config_dict.values():
                #print('config.keys()', config.keys())
                constraints_vars = constraints_vars + list(config.keys())
        
        return list_unique_unordered(constraints_vars)
    
    def get_spec_stability_ragion_bounds_dict(self, knob_config):
        input_ranges = {}
        for k, v in self.get_spec_alpha_bounds_dict.items():
            input_ranges[k] = [v['min'], v['max']]
        knobs_ranges = {}
        for k, v in self.get_spec_theta_radii_dict.items():
            knov_val = knob_config[k]['value_in_config']; #print('knov_val', knov_val)
            if v['rad-abs'] is not None:
                rad = abs(knov_val); #print('rad-abs', rad)
            elif v['rad-rel'] is not None:
                rad = v['rad-rel'] * abs(knov_val); #print('rad-rel', rad)
            else:
                raise Exception('At least on of the relative or absolute radii must mot be None')
            knobs_ranges[k] = [knov_val - rad, knov_val + rad] #{'min':knov_val - rad, 'max':knov_val + rad}
        return input_ranges | knobs_ranges