# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from fractions import Fraction
import json

from icecream import ic
import smlp
from smlp_py.smlp_terms import ModelTerms, SmlpTerms
from smlp_py.smlp_utils import np_JSONEncoder #, str_to_bool
#from smlp_py.smlp_utils import np_JSONEncoder
from smlp_py.ext import plot

ic.configureOutput(prefix=f'Debug | ', includeContext=True)

plot_instance = plot.plot_exp()

class SmlpQuery:
    def __init__(self):
        self._smlpTermsInst = SmlpTerms()
        self._modelTermsInst = None #ModelTerms()

        self._DEF_QUERY_NAMES = None
        self._DEF_QUERY_EXPRS = None
        self._DEF_LEMMA_PRECISION = 0
        
        # keys in the dictionary capturing the results of function self.query_condition()
        self._query_stable = 'stable'
        self._query_feasible = 'feasible'
        self._query_status = 'status'
        self._query_result = 'result'
        
        self.query_params_dict = {
            'query_names': {'abbr':'quer_names', 'default':str(self._DEF_QUERY_EXPRS), 'type':str,
                'help':'Names of optimization objectives [default {}]'.format(str(self._DEF_QUERY_NAMES))}, 
            'query_expressions':{'abbr':'quer_exprs', 'default':self._DEF_QUERY_EXPRS, 'type':str,
                'help':'Semicolon seperated list of expressions (functions) to be applied to the responses '
                    'to convert them into optimization objectives ' +
                    '[default: {}]'.format(str(self._DEF_QUERY_EXPRS))},
            'lemma_precision':{'abbr':'lemma_prec', 'default':self._DEF_LEMMA_PRECISION, 'type':int,
                'help':'Number of decimals after zero to use when approximating lemmas in model exploration modes. ' +
                    'The default value 0 means that lemmas should not be approximated (full precision should be used ' +
                    '[default: {}]'.format(str(self._DEF_LEMMA_PRECISION))}
        }
        
        # profiling SMLP run, the steps taken by the algorithm and solver runtimes
        self._query_tracer = None
        self._trace_runtime = None
        self._trace_precision = None
        self._trace_anonymize = None

    def set_logger(self, logger):
        self._query_logger = logger 
        self._smlpTermsInst.set_logger(logger)
        self._modelTermsInst.set_logger(logger)
    
    def set_tracer(self, tracer, trace_runtime, trace_prec, trace_anonym):
        self._query_tracer = tracer
        self._trace_runtime = trace_runtime
        self._trace_precision = trace_prec
        self._trace_anonymize = trace_anonym
        
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        #self._smlpTermsInst.set_report_file_prefix(report_file_prefix)
        self._modelTermsInst.set_report_file_prefix(report_file_prefix)
        
    # model_file_prefix is a string used as prefix in all saved model files of SMLP
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
        #self._smlpTermsInst.set_model_file_prefix(model_file_prefix)
        self._modelTermsInst.set_model_file_prefix(model_file_prefix)
    
    # set self._modelTermsInst ModelTerms()
    def set_model_terms_inst(self, model_terms_inst):
        self._modelTermsInst = model_terms_inst
    
    def set_trace_file(self, trace_file):
        self._trace_file = trace_file
    
    def set_lemma_precision(self, lemma_precision):
        self._lemma_precision = lemma_precision
    
    @property
    def query_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_query_results.json'
        
    @property
    def certify_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_certify_results.json'
    
    @property
    def verify_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_verify_results.json'
    
    @property
    def synthesis_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_synthesize_results.json'

    def find_candidate(self, solver):
        #res = solver.check()
        res = self._modelTermsInst.smlp_solver_check(solver, 'ca', self._lemma_precision)
        if self._modelTermsInst.solver_status_unknown(res): # isinstance(res, smlp.unknown):
            return None
        else:
            return res
        
    def update_consistecy_results(self, mode_status_dict, interface_consistent, model_consistent,
            mode_status, mode_results_file):
        # update mode_status_dict based on interface and model consistency results 
        # computed by get_model_exploration_base_components()
        mode_status_dict['interface_consistent'] = str(interface_consistent).lower()
        mode_status_dict['model_consistent'] = str(model_consistent).lower()
        if not interface_consistent or not model_consistent:
            mode_status_dict['smlp_execution'] = 'aborted'
            mode_status_dict[mode_status] = 'ERROR'
        with open(mode_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        return mode_status_dict
    
    def get_model_exploration_base_components(self, mode_status_dict, results_file,
            syst_expr_dict:dict, algo, model, model_features_dict:dict, feat_names:list, resp_names:list, 
            alph_expr:str, beta_expr:str, eta_expr:str, data_scaler, scale_feat, scale_resp, 
            float_approx=True, float_precision=64, data_bounds_json_path=None):
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self._modelTermsInst.create_model_exploration_base_components(
                syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
                alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, 
                float_approx, float_precision, data_bounds_json_path)
        
        mode_status_dict['interface_consistent'] = str(interface_consistent).lower()
        mode_status_dict['model_consistent'] = str(model_consistent).lower()
        if not interface_consistent or not model_consistent:
            mode_status_dict['smlp_execution'] = 'aborted'
            with open(results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            self._query_logger.info('Input and knob interface constraints are inconsistent; aborting...')
            return domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent
        with open(results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        return domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent
        

    # Enhancement !!! instead of using witn_form, bind inputs and knobs to their values by directly 
    # applying solver.add(var == val) or directly substituting these values in other formulas --
    # just for small potential speedup.
    def check_concrete_witness_consistency(self, domain:smlp.domain, model_full_term_dict:dict, 
            alpha:smlp.form2, eta:smlp.form2, query:smlp.form2, witn_form:smlp.form2, solver_logic:str):
        solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        solver.add(alpha); #print('alpha', alpha)
        solver.add(eta); #print('eta', eta)
        solver.add(witn_form); #print('witn_form', witn_form); print('query', query)
        if query is not None:
            solver.add(query)
        res = self._modelTermsInst.smlp_solver_check(solver, 'witness_consistency')
        #res = solver.check(); #print('res', res)
        return res

    
    
    # TODO: what about eta-interval and eta_global constraints, as well as eta-grid constraints for
    # integer control (knob) variables? Looks like they should be used as constrints -- look for a cex
    # to a candidate only under these (and other). Grid constraints for continuous variable should not 
    # be used
    #   ! ( theta x y -> alpha y -> beta y /\ obj y >= T ) =
    #   ! ( ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
    #   theta x y /\ alpha y /\ ! ( beta y /\ obj y >= T) 
    def find_candidate_counter_example(self, universal, domain:smlp.domain, cand:dict, query:smlp.form2, 
            model_full_term_dict:dict, alpha:smlp.form2, theta_radii_dict:dict, solver_logic:str): #, beta:smlp.form2
        solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, False, solver_logic)
        theta = self._modelTermsInst.compute_stability_formula_theta(cand, None, theta_radii_dict, universal) 
        solver.add(theta); #print('adding theta', theta)
        solver.add(alpha); #print('adding alpha', alpha)
        solver.add(self._smlpTermsInst.smlp_not(query)); #print('adding negated quert', query)
        return self._modelTermsInst.smlp_solver_check(solver, 'ce', self._lemma_precision)
        #return solver.check()
    
    # Enhancement !!!: at least add here the delta condition
    def generalize_counter_example(self, coex):
        return coex
    
    # This function is called from validate_witness() on already built model terms and formulas for constraints.
    # It check stability of witness given as witn_dict, which in case universal == True is a value assignements to knobs,
    # and in case universal = False is a value assignement to both knobs as well as inputs. 
    def validate_witness_smt(self, universal:bool, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, witn_dict:dict,
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:float, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int):
        if universal:
            self._query_logger.info('Verifying assertion {} <-> {}'.format(str(quer_name), str(quer_expr)))
        else:
            self._query_logger.info('Certifying stability of witness for query ' + str(quer_name) + ':\n   ' + str(witn_dict))
        candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        
        cond_feasible = None
        # add the remaining user constraints and the query
        candidate_solver.add(eta); #print('adding eta', eta)
        candidate_solver.add(alpha); #print('adding alpha', alpha)
        #candidate_solver.add(beta)
        candidate_solver.add(quer); #print('adding quer', quer)
        #print('adding witn_dict', witn_dict)
        for var,val in witn_dict.items():
            #candidate_solver.add(smlp.Var(var) == smlp.Cnst(val))
            candidate_solver.add(self._smlpTermsInst.smlp_eq(smlp.Var(var), smlp.Cnst(val)))
        
        candidate_check_res = self._modelTermsInst.smlp_solver_check(candidate_solver, 'ca')
        if self._modelTermsInst.solver_status_sat(candidate_check_res): #isinstance(candidate_check_res, smlp.sat):
            cond_feasible = True
            if universal:
                self._query_logger.info('The configuration is consistent with assertion ' + str(quer_name))
            else:
                self._query_logger.info('Witness to query ' + str(quer_name) + ' is a valid witness; checking its stability')
        elif self._modelTermsInst.solver_status_unsat(candidate_check_res): #isinstance(candidate_check_res, smlp.unsat):
            cond_feasible = False
            if universal:
                # Assertion cannot be satisfied (is constant False) given the knob configuration and the constraints.
                # Its negation will be true for any legal inputs, which means that the assertion fails everywhere in the
                # legal input space. This is useful info because in such a case maybe the problem instance (assertion or 
                # constraints) were not specified correctly. Also, if such a case is found as part of an optimization 
                # procedure (where assertion is of the form objective >= threshold), the get a "general" counter-example
                # and the entire legal input space can be ruled out from the consequtive search.
                # In this implementation, we do not exist here and continue in order to generate a counter-example.
                # We could also use a sat assignement found in feasibility check for that purpose.
                # When used in optimization context, one could return here the generalized counter-example that rules
                # out the entire legal spece (while dealing with that assertion / that objective and threshold).
                self._query_logger.info('The configuration is inconsistent with assertion ' + str(quer_name))
                #self._query_logger.info('Completed with result: VACUOUS PASS')
                #return {'assertion_status':'VACUOUS PASS', 'asrt': None, 'model':None} 
            else:
                # concrete witness does not satisfy the querym this witness cannot be stable witness fro that query,
                # no further checks need to be performed (thus esiting the function under this condition, here)
                self._query_logger.info('Witness to query ' + str(quer_name) + ' is not a valid witness (even without stability requirements)')
                return 'not a witness'
        else:
            raise Exception('Unexpected counter-example status ' + str(ce) + ' in function validate_witness_smt')

        # checking stability of a valid witness to the query
        witn_term_dict = self._smlpTermsInst.witness_const_to_term(witn_dict)
        ce = self.find_candidate_counter_example(universal, domain, witn_term_dict, quer, model_full_term_dict, alpha, 
            theta_radii_dict, solver_logic)
        if self._modelTermsInst.solver_status_sat(ce): #isinstance(ce, smlp.sat):
            if universal:
                self._query_logger.info('Completed with result: FAIL')
                #self._query_logger.info('Assertion ' +  str(quer_name) + ' fails (for stability radii ' + str(theta_radii_dict))
                #status = 'FAIL' if cond_feasible else 'FAIL VACUOUSLY'
                ce_model = self._modelTermsInst.get_solver_model(ce)
                return {'assertion_status':'FAIL', 'asrt': False, 'assertion_feasible': cond_feasible, 
                        'counter_example':self._smlpTermsInst.witness_term_to_const(ce_model, approximate=sat_approx, precision=sat_precision)}
            else:
                self._query_logger.info('Witness to query ' + str(quer_name) + ' is not stable for radii ' + str(theta_radii_dict))
                return 'witness, not stable'
        elif self._modelTermsInst.solver_status_unsat(ce): #isinstance(ce, smlp.unsat):
            if universal:
                self._query_logger.info('Completed with result: PASS')
                #self._query_logger.info('Assertion ' +  str(quer_name) + ' passes (for stability radii ' + str(theta_radii_dict))
                return {'assertion_status':'PASS', 'asrt':None, 'assertion_feasible': cond_feasible,  'counter_example':None}
            else:
                assert cond_feasible
                self._query_logger.info('Witness to query ' + str(quer_name) + ' is stable for radii ' + str(theta_radii_dict))
                return 'stable witness'
        else:
            raise Exception('Unexpected counter-example ststus ' + str(ce) + ' in function validate_witness_smt')
    
    
    def validate_witness(self, universal:bool, syst_expr_dict:dict, algo:str, model:dict, #universal:bool, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            quer_names:list[str], quer_exprs:list[str], witn_dict:dict, delta:dict,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
         
        if universal:
            CONSISTENCY = 'configuration_consistent'
            query_status_dict = {
                CONSISTENCY: 'unknown',
                'assertion_status': 'unknown',
                'counter_example': 'unknown'
            }
            results_file = self.verify_results_file
        else:
            CONSISTENCY = 'witness_consistent'
            query_status_dict = {
                CONSISTENCY: 'unknown',
                'witness_feasible': 'unknown',
                'witness_stable': 'unknown'
            }
            results_file = self.certify_results_file

        # TODO: implement this check in sanity_check_certification_spec
        if quer_names is None or quer_exprs is None:
            query_status_dict['amlp_execution'] = 'aborted'
            with open(results_file, 'w') as f:
                json.dump(query_status_dict, f, indent='\t', cls=np_JSONEncoder)
            self._query_logger.info('Queries are not specified in mode "certify"; aborting...')
            return
        
        # initialize the fields in the ststus dictionary mode_status_dict as unknown/running
        mode_status_dict = {}
        for quer_name in quer_names:
            # .copy() is required in next line to ensure updating different copies will be independent
            mode_status_dict[quer_name] = query_status_dict.copy() 
        mode_status_dict['smlp_execution'] = 'running'
        mode_status_dict['interface_consistent'] = 'unknown'
        
        # compute model exploration base componensts (models, constraints, etc.)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self.get_model_exploration_base_components(mode_status_dict, results_file, 
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            #delta, None, #None, None, None, None, quer_names, quer_exprs, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, #scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        
        if not interface_consistent or not model_consistent:
            for quer_name in quer_names:
                if universal:
                    mode_status_dict[quer_name]['assertion_status'] = 'ERROR'
                else:
                    mode_status_dict[quer_name]['witness_status'] = 'ERROR'
                with open(results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        
        knobs = list(theta_radii_dict.keys()); #print('knobs', knobs)
        if witn_dict is None and len(knobs) == 0:
            witn_dict = dict(zip(quer_names, [{}]*len(quer_names)))
        #print('witn_dict 2', witn_dict)
        witn_count = len(witn_dict.keys()); #print('witn_count', witn_count)
        #print('quer_names', quer_names, 'quer_exprs', quer_exprs)
        if witn_count != len(quer_exprs):
            raise Exception('The number of queries does not match the number of witnesses')
        # TODO !!!! do we need this way of defining queries as smlp.true?
        
        # drop fron witn_dict assignments to variables that do not occur in the model
        witn_dict_filtered = {}
        witn_form_dict = {}
        quer_expr_dict = dict(zip(quer_names, quer_exprs))
        for q, w in witn_dict.items():
            assert isinstance(w, dict)
            witn_dict_filtered[q] = {}
            
            for var,val in w.items():
                if universal: # univeral witness, verification, synthesis and optimized synthesis modes
                    if var in knobs:
                        assert var in feat_names
                        witn_dict_filtered[q][var] = val
                else: # certification mode
                    if var in feat_names:
                        witn_dict_filtered[q][var] = val
            witn_form_dict[q] = self._smlpTermsInst.witness_to_formula(witn_dict_filtered[q])
                                              
            # assert every input and knob is assigned in w when universal is false, and that every knob
            # is assigned in w when universal is true. This condition will be redundant (and wrong) when
            # SMLP will support range assignements to knobs and inputs (including don't care knobs/inputs).
            w_keys = w.keys()
            
            if universal:
                # sanity check is already domne in sanity_check_verification_spec
                pass
                '''
                unassigned = []
                for feat in knobs:
                    if feat not in w_keys:
                        unassigned.append(feat)
                if len(unassigned) > 0:
                    self._query_logger.info('The knob configuration has variable(s) ' + str(unassigned) + ' unassigned a value')
                    query_status_dict['assertion_status'] = 'unassigned_knobs'
                with open(results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)        
                return
                #raise Exception ('The knob configuration has variable ' + str(feat) + ' unassigned a value')
                '''
            else:
                for feat in feat_names:
                    if feat not in w_keys:
                        raise Exception ('Wirness to query ' + str(q) + ' + has variable ' + str(feat) + ' unassigned a value')

        if quer_names is None and quer_exprs is None:
            #print('updating quer_exprs')
            quer_names = list(witn_dict_filtered.keys())
            quer_exprs = ['True'] * witn_count
        
        if beta_expr is not None:
            quer_exprs = [quer_exprs[i] + ' and ' + beta_expr for i in range(witn_count)]; #print('quer_exprs', quer_exprs)
        quer_forms_dict = dict([(quer_name, self._smlpTermsInst.ast_expr_to_term(quer_expr)) \
            for quer_name, quer_expr in zip(quer_names, quer_exprs)])
        # sanity check -- queries must be formulas (not terms)
        for i, form in enumerate(quer_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Quesry ' + str(quer_exprs[i]) + ' must be a formula (not a term)')

        # instance consistency and feasibility checks (are the assumptions contradictory?)
        if vacuity:
            #print('quer_forms_dict', quer_forms_dict.keys(), 'quer_names', quer_names)
            assert list(quer_forms_dict.keys()) == quer_names
            #print('witn_form_dict', witn_form_dict)
            quer_res_dict = {}
            for quer_name, quer_form in quer_forms_dict.items():
                #print('quer_name', quer_name, 'quer_form', quer_form)
                witn_form = witn_form_dict[quer_name]; #print('witn_form', witn_form)
                # checking concrete witneses consistency alpha and beta (w/o looking at respective queries)
                if universal:
                    #self._query_logger.info('Verifying assertion {} <-> {}'.format(str(quer_name), str(quer_expr_dict[quer_name])))
                    self._query_logger.info('Verifying consistency of configuration for assertion ' + str(quer_name) + ':\n   ' + str(witn_form))
                else:
                    self._query_logger.info('Certifying consistency of witness for query ' + str(quer_name) + ':\n   ' + str(witn_form))
                witn_status = self.check_concrete_witness_consistency(domain, model_full_term_dict, 
                    alpha, eta, None, witn_form, solver_logic)
                if self._modelTermsInst.solver_status_sat(witn_status): #isinstance(witn_status, smlp.sat):
                    if universal:
                        self._query_logger.info('Input, knob and configuration constraints are consistent')
                    else:
                        self._query_logger.info('Input, knob and concrete witness constraints are consistent')
                    mode_status_dict[quer_name][CONSISTENCY] = 'true'
                elif self._modelTermsInst.solver_status_unsat(witn_status): #isinstance(witn_status, smlp.unsat):
                    if universal:
                        self._query_logger.info('Input, knob and configuration constraints are inconsistent')
                    else:
                        self._query_logger.info('Input, knob and concrete witness constraints are inconsistent')
                    mode_status_dict[quer_name][CONSISTENCY] = 'false'
                    
                else:
                    raise Exception('Input, knob and concrete witness cosnsistency check failed to complete')
                #print('after witness_consistence', mode_status_dict)
                with open(results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        else:
            mode_status_dict['interface_consistent'] = 'skipped'
            for quer_name, quer_form in quer_forms_dict.items():
                mode_status_dict[quer_name][CONSISTENCY] = 'skipped'
            with open(results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)    
        
        for quer_name, quer_form in quer_forms_dict.items():
            #print('quer_name', quer_name, 'quer_form', quer_form)
            witn_form = witn_form_dict[quer_name]; #print('witn_form', witn_form) 
            if mode_status_dict[quer_name][CONSISTENCY] != 'false':    
                witn_i_dict = witn_dict[quer_name]
                witness_status_str = self.validate_witness_smt(universal, model_full_term_dict, quer_name, quer_expr_dict[quer_name], quer_form,
                    witn_i_dict, domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
                #print('witness_status_str', witness_status_str)
                if universal:
                    for k,v in witness_status_str.items():
                        #print('k', k, 'v', v)
                        if k in ['assertion_feasible', 'assertion_status', 'counter_example']:
                            mode_status_dict[quer_name][k] = v
                else:
                    quer_res_dict[quer_name] = witness_status_str
                    if witness_status_str == 'not a witness':
                        #print(quer_name, 'case not a witness')
                        mode_status_dict[quer_name]['witness_feasible'] = 'false'
                        mode_status_dict[quer_name]['witness_stable'] = 'false'
                        mode_status_dict[quer_name]['witness_status'] = 'FAIL'
                    elif witness_status_str == 'witness, not stable':
                        #print(quer_name, 'case witness, not stable')
                        mode_status_dict[quer_name]['witness_feasible'] = 'true'
                        mode_status_dict[quer_name]['witness_stable'] = 'false'
                        mode_status_dict[quer_name]['witness_status'] = 'FAIL'
                    elif witness_status_str == 'stable witness':
                        #print(quer_name, 'case stable witness')
                        mode_status_dict[quer_name]['witness_feasible'] = 'true'
                        mode_status_dict[quer_name]['witness_stable'] = 'true'
                        mode_status_dict[quer_name]['witness_status'] = 'PASS'
                    else:
                        raise Exception('Unexpected status ' + str() + ' in function validate_witness')
            else:
                if universal:
                    self._query_logger.info('Skipping verification of assertion {} <-> {}'.format(str(quer_name), str(quer_form)))
                    self._query_logger.info('Reporting result: ERROR')
                    mode_status_dict[quer_name]['assertion_feasible'] = 'false'
                    mode_status_dict[quer_name]['assertion_status'] = 'ERROR' #VACUOUS PASS
                    mode_status_dict[quer_name]['counter_example'] = None
                else:
                    mode_status_dict[quer_name]['witness_feasible'] = 'false'
                    mode_status_dict[quer_name]['witness_stable'] = 'false'
                    mode_status_dict[quer_name]['witness_status'] = 'ERROR' #FAIL

            with open(results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)                

        mode_status_dict['smlp_execution'] = 'completed'
        with open(results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)

    
    def smlp_verify(self, syst_expr_dict:dict, algo:str, model:dict, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            asrt_names:list[str], asrt_exprs:list[str], witn_dict:dict, delta:dict,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        self.validate_witness(True, syst_expr_dict, algo, model,
            model_features_dict, feat_names, resp_names, 
            asrt_names, asrt_exprs, witn_dict, delta,
            alph_expr, beta_expr, eta_expr, theta_radii_dict, solver_logic, vacuity, 
            data_scaler, scale_feat, scale_resp, float_approx, float_precision, 
            data_bounds_json_path, bounds_factor, T_resp_bounds_csv_path)
        
    def smlp_certify(self, syst_expr_dict:dict, algo:str, model:dict, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            quer_names:list[str], quer_exprs:list[str], witn_dict:dict, delta:dict,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        self.validate_witness(False, syst_expr_dict, algo, model,
            model_features_dict, feat_names, resp_names, 
            quer_names, quer_exprs, witn_dict, delta,
            alph_expr, beta_expr, eta_expr, theta_radii_dict, solver_logic, vacuity, 
            data_scaler, scale_feat, scale_resp, float_approx, float_precision, 
            data_bounds_json_path, bounds_factor, T_resp_bounds_csv_path)
        

    # Enhancement !!!: implement timeout ? UNKNOWN return value
    def query_condition(self, universal, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:dict, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int):
        # feasibility (existence) of at least one candidate
        feasible = None
        if quer_expr is not None:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer_expr)))
        else:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer)))
        #print('query', quer, 'eta', eta, 'delta', delta)
        candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        
        # add the remaining user constraints and the query
        candidate_solver.add(eta)
        candidate_solver.add(alpha)
        #candidate_solver.add(beta)
        candidate_solver.add(quer)
        #print('eta', eta); print('alpha', alpha);  print('quer', quer); 
        #print('solving query', quer)
        self._query_tracer.info('{},{}'.format('synthesis' if universal else 'query', str(quer_name))) #, str(quer_expr) ,{}
        use_approxiamted_fractions = self._lemma_precision != 0
        assert self._lemma_precision >= 0 and isinstance(self._lemma_precision, int)
        approx_ca_models = {} # save rounded ca models to check whether rounded models occure repeaedly
        approx_ce_models = {} # save rounded ce models to check whether rounded models occure repeaedly
        while True:
            # solve Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ query)
            print('searching for a candidate', flush=True)
            
            ca = self.find_candidate(candidate_solver)
            if self._modelTermsInst.solver_status_sat(ca): # isinstance(ca, smlp.sat):
            #if isinstance(ca, smlp.sat):
                print('candidate found -- checking stability', flush=True)
                #print('ca', ca_model)
                ca_model = self._modelTermsInst.get_solver_model(ca) #ca.model
                if use_approxiamted_fractions:
                    ca_model_approx = self._smlpTermsInst.approximate_witness_term(ca_model, self._lemma_precision)
                    ic('ca_model_approx -------------', ca_model_approx)
                    knob_vals = [v for k,v in ca_model_approx.items() if k in theta_radii_dict]; #print('knob_vals', knob_vals)
                    h = hash(str(knob_vals))
                    if h in approx_ca_models:
                        #print('hit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        approx_ca_models[h] = approx_ca_models[h] + 1
                        #self._query_tracer.info('hits,{}'.format(str(sum(list(approx_ca_models.values())))))
                    else:
                        approx_ca_models[h] = 0
                    ic('ca_model_approx', ca_model_approx)
                feasible = True
                ic('Changes here ...')
                witnessvals = self._smlpTermsInst.witness_term_to_const(ca_model, sat_approx, sat_precision)
#                ic(ca_model, sat_approx, sat_precision)
                plot_instance.save_to_csv(witnessvals, data_version='witnesses')
                if use_approxiamted_fractions:
                    ce = self.find_candidate_counter_example(universal, domain, ca_model_approx, quer, model_full_term_dict, alpha, 
                        theta_radii_dict, solver_logic)
                else:
                    ce = self.find_candidate_counter_example(universal, domain, ca_model, quer, model_full_term_dict, alpha, 
                        theta_radii_dict, solver_logic)
                if self._modelTermsInst.solver_status_sat(ce): #isinstance(ce, smlp.sat):
                #if isinstance(ce, smlp.sat):
                    print('candidate not stable -- continue search', flush=True)
                    ce_model = self._modelTermsInst.get_solver_model(ce) #ce.model
                    #ic(ce_model['z'])
                    cem = ce_model.copy(); #print('ce model', cem)
                    # drop Assignements to responses from ce
                    for var in ce_model.keys():
                        if var in model_full_term_dict.keys():
                            del cem[var]
                    if use_approxiamted_fractions:
                        ce_model_approx = self._smlpTermsInst.approximate_witness_term(cem, self._lemma_precision)
                        #print('ce_model_approx ++++++++++', ce_model_approx)
                        knob_vals = [v for k,v in ce_model_approx.items() if k in theta_radii_dict]; #print('knob_vals', knob_vals)
                        h = hash(str(knob_vals))
                        if h in approx_ce_models:
                            #print('hit ??????????????????????????????????????')
                            approx_ce_models[h] = approx_ce_models[h] + 1
                            #self._query_tracer.info('hits,{}'.format(str(sum(list(approx_ce_models.values())))))
                        else:
                            approx_ce_models[h] = 0
                        #print('ce_model_approx', ce_model_approx)
                        lemma = self.generalize_counter_example(ce_model_approx); #print('lemma', lemma)
                    else:
                        lemma = self.generalize_counter_example(cem); #print('lemma', lemma)
                    theta = self._modelTermsInst.compute_stability_formula_theta(lemma, delta, theta_radii_dict, universal)
                    candidate_solver.add(self._smlpTermsInst.smlp_not(theta))
                    continue
                elif self._modelTermsInst.solver_status_unsat(ce): #isinstance(ce, smlp.unsat):
                    #print(self._modelTermsInst.solver_status_unsat(ce))
                    #print(ce)
                    #print('candidate stable -- return candidate')
                    witnessvals = self._smlpTermsInst.witness_term_to_const(ca_model, sat_approx, sat_precision)
                    #ic(witnessvals)
                    plot_instance.save_to_csv(witnessvals, data_version='stable')
                    self._query_logger.info('Query completed with result: STABLE_SAT (satisfiable)')
                    if witn: # export witness (use numbers as values, not terms)
                        ca_model = self._modelTermsInst.get_solver_model(ca) # ca.model
                        witness_vals_dict = self._smlpTermsInst.witness_term_to_const(ca_model, sat_approx, sat_precision)
                        #print('domain witness_vals_dict', witness_vals_dict)
                        # sanity check: the value of query in the sat assignment should be true
                        if quer_expr is not None:
                            quer_ce_val = eval(quer_expr, {},  witness_vals_dict); #print('quer_ce_val', quer_ce_val)
                            assert quer_ce_val
                        return {'query_status':'STABLE_SAT', 'witness':witness_vals_dict, 'feasible':feasible}
                    else:
                        return {'query_status':'STABLE_SAT', 'witness':ca_model, 'feasible':feasible}
            elif self._modelTermsInst.solver_status_unsat(ca): #isinstance(ca, smlp.unsat):
                self._query_logger.info('Query completed with result: UNSAT (unsatisfiable)')
                ic("Changes here ...")
                solver = "unsat"
                lower_bound = None
                plot_instance.witnesses(lower_bound, solver)
                if feasible is None:
                    feasible = False
                #print('candidate does not exist -- query unsuccessful')
                #print('query unsuccessful: witness does not exist (query is unsat)')
                return {'query_status':'UNSAT', 'witness':None, 'feasible':feasible}
            elif self._modelTermsInst.solver_status_unknown(ca): #isinstance(ca, smlp.unknown):
                self._opt_logger.info('Completed with result: {}'.format('UNKNOWN'))
                return {'query_status':'UNKNOWN', 'witness':None, 'feasible':feasible}
                #raise Exception('UNKNOWN return value in candidate search is currently not supported for queries')
            else:
                raise Exception('Unexpected return value ' + str(type(ca)) + ' in candidate search for queries')
        
    
    # iterate over all queries using query_condition()        
    def query_conditions(self, universal:bool, model_full_term_dict:dict, quer_names:str, quer_exprs:str, quer_forms_dict:dict, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict,
            delta:dict, solver_logic, witn:bool, sat_approx:bool, sat_precision:int):
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            quer_res_dict[quer_name] = self.query_condition(universal, model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, domain, eta, alpha, theta_radii_dict, delta, solver_logic, witn, sat_approx, sat_precision)
        return quer_res_dict
    
    # querying conditions on a model to find a stable witness satisfying this condition in entire stability region
    # (defined by stability/theta radii) around that witness (which is a SAT assignment to model interface variables)
    def smlp_query(self, syst_expr_dict:dict, algo:str, model:dict, model_features_dict:dict,
            feat_names:list[str], resp_names:list[str], quer_names:list[str], quer_exprs:list[str], delta:float,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        # *_synthrsis_results.json file fields
        QUERY_FEASIBLE = 'query_feasible'
        QUERY_STABLE = 'query_stable' 
        QUERY_STATUS = 'query_status'
        QUERY_WITNESS = 'query_result' #query_witness
        
        # initialize the fields in the more status dictionary mode_status_dict as unknown/running
        mode_status_dict = {'smlp_execution': 'running', 'interface_consistent': 'unknown'}
        for quer_name in quer_names:
            mode_status_dict[quer_name] = {
                QUERY_FEASIBLE: 'unknown',
                QUERY_STABLE: 'unknown',
                QUERY_STATUS: 'unknown',
                QUERY_WITNESS: None} 
        with open(self.query_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
        if quer_exprs is None:
            self._query_logger.error('Queries were not specified in the "query" mode: aborting...')
            mode_status_dict['smlp_execution'] = 'aborted'
            with open(self.query_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        
        # compute model exploration componenets (domain, terms, formulas)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self.get_model_exploration_base_components(mode_status_dict, self.query_results_file,
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, #delta, None,
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, #scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        
        mode_status_dict['interface_consistent'] = str(interface_consistent).lower()
        mode_status_dict['model_consistent'] == str(model_consistent).lower()
        if not interface_consistent or not model_consistent:
            mode_status_dict['smlp_execution'] = 'aborted'
            for quer_name in quer_names:
                mode_status_dict[quer_name][QUERY_STATUS] = 'ERROR'
            with open(self.query_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        with open(self.query_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)

        # compute and sanity check query smlp expressions -- queries must be formulas (not terms)
        quer_forms_dict = dict([(quer_name, self._smlpTermsInst.ast_expr_to_term(quer_expr)) \
                for quer_name, quer_expr in zip(quer_names, quer_exprs)])
        for i, form in enumerate(quer_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Query ' + str(quer_exprs[i]) + ' must be a formula (not a term)')
        # execute queries
        quer_res_dict = self.query_conditions(
            False, model_full_term_dict, quer_names, quer_exprs, quer_forms_dict, domain, 
            eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
        
        # adjust key names in the reusts
        for quer_name in quer_names:
            mode_status_dict[quer_name][QUERY_FEASIBLE] = str(quer_res_dict[quer_name]['feasible']).lower()
            mode_status_dict[quer_name][QUERY_STABLE] = str(quer_res_dict[quer_name]['query_status'] == 'STABLE_SAT').lower()
            mode_status_dict[quer_name][QUERY_STATUS] = 'PASS' if quer_res_dict[quer_name]['query_status'] == 'STABLE_SAT' else 'FAIL'
            mode_status_dict[quer_name][QUERY_WITNESS] = quer_res_dict[quer_name]['witness']
        # finalize and report results
        mode_status_dict['smlp_execution'] = 'completed'
        #print('final mode_status_dict', mode_status_dict)
        with open(self.query_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
            
    # SMLP synthesis mode: configure knobs with stability to meet beta constraints under alpha, eta, theta and constraints
    def smlp_synthesize(self, syst_expr_dict:dict, algo:str, model:dict, #X:pd.DataFrame, y:pd.DataFrame, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], asrt_names, asrt_exprs, 
            delta:float, alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, #scale_objv:bool, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        # *_synthrsis_results.json file fields
        QUERY_FEASIBLE = 'configuration_feasible'
        QUERY_STABLE = 'configuration_stable'
        QUERY_STATUS = 'synthesis_status'
        QUERY_WITNESS = 'synthesis_result'
        
        # initialize the fields in the more status dictionary mode_status_dict as unknown/running
        mode_status_dict = {
            'smlp_execution': 'running', 
            'interface_consistent': 'unknown',
            'model_consistent': 'unknown',
            QUERY_FEASIBLE: 'unknown',
            QUERY_STABLE: 'unknown',
            QUERY_STATUS: 'unknown',
            QUERY_WITNESS: None}
        with open(self.synthesis_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        
        #print('eta_expr', eta_expr); print('alph_expr', alph_expr); print('beta_expr', beta_expr)
        # compute model exploration componenets (domain, terms, formulas)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self.get_model_exploration_base_components(mode_status_dict, self.synthesis_results_file,
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            #delta, None, #None, None, None, None, None, None, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, #None, 
            float_approx, float_precision, data_bounds_json_path)
        
#<<<<<<< Updated upstream
        # update mode_status_dict based on interface and model consistency results 
        # computed by get_model_exploration_base_components()
        mode_status_dict['interface_consistent'] = str(interface_consistent).lower()
        mode_status_dict['model_consistent'] == str(model_consistent).lower()
        if not interface_consistent or not model_consistent:
            mode_status_dict['smlp_execution'] = 'aborted'
            mode_status_dict[QUERY_STATUS] = 'ERROR'
            with open(self.synthesis_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        with open(self.synthesis_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        
        # xompute asrt_conj  and synthesis_expr, and then beta expression
        if asrt_exprs is not None:
            assert asrt_names is not None
            asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                    for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
            asrt_conj = self._smlpTermsInst.smlp_and_multi(list(asrt_forms_dict.values()))
            synth_cond_list = asrt_exprs if beta_expr is None else [beta_expr] + asrt_exprs
            synthesis_expr = ' and '.join(synth_cond_list)
        else:
            asrt_conj = smlp.true
            synthesis_expr = beta_expr
        beta = self._smlpTermsInst.smlp_and(beta, asrt_conj) if beta != smlp.true else asrt_conj
        #print('beta str', beta_expr, 'synthesis_expr', synthesis_expr)
        
        # perform / attempt synthesis 
        quer_res = self.query_condition(True, model_full_term_dict, 'synthesis_feasibility', synthesis_expr, beta, 
            domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
        #print('quer_res', quer_res)
        
        # update mode_status_dict based on 'synthesis_feasibility' query results returned by query_condition()
        mode_status_dict[QUERY_FEASIBLE] = str(quer_res['feasible']).lower()
        mode_status_dict[QUERY_STABLE] = str(quer_res['query_status'] == 'STABLE_SAT').lower()
        mode_status_dict[QUERY_STATUS] = 'PASS' if quer_res['query_status'] == 'STABLE_SAT' else 'FAIL'
        
        if quer_res['query_status'] == 'UNSAT': #  synthesis FAIL / not feasible
            self._query_logger.info('Model configuration synthesis is infeasible under given constraints')
        elif quer_res['query_status'] == 'STABLE_SAT': #  synthesis PASS / successful
            witness_vals_dict = quer_res['witness']
            self._query_logger.info('Model configuration synthesis completed successfully')
            #print('witness_vals_dict', witness_vals_dict)
            config_dict = {}
            # TODO: need cleaner code, avoid accessing internal field ._specInst.get_spec_knobs of self._modelTermsInst
            knobs = self._modelTermsInst._specInst.get_spec_knobs
            for key, val in witness_vals_dict.items():
                #print('key', key, 'val', val)
                if key in knobs:
                    config_dict[key] = val
            synthesis_config_dict = {'synthesis':config_dict}; #print('synthesis_config_dict', synthesis_config_dict)
            mode_status_dict[QUERY_WITNESS] = config_dict
            
        mode_status_dict['smlp_execution'] = 'completed'
        with open(self.synthesis_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
