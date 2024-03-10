from fractions import Fraction
import json

import smlp
from smlp_py.smlp_terms import ModelTerms, SmlpTerms
from smlp_py.smlp_utils import np_JSONEncoder


class SmlpQuery:
    def __init__(self):
        self._smlpTermsInst = SmlpTerms()
        self._modelTermsInst = None #ModelTerms()
        
        self._DEF_QUERY_NAMES = None
        self._DEF_QUERY_EXPRS = None
        self.query_params_dict = {
            'query_names': {'abbr':'quer_names', 'default':str(self._DEF_QUERY_NAMES), 'type':str,
                'help':'Names of optimization objectives [default {}]'.format(str(self._DEF_QUERY_NAMES))}, 
            'query_expressions':{'abbr':'quer_exprs', 'default':self._DEF_QUERY_EXPRS, 'type':str,
                'help':'Semicolon seperated list of expressions (functions) to be applied to the responses '
                    'to convert them into optimization objectives ' +
                    '[default: {}]'.format(str(self._DEF_QUERY_EXPRS))}
        }

    def set_logger(self, logger):
        self._query_logger = logger 
        self._smlpTermsInst.set_logger(logger)
        self._modelTermsInst.set_logger(logger)
            
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
    
    @property
    def query_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_query_results.json'
        
    @property
    def certify_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_certify_results.json'
    
    @property
    def synthesis_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_synthesis_results.json'
    
    def find_candidate(self, solver):
        cand_found = solver.check()
        if isinstance(cand_found, smlp.unknown):
            return None
        else:
            return cand_found
    
    # function to check that alpha and eta constraints on inputs and knobs are consistent.
    # TODO: model_full_term_dict is not required here but omiting it causes z3 error 
    # result smlp::z3_solver::check(): Assertion `m.num_consts() == size(symbols)' failed.
    # This is likely because the domain declares mosel outputs as well and without 
    # model_full_term_dict these outputs have no logic (no definition). This function is
    # not a performance bottleneck, but if one wants to speed it up one solution could be
    # to create alpha_eta domain without declaring the outputs and feed it to thos function 
    # instead of the domain that contains output declarations as well (the argument 'domain').
    def check_alpha_eta_consistency(self, domain:smlp.domain, model_full_term_dict:dict, 
            alpha:smlp.form2, eta:smlp.form2, solver_logic:str, mode_status_dict:dict=None):
        solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, False, solver_logic)
        solver.add(alpha); #print('alpha', alpha)
        solver.add(eta); #print('eta', eta)
        res = solver.check(); #print('res', res)
        if mode_status_dict is not None:
            if isinstance(res, smlp.sat):
                self._query_logger.info('Input and knob interface constraints are consistent')
                mode_status_dict['interface_consistent'] = 'true'
            elif isinstance(res, smlp.unsat):
                self._query_logger.info('Input and knob interface constraints are inconsistent')
                mode_status_dict['interface_consistent'] = 'false'
                mode_status_dict['smlp_execution'] = 'completed'
            else:
                raise Exception('alpha and eta cosnsistency check failed to complete')
            #with open(self.certify_results_file, 'w') as f:
            #    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        return mode_status_dict
    
    
    # TODO !!! instead of using witn_form, bind inputs and knobs to their values by directly 
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
        res = solver.check(); #print('res', res)
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
        solver.add(theta)
        solver.add(alpha)
        solver.add(self._smlpTermsInst.smlp_not(query))
        return solver.check() # returns UNSAT or a single SAT model

    # TODO !!!: at least add here the delta condition
    def generalize_counter_example(self, coex):
        return coex
    # TODO !!! extend to concrete witness
    def certify_witness(self, universal:bool, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, witn_dict:dict,
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:float, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int):
        self._query_logger.info('Certifying stability of witness for query ' + str(quer_name) + ':\n   ' + str(witn_dict))
        candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        
        # add the remaining user constraints and the query
        candidate_solver.add(eta); #print('eta', eta)
        candidate_solver.add(alpha); #print('alpha', alpha)
        #candidate_solver.add(beta)
        candidate_solver.add(quer); #print('quer', quer)
        for var,val in witn_dict.items():
            #candidate_solver.add(smlp.Var(var) == smlp.Cnst(val))
            candidate_solver.add(self._smlpTermsInst.smlp_eq(smlp.Var(var), smlp.Cnst(val)))
        
        candidate_check_res = candidate_solver.check() 
        if isinstance(candidate_check_res, smlp.sat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is a valid witness; checking its stability')
        elif isinstance(candidate_check_res, smlp.unsat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is not a valid witness (even without stability requirements)')
            return 'not a witness'
        else:
            raise Exception('Unexpected counter-example ststus ' + str(ce) + ' in function certify_witness')
        
        # checking stability of a valid witness to the query
        witn_term_dict = self._smlpTermsInst.witness_const_to_term(witn_dict)
        ce = self.find_candidate_counter_example(universal, domain, witn_term_dict, quer, model_full_term_dict, alpha, 
            theta_radii_dict, solver_logic)
        if isinstance(ce, smlp.sat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is not stable for radii ' + str(theta_radii_dict))
            return 'witness, not stable'
        elif isinstance(ce, smlp.unsat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is stable for radii ' + str(theta_radii_dict))
            return 'stable witness'
        else:
            raise Exception('Unexpected counter-example ststus ' + str(ce) + ' in function certify_witness')
    
    '''            
    def certify_witnesses(self, universal:bool, model_full_term_dict:dict, quer_names:list[str], quer_exprs:list[str], quer_forms_dict:dict, witn_dict:dict,
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict,
            delta:dict, solver_logic, witn:bool, sat_approx:bool, sat_precision:int): 
        print('quer_forms_dict', quer_forms_dict.keys(), 'quer_names', quer_names)
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            witn_i_dict = witn_dict[quer_name]
            witness_status_str = self.certify_witness(universal, model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, witn_i_dict, domain, eta, alpha, theta_radii_dict, delta, solver_logic, witn, sat_approx, sat_precision)
            quer_res_dict[quer_name] = witness_status_str
            
        #print('quer_res_dict', quer_res_dict)
        with open(self.certify_results_file, 'w') as f:
            json.dump(quer_res_dict, f, indent='\t', cls=np_JSONEncoder)
    '''
    def smlp_certify(self, syst_expr_dict:dict, algo:str, model:dict, #universal:bool, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            quer_names:list[str], quer_exprs:list[str], witn_dict:dict, delta:dict,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        
        query_status_dict = {
            'witness_consistent': 'unknown',
            'witness_to_query': 'unknown',
            'witness_stable': 'unknown'
        }
        
        # initialize the fields in the ststus dictionary mode_status_dict as unknown/running
        mode_status_dict = {}
        for quer_name in quer_names:
            # .copy() is required in next line to ensure updating different copies will be independent
            mode_status_dict[quer_name] = query_status_dict.copy() 
        mode_status_dict['smlp_execution'] = 'running'
        mode_status_dict['interface_consistent'] = 'unknown'
        
        # compute model exploration base componensts (models, constraints, etc.)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            delta, None, #None, None, None, None, quer_names, quer_exprs, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        #print('syst_expr_dict', syst_expr_dict); print('model_full_term_dict', model_full_term_dict);
        #print('smlp_certify: beta_expr', beta_expr, 'quer_exprs', quer_exprs)
        #with open(witn_file+'.json', 'r') as wf:
        #    witn_dict = json.load(wf, parse_float=Fraction); 
        
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
                if var in feat_names: # TODO: it is enough to only add knob values (and ignore input values)
                    witn_dict_filtered[q][var] = val
            witn_form_dict[q] = self._smlpTermsInst.witness_to_formula(witn_dict_filtered[q])
                                              
            # assert every input and know is assigned in w:
            w_keys = w.keys()
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
            #print('before consistency', mode_status_dict)
            # input and knob constraints consistency check
            mode_status_dict = self.check_alpha_eta_consistency(domain, model_full_term_dict, alpha, eta, solver_logic, mode_status_dict)
            with open(self.certify_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                
            if mode_status_dict['interface_consistent'] == 'false':
                with open(self.certify_results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                self._query_logger.info('Input and knob interface constraints are inconsistent; aborting...')
                return
            #print('after consistency', mode_status_dict)
            #print('quer_forms_dict', quer_forms_dict.keys(), 'quer_names', quer_names)
            assert list(quer_forms_dict.keys()) == quer_names
            #print('witn_form_dict', witn_form_dict)
            quer_res_dict = {}
            for quer_name, quer_form in quer_forms_dict.items():
                #print('quer_name', quer_name, 'quer_form', quer_form)
                witn_form = witn_form_dict[quer_name]; #print('witn_form', witn_form)
                # checking concrete witneses consistency alpha and beta (w/o looking at respective queries)
                self._query_logger.info('Certifying consistency of witness for query ' + str(quer_name) + ':\n   ' + str(witn_form))
                witn_status = self.check_concrete_witness_consistency(domain, model_full_term_dict, 
                    alpha, eta, None, witn_form, solver_logic)
                if isinstance(witn_status, smlp.sat):
                    self._query_logger.info('Input, knob and concrete witness constraints are consistent')
                    mode_status_dict[quer_name]['witness_consistent'] = 'true'
                elif isinstance(witn_status, smlp.unsat):
                    self._query_logger.info('Input, knob and concrete witness constraints are inconsistent')
                    mode_status_dict[quer_name]['witness_consistent'] = 'false'
                    mode_status_dict[quer_name]['witness_to_query'] = 'false'
                    mode_status_dict[quer_name]['witness_stable'] = 'false'
                    continue
                else:
                    raise Exception('Input, knob and concrete witness cosnsistency check failed to complete')
                #print('after witness_consistence', mode_status_dict)
                with open(self.certify_results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                
                # concrete witneses consistency check, including checking that the query is valid too
                witn_i_dict = witn_dict[quer_name]
                witness_status_str = self.certify_witness(False, model_full_term_dict, quer_name, quer_expr_dict[quer_name], quer_form,
                    witn_i_dict, domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
                quer_res_dict[quer_name] = witness_status_str
                if witness_status_str == 'not a witness':
                    #print(quer_name, 'case not a witness')
                    mode_status_dict[quer_name]['witness_to_query'] = 'false'
                    mode_status_dict[quer_name]['witness_stable'] = 'false'
                elif witness_status_str == 'witness, not stable':
                    #print(quer_name, 'case witness, not stable')
                    mode_status_dict[quer_name]['witness_to_query'] = 'true'
                    mode_status_dict[quer_name]['witness_stable'] = 'false'
                elif witness_status_str == 'stable witness':
                    #print(quer_name, 'case stable witness')
                    mode_status_dict[quer_name]['witness_to_query'] = 'true'
                    mode_status_dict[quer_name]['witness_stable'] = 'true'
                else:
                    raise Exception('Unexpected status ' + str() + ' in function ' + certify_witness)
                with open(self.certify_results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)                
                
            mode_status_dict['smlp_execution'] = 'completed'
            with open(self.certify_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            #print('quer_res_dict', quer_res_dict)
            #print('mode_status_dict', mode_status_dict)

    # TODO !!!: implement timeout ? UNKNOWN return value
    # x -- inputs/features (theta will know which ones are knobs vs free)
    # y -- outputs/responses # TODO !!! or y is a point within Q radius from x? this beta and 
    # Find a solution to: Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ query(x,y)) assuming 
    # solver already knows model definitions model(x) = y
    def query_condition(self, universal, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:dict, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int, mode_status_dict:dict=None): 
        #def update_and_return(quer_name, result, update_var, update_val, status_dict):
        #    status_dict[quer_name][update_var] = update_val; print('status_dict', status_dict)
        #    return result
            
        if mode_status_dict is not None:
            if quer_name not in mode_status_dict.keys():
                mode_status_dict[quer_name] = {}
        
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
        while True:
            # solve Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ query)
            print('searching for a candidate')
            ca = self.find_candidate(candidate_solver)
            #print('ca', ca)
            if isinstance(ca, smlp.sat):
                print('candidate found -- checking stability')
                if mode_status_dict is not None:
                    mode_status_dict[quer_name]['witness_exists'] = 'true'
                ce = self.find_candidate_counter_example(universal, domain, ca.model, quer, model_full_term_dict, alpha, 
                    theta_radii_dict, solver_logic)
                if isinstance(ce, smlp.sat):
                    print('candidate not stable -- continue search')
                    cem = ce.model.copy(); #print('cem', cem)
                    for var in ce.model.keys():
                        if var in model_full_term_dict.keys():
                            del cem[var]
                    lemma = self.generalize_counter_example(cem) #
                    #print('theta_radii_dict in lemma', theta_radii_dict)
                    theta = self._modelTermsInst.compute_stability_formula_theta(lemma, delta, theta_radii_dict, universal)
                    candidate_solver.add(self._smlpTermsInst.smlp_not(theta))
                    continue
                elif isinstance(ce, smlp.unsat):
                    print('candidate stable -- return candidate')
                    if witn: # export witness (use numbers as values, not terms)
                        witness_vals_dict = self._smlpTermsInst.witness_term_to_const(ca.model, sat_approx,  
                            sat_precision)
                        #print('domain witness_vals_dict', witness_vals_dict)
                        # sanity check: the value of query in the sat assignment should be true
                        if quer_expr is not None:
                            quer_ce_val = eval(quer_expr, {},  witness_vals_dict); #print('quer_ce_val', quer_ce_val)
                            assert quer_ce_val
                        if mode_status_dict is not None:
                            mode_status_dict[quer_name]['stable_witness'] = 'true'
                            mode_status_dict[quer_name]['query_status'] = 'STABLE_SAT'
                            mode_status_dict[quer_name]['query_witness'] = witness_vals_dict
                            #print('mode_status_dict', mode_status_dict)
                        return {'status':'STABLE_SAT', 'witness':witness_vals_dict}
                    else:
                        #print('mode_status_dict', mode_status_dict)
                        if mode_status_dict is not None:
                            mode_status_dict[quer_name]['stable_witness'] = 'true'
                            mode_status_dict[quer_name]['query_status'] = 'STABLE_SAT'
                            mode_status_dict[quer_name]['query_witness'] = ca.model
                            #print('mode_status_dict', mode_status_dict)
                        return {'status':'STABLE_SAT', 'witness':ca.model}
            elif isinstance(ca, smlp.unsat):
                #print('query unsuccessful: witness does not exist (query is unsat)')
                if mode_status_dict is not None:
                    mode_status_dict[quer_name]['witness_exists'] = 'false'
                    mode_status_dict[quer_name]['stable_witness'] = 'false'
                    mode_status_dict[quer_name]['query_status'] = 'UNSAT'
                    mode_status_dict[quer_name]['query_witness'] = None
                    #print('mode_status_dict', mode_status_dict)
                return {'status':'UNSAT', 'witness':None}
            elif isinstance(ca, smlp.unknown):
                self._opt_logger.info('Completed with result: {}'.format('UNKNOWN'))
                if mode_status_dict is not None:
                    mode_status_dict[quer_name]['witness_exists'] = 'unknown'
                    mode_status_dict[quer_name]['stable_witness'] = 'unknown'
                    mode_status_dict[quer_name]['query_status'] = 'UNSAT'
                    mode_status_dict[quer_name]['query_witness'] = None
                    #print('mode_status_dict', mode_status_dict)
                return {'status':'UNKNOWN', 'witness':None}
                #raise Exception('UNKNOWN return value in candidate search is currently not supported for queries')
            else:
                raise Exception('Unexpected return value ' + str(type(ca)) + ' in candidate search for queries')

            
    def query_conditions(self, universal:bool, model_full_term_dict:dict, quer_names:str, quer_exprs:str, quer_forms_dict:dict, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict,
            delta:dict, solver_logic, witn:bool, sat_approx:bool, sat_precision:int, mode_status_dict:dict=None): 
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            quer_res_dict[quer_name] = self.query_condition(universal, model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, domain, eta, alpha, theta_radii_dict, delta, solver_logic, witn, sat_approx, sat_precision, mode_status_dict) 
        #print('quer_res_dict', quer_res_dict)
        #with open(self.query_results_file, 'w') as f:
        #    json.dump(quer_res_dict, f, indent='\t', cls=np_JSONEncoder) #cls= , use_decimal=True

    # querying conditions on a model to find a stable witness satisfying this condition in entire stability region
    # (defined by stability/theta radii) around that witness (which is a SAT assignment to model interface variables)
    def smlp_query(self, syst_expr_dict:dict, algo:str, model:dict, universal:bool, model_features_dict:dict,  
            feat_names:list[str], resp_names:list[str], quer_names:list[str], quer_exprs:list[str], delta:float,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        
        # initialize the fields in the more status dictionary mode_status_dict as unknown/running
        mode_status_dict = {'smlp_execution': 'running', 'interface_consistent': 'unknown'}
        for quer_name in quer_names:
            mode_status_dict[quer_name] = {
                'witness_exists': 'unknown',
                'stable_witness': 'unknown',
                'query_status': 'unknown',
                'query_witness': None} 
        with open(self.query_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
        if quer_exprs is None:
            self._query_logger.error('Queries were not specified in the "query" mode: aborting...')
            mode_status_dict['smlp_execution'] = 'aborted'
            with open(self.query_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        
        # compute model exploration componenets (domain, terms, formulas)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components( 
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, delta, None,
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        
        quer_forms_dict = dict([(quer_name, self._smlpTermsInst.ast_expr_to_term(quer_expr)) \
                for quer_name, quer_expr in zip(quer_names, quer_exprs)])
        
        # sanity check -- queries must be formulas (not terms)
        for i, form in enumerate(quer_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Query ' + str(quer_exprs[i]) + ' must be a formula (not a term)')
        
        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            # alpha-eta consistency check
            mode_status_dict = self.check_alpha_eta_consistency(domain, model_full_term_dict, alpha, eta, solver_logic, mode_status_dict)
            if mode_status_dict['interface_consistent'] == 'false':
                mode_status_dict['smlp_execution'] = 'aborted'
                with open(self.query_results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                self._query_logger.info('Input and knob interface constraints are inconsistent; aborting...')
                return
            with open(self.query_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                
            # alpha, eta and model consistency check    
            quer_res = self.query_condition( # universal
                False, model_full_term_dict, 'consistency_check', 'True', smlp.true, domain,
                eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision, None) 
            if quer_res['status'] == 'UNSAT':
                mode_status_dict['model_consistent'] = 'false'
                mode_status_dict['smlp_execution'] = 'aborted'
            else:
                mode_status_dict['model_consistent'] = 'true'
            
            with open(self.query_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
            if mode_status_dict['model_consistent'] == 'false':
                self._query_logger.info('Model querying instance is inconsistent; aborting...')
                return
                
        self.query_conditions( # universal
            False, model_full_term_dict, quer_names, quer_exprs, quer_forms_dict, domain, 
            eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision, mode_status_dict)
        
        mode_status_dict['smlp_execution'] = 'completed'
        #print('final mode_status_dict', mode_status_dict)
        with open(self.query_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
            
            
    # SMLP synthesis mode: configure knobs with stability to meet beta constraints under alpha, eta, theta and constraints
    def smlp_synthesize(self, syst_expr_dict:dict, algo:str, model:dict, #X:pd.DataFrame, y:pd.DataFrame, 
            model_features_dict:dict, feat_names:list[str], resp_names:list[str], asrt_names, asrt_exprs, 
            delta:float, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, #scale_objv:bool, 
            sat_thresholds:bool,
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):

        # initialize the fields in the more status dictionary mode_status_dict as unknown/running
        mode_status_dict = {
            'smlp_execution': 'running', 
            'interface_consistent': 'unknown',
            'model_consistent': 'unknown',
            'witness_exists': 'unknown',
            'stable_witness': 'unknown',
            'query_status': 'unknown',
            'query_witness': None}
        with open(self.synthesis_results_file, 'w') as f:
            json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
        
        #print('eta_expr', eta_expr); print('alph_expr', alph_expr); print('beta_expr', beta_expr); #assert False
        # compute model exploration componenets (domain, terms, formulas)
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            delta, None, #None, None, None, None, None, None, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, None, 
            float_approx, float_precision, data_bounds_json_path)
        
        if asrt_exprs is not None:
            assert asrt_names is not None
            asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                    for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
            asrt_conj = self._smlpTermsInst.smlp_and_multi(list(asrt_forms_dict.values()))
            synthesis_expr = beta_expr
            for asrt_expr in asrt_exprs:
                synthesis_expr = '{} and {}'.format(synthesis_expr, asrt_expr)
        else:
            asrt_conj = smlp.true
            synthesis_cond = beta_expr
        
        beta = self._smlpTermsInst.smlp_and(beta, asrt_conj) if beta != smlp.true else asrt_conj
         
        if vacuity:
            # alpha-eta consistency check
            mode_status_dict = self.check_alpha_eta_consistency(domain, model_full_term_dict, alpha, eta, solver_logic, mode_status_dict)
            if mode_status_dict['interface_consistent'] == 'false':
                mode_status_dict['smlp_execution'] = 'aborted'
                with open(self.synthesis_results_file, 'w') as f:
                    json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
                self._query_logger.info('Input and knob interface constraints are inconsistent; aborting...')
                return
            with open(self.synthesis_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)

            # alpha, eta and model consistency check    
            quer_res = self.query_condition( # universal
                True, model_full_term_dict, 'consistency_check', 'True', smlp.true, domain,
                eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision, None) 
            if quer_res['status'] == 'UNSAT':
                mode_status_dict['model_consistent'] = 'false'
                mode_status_dict['smlp_execution'] = 'aborted'
            else:
                mode_status_dict['model_consistent'] = 'true'
            
            with open(self.synthesis_results_file, 'w') as f:
                json.dump(mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
            if mode_status_dict['model_consistent'] == 'false':
                self._query_logger.info('Model querying instance is inconsistent; aborting...')
                return
        #print('beta str', beta_expr, 'synthesis_expr', synthesis_expr)
        # synthesis feasibility check
        quer_res = self.query_condition(True, model_full_term_dict, 'synthesis_feasibility', synthesis_expr, beta, 
            domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
        #print('quer_res', quer_res)
        if quer_res['status'] == 'UNSAT':
            self._query_logger.info('Model configuration synthesis is infeasible under given constraints')
            return True, None
        elif quer_res['status'] == 'STABLE_SAT':
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
            synthesis_config_dict = {'synthesis':config_dict}
            #print('synthesis_config_dict', synthesis_config_dict)
            with open(self.synthesis_results_file, 'w') as f:
                json.dump(synthesis_config_dict, f, indent='\t', cls=np_JSONEncoder)
            return False, witness_vals_dict