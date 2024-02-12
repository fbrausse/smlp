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
                    '[default: {}]'.format(str(self._DEF_QUERY_EXPRS))},
            'witness_file': {'abbr':'witness', 'default':None, 'type':str, 
                'help': 'File name (including the path) that specifies a (potential) ' +
                        'stable witness for a query, under all domain constraints ' +
                        '[default: None]'}
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
    
    def find_candidate(self, solver):
        cand_found = solver.check()
        if isinstance(cand_found, smlp.unknown):
            return None
        else:
            return cand_found
    
    # TODO: what about eta-interval and eta_global constraints, as well as eta-grid constraints for
    # integer control (knob) variables? Looks like they should be used as constrints -- look for a cex
    # to a candidate only under these (and other). Grid constraints for continuous variable should not 
    # be used
    #   ! ( theta x y -> alpha y -> beta y /\ obj y >= T ) =
    #   ! ( ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
    #   theta x y /\ alpha y /\ ! ( beta y /\ obj y >= T) 
    def find_candidate_counter_example(self, domain:smlp.domain, cand:dict, query:smlp.form2, 
            model_full_term_dict:dict, alpha:smlp.form2, theta_radii_dict:dict, solver_logic:str): #, beta:smlp.form2
        if False:
            solver = smlp.solver(False)
            solver.declare(domain)

            # let solver know definition of responses; we will see response names in the sat models
            for resp_name, resp_term in model_full_term_dict.items():
                solver.add(smlp.Var(resp_name) == resp_term)
        else:
            solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
                domain, model_full_term_dict, False, solver_logic)
        theta = self._modelTermsInst.compute_stability_formula_theta(cand, None, theta_radii_dict) 
        solver.add(theta)
        solver.add(alpha)
        solver.add(self._smlpTermsInst.smlp_not(query))
        return solver.check() # returns UNSAT or a single SAT model

    # TODO !!!: at least add here the delta condition
    def generalize_counter_example(self, coex):
        return coex

    def certify_witness(self, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, witn_dict:dict,
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:float, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int):
        self._query_logger.info('Certifying stability of witness for query ' + str(quer_name) + ':\n   ' + str(witn_dict))
        candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        
        # add the remaining user constraints and the query
        candidate_solver.add(eta); print('eta', eta)
        candidate_solver.add(alpha); print('alpha', alpha)
        #candidate_solver.add(beta)
        candidate_solver.add(quer); print('quer', quer)
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
        ce = self.find_candidate_counter_example(domain, witn_term_dict, quer, model_full_term_dict, alpha, 
            theta_radii_dict, solver_logic)
        if isinstance(ce, smlp.sat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is not stable for radii ' + str(theta_radii_dict))
            return 'witness, not stable'
        elif isinstance(ce, smlp.unsat):
            self._query_logger.info('Witness to query ' + str(quer_name) + ' is stable for radii ' + str(theta_radii_dict))
            return 'stable witness'
        else:
            raise Exception('Unexpected counter-example ststus ' + str(ce) + ' in function certify_witness')
    
    def certify_witnesses(self, model_full_term_dict:dict, quer_names:list[str], quer_exprs:list[str], quer_forms_dict:dict, witn_dict:dict,
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict,
            delta:float, solver_logic, witn:bool, sat_approx:bool, sat_precision:int): 
        print('quer_forms_dict', quer_forms_dict.keys(), 'quer_names', quer_names)
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            witn_i_dict = witn_dict[quer_name]
            witness_status_str = self.certify_witness(model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, witn_i_dict, domain, eta, alpha, theta_radii_dict, delta, solver_logic, witn, sat_approx, sat_precision)
            #witness_status_str = 'stable witness' if witness_status_bool else 'witness, not stable'
            quer_res_dict[quer_name] = witness_status_str #{'witness': list(witn_i_dict.items()), 'status':witness_status_str} 
            
        #print('quer_res_dict', quer_res_dict)
        with open(self.certify_results_file, 'w') as f:
            json.dump(quer_res_dict, f, indent='\t', cls=np_JSONEncoder)

    def smlp_certify(self, algo:str, model:dict, model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            quer_names:list[str], quer_exprs:list[str], witn_file:str, delta:float,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        
        domain, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            algo, model, model_features_dict, feat_names, resp_names, 
            delta, None, #None, None, None, None, quer_names, quer_exprs, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        
        #print('smlp_certify: beta_expr', beta_expr, 'quer_exprs', quer_exprs)
        with open(witn_file+'.json', 'r') as wf:
            witn_dict = json.load(wf, parse_float=Fraction); 

        # drop fron witn_dict assignments to variables that do not occur in the model
        witn_dict_filtered = {}
        for q, w in witn_dict.items():
            assert isinstance(w, dict)
            witn_dict_filtered[q] = {}
            for var,val in w.items():
                if var in feat_names: # TODO: it i senough to only add knov values (and ignore input values)
                    witn_dict_filtered[q][var] = val
        
        witn_count = len(witn_dict.keys()); print('witn_count', witn_count)
        print('quer_names', quer_names, 'quer_exprs', quer_exprs)
        # TODO !!!! do we need this way of defining queries as smlp.true?
              
        if quer_names is None and quer_exprs is None:
            print('updating quer_exprs')
            quer_names = list(witn_dict_filtered.keys())
            quer_exprs = ['True'] * witn_count
        
        if beta_expr is not None:
            quer_exprs = [quer_exprs[i] + ' and ' + beta_expr for i in range(witn_count)]; print('quer_exprs', quer_exprs)
        quer_forms_dict = dict([(quer_name, self._smlpTermsInst.ast_expr_to_term(quer_expr)) \
            for quer_name, quer_expr in zip(quer_names, quer_exprs)])
        
        # sanity check -- queries must be formulas (not terms)
        for i, form in enumerate(quer_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Quesry ' + str(quer_exprs[i]) + ' must be a formula (not a term)')
        
        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            quer_res = self.query_condition(
                model_full_term_dict, 'consistency_check', 'True', smlp.true, domain,
                eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision) 
            print('quer_res', quer_res)
            if quer_res['status'] == 'UNSAT':
                self._query_logger.info('Model querying instance is inconsistent; aborting...')
                return
            elif quer_res['status'] == 'STABLE_SAT':
                self._query_logger.info('Model querying instance is consistent; vacuity check was successful')
                '''
                witness_vals_dict = self._smlpTermsInst.witness_term_to_const(ca.model, sat_approx,  
                            sat_precision)
                
                '''
            else:
                raise Exception('Unexpected status ' + str(quer_res['status']) + ' in vacuity check in certify mode')
                
        
        self.certify_witnesses(model_full_term_dict, quer_names, quer_exprs, quer_forms_dict, witn_dict_filtered, domain, 
            eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
       
    # TODO !!!: implement timeout ? UNKNOWN return value
    # x -- inputs/features (theta will know which ones are knobs vs free)
    # y -- outputs/responses # TODO !!! or y is a point within Q radius from x? this beta and 
    #Find a solution to: Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ query(x,y)) assuming 
    # solver already knows model definitions model(x) = y
    def query_condition(self, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, #beta:smlp.form2, 
            delta:float, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int): 
        if quer_expr is not None:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer_expr)))
        else:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer)))
        #print('query', quer, 'eta', eta, 'delta', delta)
        if False:
            candidate_solver = smlp.solver(incremental=True)
            candidate_solver.declare(domain)

            # let solver know definition of responses
            for resp_name, resp_term in model_full_term_dict.items():
                candidate_solver.add(smlp.Var(resp_name) == resp_term)
        else:
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
                ce = self.find_candidate_counter_example(domain, ca.model, quer, model_full_term_dict, alpha, 
                    theta_radii_dict, solver_logic)
                if isinstance(ce, smlp.sat):
                    print('candidate not stable -- continue search')
                    lemma = self.generalize_counter_example(ce.model)
                    #print('theta_radii_dict in lemma', theta_radii_dict)
                    theta = self._modelTermsInst.compute_stability_formula_theta(lemma, delta, theta_radii_dict)
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
                        return {'status':'STABLE_SAT', 'witness':witness_vals_dict}
                    else:
                        return {'status':'STABLE_SAT', 'witness':ca.model}
            elif isinstance(ca, smlp.unsat):
                print('query unsuccessful: witness does not exist (query is unsat)')
                return {'status':'UNSAT', 'witness':None}
            elif isinstance(ca, smlp.unknown):
                self._opt_logger.info('Completed with result: {}'.format('UNKNOWN'))
                return {'status':'UNKNOWN', 'witness':None}
                #raise Exception('UNKNOWN return value in candidate search is currently not supported for queries')
            else:
                raise Exception('Unexpected return value ' + str(type(ca)) + ' in candidate search for queries')
            
    def query_conditions(self, model_full_term_dict:dict, quer_names:str, quer_exprs:str, quer_forms_dict:dict, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict,
            delta:float, solver_logic, witn:bool, sat_approx:bool, sat_precision:int): 
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            quer_res_dict[quer_name] = self.query_condition(model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, domain, eta, alpha, theta_radii_dict, delta, solver_logic, witn, sat_approx, sat_precision) 
        #print('quer_res_dict', quer_res_dict)
        with open(self.query_results_file, 'w') as f:
            json.dump(quer_res_dict, f, indent='\t', cls=np_JSONEncoder) #cls= , use_decimal=True

    # querying conditions on a model to find a stable witness satisfying this condition in entire stability region
    # (defined by stability/theta radii) around that witness (which is a SAT assignment to model interface variables)
    def smlp_query(self, algo:str, model:dict, model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            quer_names:list[str], quer_exprs:list[str], delta:float,
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool, float_approx=True, float_precision=64, 
            data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        if quer_exprs is None:
            self._query_logger.error('Queries were not specified in the "query" mode: aborting...')
            return
            
        domain, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            algo, model, model_features_dict, feat_names, resp_names, delta, None, 
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
            quer_res = self.query_condition(
                model_full_term_dict, 'consistency_check', 'True', smlp.true, domain,
                eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision) 
            if quer_res['status'] == 'UNSAT':
                self._query_logger.info('Model querying instance is inconsistent; aborting...')
                return
        
        self.query_conditions(model_full_term_dict, quer_names, quer_exprs, quer_forms_dict, domain, 
            eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)