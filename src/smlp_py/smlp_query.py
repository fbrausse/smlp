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
    
    def find_candidate(self, solver):
        cand_found = solver.check()
        if isinstance(cand_found, smlp.unknown):
            return None
        else:
            return cand_found

    def get_queries(self, arg_query_names, arg_query_exprs, commandline_condition_separator):
        if arg_query_exprs is None:
            return None, None
        else:
            query_exprs = arg_query_exprs.split(commandline_condition_separator)
            if arg_query_names is not None:
                query_names = arg_query_names.split(',')
            else:
                query_names = ['query_'+str(i) for i in enumerate(len(query_exprs))];
        assert query_names is not None and query_exprs is not None
        assert len(query_names) == len(query_exprs); 
        print('query_names', query_names); print('query_exprs', query_exprs)
        return query_names, query_exprs
    
    # TODO:what about eta-interval and eta_global constraints, as well as eta-grid constraints for
    # integer control (knob) variables? Looks like they should be used as constrints -- look for a cex
    # to a candidate only under these (and other). Grif constraints for continuous variable should not 
    # be used (Shai's use case)
    #   ! ( theta x y -> alpha y -> beta y /\ obj y >= T ) =
    #   ! ( ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
    #   theta x y /\ alpha y /\ ! ( beta y /\ obj y >= T) 
    def find_candidate_counter_example(self, domain:smlp.domain, cand:dict, query:smlp.form2, 
            model_full_term_dict:dict, alpha:smlp.form2): #, beta:smlp.form2
        solver = smlp.solver(False)
        solver.declare(domain) # pp.dom
        
        # let solver know definition of responses; we will see response names in the sat models
        for resp_name, resp_term in model_full_term_dict.items():
            solver.add(smlp.Var(resp_name) == resp_term)
        
        theta = self._modelTermsInst.compute_stability_formula_theta(cand, None) 
        #print('theta', theta)
        solver.add(theta)
        solver.add(alpha)
        solver.add(self._smlpTermsInst.smlp_not(query))
        return solver.check() # returns UNSAT or a single SAT model

    # TODO !!!: at least add here the delta condition
    def generalize_counter_example(self, coex):
        return coex

    
    # TODO !!!: implement timeout ? UNKNOWN return value
    # x -- inputs/features (theta will know which ones are knobs vs free)
    # y -- outputs/responses # TODO !!! or y is a point within Q radius from x? this beta and 
    #Find a solution to: Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ query(x,y)) assuming 
    # solver already knows model definitions model(x) = y
    def query_condition(self, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:smlp.form2, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, #beta:smlp.form2, 
            delta:float, witn:bool, sat_approx:bool, sat_precision:int): 
        if quer_expr is not None:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer_expr)))
        else:
            self._query_logger.info('Querying condition {} <-> {}'.format(str(quer_name), str(quer)))
        #print('query', quer, 'eta', eta, 'delta', delta)
        candidate_solver = smlp.solver(incremental=True)
        candidate_solver.declare(domain) #pp.dom
        
        # let solver know definition of responses
        for resp_name, resp_term in model_full_term_dict.items():
            candidate_solver.add(smlp.Var(resp_name) == resp_term)
        
        # add the remaining iser constraints and the query
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
                ce = self.find_candidate_counter_example(domain, ca.model, quer, model_full_term_dict, alpha)
                if isinstance(ce, smlp.sat):
                    print('candidate not stable -- continue search')
                    lemma = self.generalize_counter_example(ce.model)
                    theta = self._modelTermsInst.compute_stability_formula_theta(lemma, delta)
                    candidate_solver.add(self._smlpTermsInst.smlp_not(theta))
                    continue
                elif isinstance(ce, smlp.unsat):
                    print('candidate stable -- return candidate')
                    if witn: # export witness (use numbers as values, not terms)
                        sat_model_vals_dict = self._smlpTermsInst.sat_model_term_to_const(ca.model, sat_approx,  
                            sat_precision)
                        #print('domain sat_model_vals_dict', sat_model_vals_dict)
                        # sanity check: the value of query in the sat assignment should be true
                        if quer_expr is not None:
                            quer_ce_val = eval(quer_expr, {},  sat_model_vals_dict); print('quer_ce_val', quer_ce_val)
                            assert quer_ce_val
                        return {'status':'STABLE_SAT', 'witness':sat_model_vals_dict}
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
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, #beta:smlp.form2, 
            delta:float, witn:bool, sat_approx:bool, sat_precision:int): 
        assert list(quer_forms_dict.keys()) == quer_names
        quer_res_dict = {}
        for i, (quer_name, quer_form) in enumerate(quer_forms_dict.items()):
            quer_res_dict[quer_name] = self.query_condition(model_full_term_dict, quer_name, quer_exprs[i], 
                quer_form, domain, eta, alpha, delta, witn, sat_approx, sat_precision) 
        #print('quer_res_dict', quer_res_dict)
        with open(self.query_results_file, 'w') as f:
            json.dump(quer_res_dict, f, indent='\t', cls=np_JSONEncoder) #cls= , use_decimal=True


    def smlp_query(self, algo, model, X, y, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, pareto, asrt_names, asrt_exprs, quer_names, quer_exprs, delta, epsilon,
            alph_expr:str, beta_expr:str, eta_expr:str, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        domain, model_full_term_dict, eta, alpha, beta, base_solver = self._modelTermsInst.create_model_exploration_base_instance(
            algo, model, X, y, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, asrt_names, asrt_exprs, quer_names, quer_exprs, delta, epsilon, 
            alph_expr, beta_expr, eta_expr, True, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)
        print('eta, alpha, beta', eta, alpha, beta)
        quer_forms_dict = dict([(quer_name, self._smlpTermsInst.ast_expr_to_term(quer_expr)) \
                for quer_name, quer_expr in zip(quer_names, quer_exprs)])
        # sanity check --queries must be formulas (not terms)
        for i, form in enumerate(quer_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Assertion ' + str(quer_exprs[i]) + ' must be a formula (not a term)')
        self.query_conditions(model_full_term_dict, quer_names, quer_exprs, quer_forms_dict, domain, 
            eta, alpha, delta, True, float_approx, float_precision)