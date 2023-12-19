from fractions import Fraction
import json

import smlp
from smlp_py.smlp_terms import ModelTerms, SmlpTerms
from smlp_py.smlp_utils import np_JSONEncoder

class SmlpVerify:
    def __init__(self):
        self._smlpTermsInst = SmlpTerms()
        self._modelTermsInst = None #ModelTerms()

    def set_logger(self, logger):
        self._verify_logger = logger 
        self._smlpTermsInst.set_logger(logger)
        self._modelTermsInst.set_logger(logger)
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self._modelTermsInst.set_report_file_prefix(report_file_prefix)
        
    # model_file_prefix is a string used as prefix in all saved model files of SMLP
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
        self._modelTermsInst.set_model_file_prefix(model_file_prefix)
    
    # set self._modelTermsInst ModelTerms()
    def set_model_terms_inst(self, model_terms_inst):
        self._modelTermsInst = model_terms_inst
    
    @property
    def assertions_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_assertions_results.json'

    def verify_asrt(self, model_full_term_dict:dict, asrt_name:str, asrt_expr:str, asrt_form:smlp.form2, 
            domain:smlp.domain, alpha:smlp.form2, beta:smlp.form2, eta:smlp.form2, solver_logic:str, sat_approx:bool, sat_precision:int):
        self._verify_logger.info('Verifying assertion {} <-> {}'.format(str(asrt_name), str(asrt_expr)))
        # TODO !!!: take care of usage of beta; currently we assume that if beta is required it is part of assertion
        assert beta == smlp.true

        solver_instance = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        solver_instance.add(alpha)
        #solver_instance.add(eta)
        solver_instance.add(self._smlpTermsInst.smlp_not(asrt_form))
        res = solver_instance.check(); #self.print_result(res)
        
        if isinstance(res, smlp.unsat):
            self._verify_logger.info('Completed with result: {}'.format('UNSAT'))
            asrt_res_dict = {'status':'UNSAT', 'asrt':None, 'model':None}
        elif isinstance(res, smlp.sat):
            self._verify_logger.info('Completed with result: {}'.format('SAT'))
            #print('res/model', res.model, type(res.model), type(res))
            sat_model_vals_dict = self._smlpTermsInst.sat_model_term_to_const(res.model, 
                approximate=sat_approx, precision=sat_precision)
            #print('domain sat_model_vals_dict', sat_model_vals_dict)
            # sanity check: the value of the negated assertion in the sat assignment should be true
            asrt_ce_val = eval(asrt_expr, {},  sat_model_vals_dict); #print('asrt_ce_val', asrt_ce_val)
            assert not asrt_ce_val
            asrt_res_dict = {'status':'SAT', 'asrt': asrt_ce_val, 'model':sat_model_vals_dict}
        elif isinstance(res, smlp.unknown):
            self._verify_logger.info('Completed with result: {}'.format('UNKNOWN'))
            # TODO !!!: add reason for UNKNOWN or report that reason as 'status' field
            asrt_res_dict = {'status':'UNKNOWN', 'asrt':None, 'model':None}
        else:
            raise Exception('Unexpected resuld from solver')
        return asrt_res_dict
        
        
    def verify_assertions(self, model_full_term_dict:dict, asrt_names:list, asrt_exprs:list, asrt_forms_dict:dict, 
            domain:smlp.domain, alpha:smlp.form2, beta:smlp.form2, eta:smlp.form2, solver_logic:str, sat_approx=False, sat_precision=64):
        #print('asrt_forms_dict', asrt_forms_dict)
        assert list(asrt_forms_dict.keys()) == asrt_names
        asrt_res_dict = {}
        for i, (asrt_name, asrt_form) in enumerate(asrt_forms_dict.items()):
            asrt_res_dict[asrt_name] = self.verify_asrt(model_full_term_dict, asrt_name, asrt_exprs[i], asrt_form, 
                domain, alpha, beta, eta, solver_logic, sat_approx, sat_precision)
        #print('asrt_res_dict', asrt_res_dict)
        with open(self.assertions_results_file, 'w') as f: #json.dump(asrt_res_dict, f)
            json.dump(asrt_res_dict, f, indent='\t', cls=np_JSONEncoder) #cls= , use_decimal=True
            
    def smlp_verify(self, algo, model, model_features_dict, feat_names, resp_names, asrt_names, asrt_exprs,
            alph_expr:str, solver_logic:str, vacuity:bool, data_scaler, scale_feat, scale_resp, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        if asrt_exprs is None:
            self._query_logger.error('Assertions were not specified in the "verify" mode: aborting...')
            return
        
        domain, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            algo, model, model_features_dict, feat_names, resp_names, 
            None, None, asrt_names, asrt_exprs, None, None, None, None, #delta, epsilon,
            alph_expr, None, None, data_scaler, scale_feat, scale_resp, None, 
            float_approx, float_precision, data_bounds_json_path)
        #print('eta', eta); print('alpha', alpha); print('beta',  beta)
        asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
        for i, form in enumerate(asrt_forms_dict.values()):
            if not isinstance(form, smlp.libsmlp.form2):
                raise Exception('Assertion ' + str(asrt_exprs[i]) + ' must be a formula (not a term)')
                
        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            asrt_res = self.verify_asrt(
                model_full_term_dict, 'consistency_check', 'False', smlp.false, 
                domain, alpha, beta, eta, solver_logic, float_approx, float_precision)
            if asrt_res['status'] == 'UNSAT':
                self._verify_logger.info('Model querying instance is inconsistent; aborting...')
                return
        
        self.verify_assertions(model_full_term_dict, asrt_names, asrt_exprs, asrt_forms_dict, domain, alpha, beta, eta, solver_logic, float_approx, float_precision)

