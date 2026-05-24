# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

from fractions import Fraction
import json

from smlp import core

from .smlp_terms import ModelTerms, SmlpTerms
from .smlp_utils import np_JSONEncoder

class SmlpVerify:
    def __init__(self):
        self._smlpTermsInst = SmlpTerms()
        self._modelTermsInst = None #ModelTerms()
        
        self._VACUITY_ASSERTION_NAME = 'consistency_check'
        self._DEF_ASSERTIONS_NAMES = None
        self._DEF_ASSERTIONS_EXPRS = None
        self.asrt_params_dict = {
            'assertions_names': {'abbr':'asrt_names', 'default':str(self._DEF_ASSERTIONS_NAMES), 'type':str,
                'help':'Names of optimization objectives [default {}]'.format(str(self._DEF_ASSERTIONS_NAMES))}, 
            'assertions_expressions':{'abbr':'asrt_exprs', 'default':self._DEF_ASSERTIONS_EXPRS, 'type':str,
                'help':'Semicolon seperated list of expressions (functions) to be applied to the responses '
                    'to convert them into optimization objectives ' +
                    '[default: {}]'.format(str(self._DEF_ASSERTIONS_EXPRS))}
        }
    
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

    def verify_asrt(self, model_full_term_dict:dict, asrt_name:str, asrt_expr:str, asrt_form:core.form2, 
            domain:core.domain, alpha:core.form2, beta:core.form2, eta:core.form2, solver_logic:str, sat_approx:bool, sat_precision:int):
        self._verify_logger.info('Verifying assertion {} <-> {}'.format(str(asrt_name), str(asrt_expr)))
        # TODO !!!: take care of usage of beta; currently we assume that if beta is required it is part of assertion
        assert beta == core.true

        solver_instance = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
            domain, model_full_term_dict, True, solver_logic)
        solver_instance.add(alpha)
        solver_instance.add(eta)
        solver_instance.add(self._smlpTermsInst.smlp_not(asrt_form))
        res = solver_instance.check()
        
        if self._modelTermsInst.solver_status_is_unsat(res): #isinstance(res, core.unsat):
            status = 'UNSAT' if asrt_name == self._VACUITY_ASSERTION_NAME else 'PASS'
            self._verify_logger.info('Completed with result: {}'.format(status)) #UNSAT 'PASS'
            asrt_res_dict = {'status':'PASS', 'asrt':None, 'model':None}
        elif self._modelTermsInst.solver_status_is_sat(res): #isinstance(res, core.sat):
            status = 'SAT' if asrt_name == self._VACUITY_ASSERTION_NAME else 'FAIL'
            self._verify_logger.info('Completed with result: {}'.format(status)) #SAT 'FAIL (SAT)'
            witness_vals_dict = self._smlpTermsInst.witness_term_to_const(self._modelTermsInst.get_solver_model(res),
                approximate=sat_approx, precision=sat_precision)
            # sanity check: the value of the negated assertion in the sat assignment should be true
            asrt_ce_val = eval(asrt_expr, {},  witness_vals_dict)
            assert not asrt_ce_val
            asrt_res_dict = {'status':'FAIL', 'asrt': asrt_ce_val, 'model':witness_vals_dict}
        elif self._modelTermsInst.solver_status_is_unknwn(res): #isinstance(res, core.unknown):
            self._verify_logger.info('Completed with result: {}'.format('UNKNOWN'))
            # TODO !!!: add reason for UNKNOWN or report that reason as 'status' field
            asrt_res_dict = {'status':'UNKNOWN', 'asrt':None, 'model':None}
        else:
            raise Exception('Unexpected resuld from solver')
        return asrt_res_dict
        
        
    def verify_assertions(self, model_full_term_dict:dict, asrt_names:list, asrt_exprs:list, asrt_forms_dict:dict, 
            domain:core.domain, alpha:core.form2, beta:core.form2, eta:core.form2, solver_logic:str, sat_approx=False, sat_precision=64):
        assert list(asrt_forms_dict.keys()) == asrt_names
        asrt_res_dict = {}
        for i, (asrt_name, asrt_form) in enumerate(asrt_forms_dict.items()):
            asrt_res_dict[asrt_name] = self.verify_asrt(model_full_term_dict, asrt_name, asrt_exprs[i], asrt_form, 
                domain, alpha, beta, eta, solver_logic, sat_approx, sat_precision)
        
        with open(self.assertions_results_file, 'w') as f: #json.dump(asrt_res_dict, f)
            json.dump(asrt_res_dict, f, indent='\t', cls=np_JSONEncoder) #cls= , use_decimal=True
            
    def smlp_verify(self, syst_expr_dict:dict, algo, model, model_features_dict, feat_names, resp_names, asrt_names, asrt_exprs,
            alph_expr:str, solver_logic:str, vacuity:bool, data_scaler, scale_feat, scale_resp, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        # sanity checking of knob values and assertions in the specification
        #sanity_check_verification_spec()
        if asrt_exprs is None:
            self._query_logger.error('Assertions were not specified in the "verify" mode: aborting...')
            return
        
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = self._modelTermsInst.create_model_exploration_base_components(
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            alph_expr, None, None, data_scaler, scale_feat, scale_resp,  
            float_approx, float_precision, data_bounds_json_path)
        '''
        certify_witness(self, universal:bool, model_full_term_dict:dict, quer_name:str, quer_expr:str, quer:core.form2, witn_dict:dict,
            domain:core.domain, eta:core.form2, alpha:core.form2, theta_radii_dict:dict, #beta:core.form2, 
            delta:float, solver_logic:str, witn:bool, sat_approx:bool, sat_precision:int)
        '''
        asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
        for i, form in enumerate(asrt_forms_dict.values()):
            if not isinstance(form, core.libsmlp.form2):
                raise Exception('Assertion ' + str(asrt_exprs[i]) + ' must be a formula (not a term)')
                
        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            asrt_res = self.verify_asrt(
                model_full_term_dict, self._VACUITY_ASSERTION_NAME, 'False', core.false, 
                domain, alpha, beta, eta, solver_logic, float_approx, float_precision)
            if asrt_res['status'] == 'PASS': #UNSAT
                self._verify_logger.info('Model querying instance is inconsistent; aborting...')
                return
        
        self.verify_assertions(model_full_term_dict, asrt_names, asrt_exprs, asrt_forms_dict, 
            domain, alpha, beta, eta, solver_logic, float_approx, float_precision)

