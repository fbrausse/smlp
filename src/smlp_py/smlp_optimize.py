# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import smlp
from smlp_py.smlp_terms import SmlpTerms, ModelTerms, ScalerTerms
from smlp_py.smlp_query import SmlpQuery
from smlp_py.smlp_utils import (str_to_bool, np_JSONEncoder)
            
from fractions import Fraction
from decimal import Decimal
from typing import Union
import json
import pandas as pd
#import keras
import numpy as np

# single or multi-objective optimization, with stability constraints and any user
# given constraints on free input, control (knob) and output variables satisfied.
class SmlpOptimize:
    def __init__(self):
        self._opt_logger = None 
        self._smlpTermsInst = SmlpTerms()
        self._modelTermsInst = None # ModelTerms()
        self._queryInst = None # SmlpQuery()
        self._scalerTermsInst = ScalerTerms()
        # TMP !!!!!! 
        # use value 
        self._tmp_return_eager_tuples = False 
        # TMP !!!!!! 
        # to reproduce don't care variable problem in test 202, use value False
        # to reproduce infinite loop im Bin-Opt-Max() in test 83 run with eager strategy, use value False
        self._tmp_use_model_in_phi_cand = False 
        
        # Save best so far (near optimal, stable) configurations of knobs as soon as
        # configurations are improved (keep the earlier updates), in two formats:
        # as a dict to be reported as json file and as df to be reported as csv file.
        self.best_config_dict = {}
        self.best_config_df = None
                
        # solver options
        #self._DEF_DELTA = 0.01 
        self._DEF_EPSILON = 0.05 
        #self._DEF_ALPHA = None
        #self._DEF_BETA = None
        #self._DEF_ETA = None 
        self._DEF_CENTER_OFFSET = '0'
        #self._DEF_BO_CEX = 'no'
        #self._DEF_BO_CAND = 'no'
        self._DEF_SCALE_OBJECTIVES = True
        self._OPTIMIZE_PARETO = True
        self._DEF_VACUITY_CHECK = True
        self._DEF_OBJECTIVES_NAMES = None
        self._DEF_OBJECTIVES_EXPRS = None
        self._DEF_APPROXIMATE_FRACTIONS:bool = True
        self._DEF_FRACTION_PRECISION:int = 64
        self._DEF_OPTIMIZATION_STRATEGY:str = 'lazy' # TODO !!! define enum type for lazy/eager and API functions to get the strategy value, so neither strings no enum types will be ued outside this file
        
        # Formulae alpha, beta, eta are used in single and pareto optimization tasks.
        # They are used to constrain control variables x and response variables y as follows:
        #
        # eta  : constraints on candidates; in addition to a eta-grid and eta-ranges; eta is actually eta-global
        # theta: stability region
        # alpha: constraint on whether a particular y is considered eligible as a counter-example
        # beta : additional constraint valid solutions have to satisfy; if not, y is a counter-example
        # delta: is a real constant, that is used to increase the radius around counter-examples
        #
        # epsilon: is a real constant used as modifier(?) for the thresholds T that allows
        #          to prove completeness of the algorithm
        #
        # optimize threshold T in obj_range such that (assuming direction is >=):
        #
        # Ex. eta x /\ Ay. theta x y -> alpha y -> (beta y /\ obj y >= T)
        # 
        # domain constraints eta from 'dom' have to hold for x and y.

        self.opt_params_dict = {
            'epsilon': {'abbr':'epsilon', 'default':self._DEF_EPSILON, 'type':float, 
                'help':'ratio of the length of an estimated range of an objective, '+ 
                        'computed per objective based on its estimated min and max bounds ' +
                        '[default: {}]'.format(str(self._DEF_EPSILON))},
            'center_offset': {'abbr':'center_offset', 'default':self._DEF_CENTER_OFFSET, 'type':str, 
                'help':'Center threshold offset of threshold ' +
                        '[default: {}]'.format(str(self._DEF_CENTER_OFFSET))},
            #'bo_cex': {'abbr':'bo_cex', 'default':self._DEF_BO_CEX, 'type':str, 
            #    'help':'use BO_CEX >= 10 iterations of BO to find counter-examples ' +
            #            '[default: {}]'.format(str(self._DEF_BO_CEX))},
            #'bo_cand': {'abbr':'bo_cand', 'default':self._DEF_BO_CAND, 'type':str, 
            #    'help':'use BO_CAD iterations of BO to find a candidate prior to falling back to Z3  ' +
            #            '[default: {}]'.format(str(self._DEF_BO_CAND))},
            'objectives_names': {'abbr':'objv_names', 'default':str(self._DEF_OBJECTIVES_NAMES), 'type':str,
                'help':'Names of optimization objectives [default {}]'.format(str(self._DEF_OBJECTIVES_NAMES))}, 
            'objectives_expressions':{'abbr':'objv_exprs', 'default':self._DEF_OBJECTIVES_EXPRS, 'type':str,
                'help':'Semicolon seperated list of expressions (functions) to be applied to the responses '
                    'to convert them into optimization objectives ' +
                    '[default: {}]'.format(str(self._DEF_OBJECTIVES_EXPRS))},
            'scale_objectives': {'abbr':'scale_objv', 'default': self._DEF_SCALE_OBJECTIVES, 'type':str_to_bool,
                'help': 'Should optimization objectives be scaled using scaler specified through ' + 
                    'option "data_scaler"? [default: ' + str(self._DEF_SCALE_OBJECTIVES) + ']'},
            'optimize_pareto': {'abbr':'pareto', 'default': self._OPTIMIZE_PARETO, 'type':str_to_bool,
                'help': 'Should optimization be per objective (even if there are multiple objectives) ' + 
                    'or pareto optimization must be performed? ' + 
                    '[default: ' + str(self._OPTIMIZE_PARETO) + ']'},
            'approximate_fractions': {'abbr':'frac_aprox', 'default': self._DEF_APPROXIMATE_FRACTIONS, 'type':str_to_bool,
                'help': 'Should fraction values form satisfying assignments be converted to approximate reals? ' + 
                    '[default: ' + str(self._DEF_APPROXIMATE_FRACTIONS) + ']'},
            'fraction_precision': {'abbr':'frac_prec', 'default':str(self._DEF_FRACTION_PRECISION), 'type':int,
                'help':'Decimal precision when approximating fractions by reals [default {}]'.format(str(self._DEF_FRACTION_PRECISION))},
            'vacuity_check': {'abbr':'vacuity', 'default': self._DEF_VACUITY_CHECK, 'type':str_to_bool,
                'help': 'Should solver problem instance vacuity check be performed? ' + 
                    'Vacuity checks whether the constraints are consistent and therefore at least ' +
                    'one satisfiable assignment exist to solver constraints. Relevant in "verify", "query", ' +
                    '"optimize" and "optsyn" modes [default: ' + str(self._DEF_VACUITY_CHECK) + ']'},
            'optimization_strategy': {'abbr':'opt_strategy', 'default':str(self._DEF_OPTIMIZATION_STRATEGY), 'type':str,
                'help':'Strategy (algorithm) to use for single objective optimization in the "optimize" and "optsyn" modes. ' +
                    'Supported options are "lazy" and "eager" [default {}]'.format(str(self._DEF_OPTIMIZATION_STRATEGY))},
        }
        
        # initialize the fields in the more status dictionary mode_status_dict as unknown/running
        self.mode_status_dict = {
            'smlp_execution': 'running', 
            'interface_consistent': 'unknown',
            'model_consistent': 'unknown',
            'synthesis_feasible': 'unknown'}
    
    def set_logger(self, logger):
        self._opt_logger = logger 
        self._queryInst.set_logger(logger)
        self._modelTermsInst.set_logger(logger)
        
    def set_tracer(self, tracer, trace_runtime, trace_prec, trace_anonym):
        self._opt_tracer = tracer
        self._trace_runtime = trace_runtime
        self._trace_precision = trace_prec
        self._trace_anonymize = trace_anonym
        self._queryInst.set_tracer(tracer, trace_runtime, trace_prec, trace_anonym)
        self._modelTermsInst.set_tracer(tracer, trace_runtime, trace_prec, trace_anonym)
        
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
        self._modelTermsInst.set_report_file_prefix(report_file_prefix)
        self._queryInst.set_report_file_prefix(report_file_prefix)
    
    # model_file_prefix is a string used as prefix in all saved model files of SMLP
    def set_model_file_prefix(self, model_file_prefix):
        self.model_file_prefix = model_file_prefix
        self._modelTermsInst.set_model_file_prefix(model_file_prefix)
        #self._queryInst.set_model_file_prefix(model_file_prefix)
    
    def set_spec_file(self, spec_file):
        self._modelTermsInst.set_spec_file(spec_file)
    
    # set self._modelTermsInst ModelTerms()
    def set_model_terms_inst(self, model_terms_inst):
        self._modelTermsInst = model_terms_inst
    
    def set_smlp_query_inst(self, smlp_query_inst):
        self._queryInst = smlp_query_inst
    
    # record vacuity and best achieved objectives' thresholds while pareto optimization
    # is still in progress
    @property
    def optimization_progress_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_optimization_progress'
    
    # final optimization results file
    @property
    def optimization_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_optimization_results'

    '''
    def print_result(self, res):
        if isinstance(res, smlp.sat):
            #print('SAT with model')
            for var, cnst in res.model.items():
                #print(' ', var, '=', smlp.Cnst(cnst))
        elif isinstance(res, smlp.unsat):
            #print('UNSAT')
        else:
            #print('unknown:', res.reason)
    '''
    # unscale constant const with respect to max and min values of feat_name
    # specified in data_bounds dictionary
    def scale_constant_val(self, data_bounds, feat_name, const):
        orig_max = data_bounds[feat_name]['max']
        orig_min = data_bounds[feat_name]['min']
        return (const - orig_min) / (orig_max - orig_min)
    
    # Unscale constant const with respect to max and min values of feat_name
    def unscale_constant_val(self, data_bounds, feat_name, const):
        orig_max = data_bounds[feat_name]['max']
        orig_min = data_bounds[feat_name]['min']
        return const * (orig_max - orig_min) + orig_min
    
    # Unscale constant const which is intended to be relative (say percentage, ratio) 
    # with respect to max and min values of feat_name. Current usage is for constant
    # epsilon, which defines how close, in terms of percentage / ration. an approximate 
    # optimum value must be the to real optimum for the optimization procedure to halt.
    def unscale_relative_constant_val(self, data_bounds, feat_name, const):
        orig_max = data_bounds[feat_name]['max']
        orig_min = data_bounds[feat_name]['min']
        return const * (orig_max - orig_min)
    
    # compute bounds on the objectives, required for scaling objectives
    def compute_objectives_bounds(self, X:pd.DataFrame, y:pd.DataFrame, objv_names:list[str], objv_exprs:list[str]):
        df_resp_feat = pd.concat([X,y], axis=1); #print('df_resp_feat\n', df_resp_feat)
        objv_bounds_dict = {}

        # Function to evaluate objectives' expressions objv_cond on each row of training data.
        # It computes the environment to be used by eval() function to assign values to the
        # leaves of the objectives expressions and then propagate them to compute the value of
        # each objective on training data. It is important to make sure that X and y parts of
        # the training data (the features and the responses) are in the original scale (are not
        # scaled to say [0,1] for improving training performance, using say the min-max scaler).
        def eval_objv(row):
            #print('inp_row\n', row, type(row))
            eval_env = {}
            for col in df_resp_feat.columns.tolist():
                eval_env[col] = row[col]
            #print('eval_env', eval_env)
            res_row = eval(objv_cond, {}, eval_env); #print('res_row', res_row, type(res_row))
            return res_row

        for i, (objv_name, objv_cond) in enumerate(zip(objv_names, objv_exprs)):
            #print('objv_cond', objv_cond, type(objv_cond))
            #print('y', y.columns.tolist(), '\n', y) 
            objv_series = df_resp_feat.apply(lambda row : eval_objv(row), axis=1); #print('objv_series', objv_series.tolist())
            objv_bounds_dict[objv_name] = {'min': float(objv_series.min()), 'max': float(objv_series.max())}
        
        for o, b in objv_bounds_dict.items():
            if b['min'] == b['max']:
                raise Exception('Objective ' + str(o) + ' is constant ' + str(b['min']) + ' on training set')
        return objv_bounds_dict

    # Otimization procedure for eager single objective optimization optimize_single_objective_eager().
    # It does not deal with stability of solutions and also it is assumed that there are no (free) inputs.
    # y_cand is returned only for reprting optimization progress -- it holds values of the reeponses in 
    # the candidate knob configuration p_cand
    def bin_opt_max(self, smlp_domain:smlp.domain, model_full_term_dict:dict, phi_cond:smlp.form2, objv_name:str, objv_expr:str, 
            objv_term:smlp.term2, l0:float, u0:float, epsilon:float, solver_logic:str):
        self._opt_tracer.info('bin_opt_max start, objective_thresholds_u0_l0, {} : {}'.format(str(u0),str(l0)))
        assert l0 < u0
        assert l0 not in [-np.inf, np.inf] 
        assert u0 not in [-np.inf, np.inf]
        
        l = (-np.inf) # lower bound 
        u = np.inf # upper bound 
        p_cand = None # initializing, assertion ensures it gets assigned
        while True:
            self._opt_tracer.info('bin_opt_max iter, objective_thresholds_u0_l0_u_l, {} : {} : {} : {}'.format(str(u0),str(l0),str(u),str(l)))
            #print('top of while loop: l0', l0, 'u0', u0, 'l', l, 'u', u)
            if u == np.inf:
                (T, u0) = (u0, 2*u0 - l0)
            elif l == -np.inf:
                (T, l0) = (l0, 2*l0 - u0)
            else:
                T = (l + u) / 2
                
            # inviriants of the algprthm -- sanity checks
            assert l0 < u0
            assert l < u
            assert l0 not in [-np.inf, np.inf] 
            assert u0 not in [-np.inf, np.inf]
            assert T not in [-np.inf, np.inf]
            
            quer_form = self._smlpTermsInst.smlp_ge(objv_term, smlp.Cnst(T)); #print('quer_form', quer_form)
            quer_expr = '{} >= {}'.format(objv_expr, str(T)) if objv_expr is not None else None
            quer_name = objv_name + '_' + str(T)
            #quer_and_beta = self._smlpTermsInst.smlp_and(quer_form, beta) if not beta == smlp.true else quer_form
            #print('quer_and_beta', quer_and_beta) 'u0_l0_u_l_T'
            self._opt_tracer.info('adjusted thresholds, objective_thresholds_u0_l0_u_l_T, {} : {} : {} : {} : {}'.format(str(u0),str(l0),str(u),str(l),str(T)))
            max_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
                smlp_domain, model_full_term_dict, True, solver_logic)
            max_solver.add(phi_cond) 
            #print('objv_expr', objv_expr, 'objv_term', objv_term, flush=True); 
            max_solver.add(quer_form)    
            #print('solving threshold', T, flush=True)
            #res = self._queryInst.find_candidate(max_solver); print('res', res)
            res = self._queryInst._modelTermsInst.smlp_solver_check(max_solver, 'max_ca', self._queryInst._lemma_precision); #print('res', res, flush=True)
            assert res is not None
            assert not self._modelTermsInst.solver_status_unknown(res)
            if self._modelTermsInst.solver_status_unsat(res):
                self._opt_tracer.info('bin_opt_max iter, unsat')
                u = T
            else:
                self._opt_tracer.info('bin_opt_max iter, sat')
                l = T
                p_cand = self._modelTermsInst.get_solver_knobs_model(res); #print('p_cand', p_cand, flush=True)
                y_cand = self._modelTermsInst.get_solver_resps_model(res);
            assert l < u
            if l + epsilon > u:
                break
        #self._opt_tracer.info('bin_opt_max end, p_cand_l_u, {} : {} : {}'.format(str(p_cand), str(l),str(u)))
        if self._trace_precision > 0:
            p_cand_trace = self._modelTermsInst.witness_term_to_const(p_cand, approximate=True, precision=self._trace_precision)
        else:
            p_cand_trace = self._modelTermsInst.witness_term_to_const(p_cand, approximate=False, precision=None)
        self._opt_tracer.info('bin_opt_max end, p_cand_l_u, {} : {} : {}'.format(str(p_cand_trace), str(l),str(u)))
        assert p_cand is not None
        assert l < u
        return p_cand, y_cand, l, u   
    
    # internal optimization procedure for eager single objective optimization (finding min reduced to finding max)
    def bin_opt_min(self, smlp_domain:smlp.domain, model_full_term_dict:dict, phi_cond:smlp.form2, objv_name:str, objv_expr:str, 
            objv_term:smlp.term2, l0:float, u0:float, epsilon:float, solver_logic:str):
        self._opt_tracer.info('bin_opt_min start, objective_thresholds_u0_l0, {} : {}'.format(str(u0),str(l0)))
        assert l0 < u0
        assert l0 not in [-np.inf, np.inf] 
        assert u0 not in [-np.inf, np.inf]
        p_cand, y_cand, l_prime, u_prime = self.bin_opt_max(smlp_domain, model_full_term_dict, phi_cond, '_negated_'+objv_name, None if objv_expr is None else '-('+objv_expr+')', 
            self._smlpTermsInst.smlp_neg(objv_term), (-u0), (-l0), epsilon, solver_logic)
        self._opt_tracer.info('bin_opt_min end, p_cand_l_u, {} : {} : {}'.format(str(p_cand), str((-u_prime)),str((-l_prime))))
        assert (-u_prime) < (-l_prime)
        return p_cand, y_cand, (-u_prime), (-l_prime)
    
    
    # an improved / eager optimization procedure
    def optimize_single_objective_eager(self, model_full_term_dict:dict, objv_name:str, objv_expr:str, objv_term:smlp.term2, 
            epsilon:float, smlp_domain:smlp.domain, eta:smlp.form2, theta_radii_dict:dict, alpha:smlp.form2, beta:smlp.form2, delta:float, solver_logic:str, 
            scale_objectives:bool, orig_objv_name:str, objv_bounds:dict, call_info=None, sat_approx=False, sat_precision=64, save_trace=False,
            l0=None, u0=None): #, l=(-np.inf), u=np.inf): #TODO !!! l, u are not needed???

        self._opt_logger.info('Optimize single objective with eager strategy ' + str(objv_name) + ': Start')
        
        # initial lower bound l0 and initial upper bound u0
        if u0 is None and l0 is None:
            if scale_objectives or objv_bounds is None:
                l0 = 0 
                u0 = 1 
            else:
                l0 = objv_bounds[orig_objv_name]['min']
                u0 = objv_bounds[orig_objv_name]['max']
            #print('l0', l0, 'u0', u0)
            assert l0 < u0

        self._opt_tracer.info('single_objective_u0_l0, {} : {}'.format(str(u0),str(l0)))
        #print('eta', eta); print('alpha', alpha); print('beta', beta); print('objv_term', objv_term)
        phi_cand = self._smlpTermsInst.smlp_and_multi([eta, alpha, beta]); #print('phi_cand', phi_cand)
        
        P = [] # known candidates and bounds -- deviates from the pseudo-algorithm for yje purpose pf enhanced reporting
        P_eager = [] # for sanity check only -- it follows the pseido-algorithm
        L = self._smlpTermsInst.smlp_true # lemmas
        u_cand = np.inf # upper bound for candidate # TODO !!! use argument u or delete this argument
        l_stab = (-np.inf) # lower bound for stability # TODO !!! use argument l or delete this argument
        u_e = u0 # upper bound estimat
        l_e = l0 # ower bound estimate
        assert l_e < u_e
        iter_count = 0
        while True:
            self._opt_tracer.info('eager opt iter, objective_thresholds_ue_le_ucand_lstab, {} : {} : {} : {}'.format(str(u_e),str(l_e),str(u_cand),str(l_stab)))
            if l_stab + epsilon > u_cand:
                self._opt_tracer.info('eager opt end, optimization converged with required accuracy')
                break
            phi_cond = self._smlpTermsInst.smlp_and(phi_cand, L); #print('phi_cond', phi_oand)
            # TODO !!!!!!! temp workaround to use model_full_term_dict instead of None !!!!!!!!!!
            if self._tmp_use_model_in_phi_cand:
                candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
                    smlp_domain, model_full_term_dict, True, solver_logic)
            else:
                candidate_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
                    smlp_domain, None, True, solver_logic)
            candidate_solver.add(phi_cond)
            #print('solving phi_cond', flush=True)
            res = self._queryInst._modelTermsInst.smlp_solver_check(candidate_solver, 'ca_eager', self._queryInst._lemma_precision)
            assert res is not None
            assert not self._modelTermsInst.solver_status_unknown(res)
            if self._modelTermsInst.solver_status_unsat(res):
                self._opt_tracer.info('eager opt end, optimization result cannot be improved')
                break
            p_cand, y_cand, l_cand, u_cand = self.bin_opt_max(smlp_domain, model_full_term_dict, phi_cond, objv_name, objv_expr, objv_term, l_e, u_e, epsilon, solver_logic)
            u_e = u_cand
            l_e = l_cand # TODO !!!!!! verify this line of code, inserted to fix algorithm
            assert l_e < u_e
            
            # Here we do not use delta (use delta = None) because theta_form is not used in lemma generation
            theta_p_cand = self._modelTermsInst.compute_stability_formula_theta(p_cand, None, theta_radii_dict, True)
            #print('theta_p_cand', theta_p_cand); print('l_stab', l_stab); print('objv_term', objv_term, flush=True)
            # creating formula objv_term <= l_stab
            if l_stab == (-np.inf):
                low_thresh_form = self._smlpTermsInst.smlp_false
            else:
                low_thresh_form = self._smlpTermsInst.smlp_le(objv_term, self._smlpTermsInst.smlp_cnst(l_stab)); 
            #print('low_thresh_form', low_thresh_form, flush=True)
            beta_implies_thresh_form = self._smlpTermsInst.smlp_implies(beta, low_thresh_form); 
            #print('beta_implies_thresh_form', beta_implies_thresh_form, flush=True)
            #phi_cex_cand = self._smlpTermsInst.smlp_and_multi([theta_p_cand, alpha, beta_implies_thresh_form])
            #print('phi_cex_cand', phi_cex_cand)
            cex_cand_solver = self._modelTermsInst.create_model_exploration_instance_from_smlp_components(
                smlp_domain, model_full_term_dict, True, solver_logic)
            # adding the conjuncts of formula phi_cex_cand to cex_cand_solver:
            # phi_cex_cand = self._smlpTermsInst.smlp_and_multi([theta_p_cand, alpha, beta_implies_thresh_form])
            cex_cand_solver.add(alpha)
            cex_cand_solver.add(theta_p_cand)
            cex_cand_solver.add(beta_implies_thresh_form)
            # TODO !!!!!! give a proper name to solver problem below
            #print('solving phi_cex_cand', flush=True)
            res = self._modelTermsInst.smlp_solver_check(cex_cand_solver, 'phi_cex_cand', self._queryInst._lemma_precision)
            assert not self._modelTermsInst.solver_status_unknown(res)
            if self._modelTermsInst.solver_status_sat(res):
                p_cex, y_cex = (self._modelTermsInst.get_solver_knobs_model(res), self._modelTermsInst.get_solver_resps_model(res))
                #print('SAT:', 'p_cex', p_cex, 'y_cex', y_cex, flush=True)
            else:
                #print('UNSAT', flush=True)
                phi_cex = self._smlpTermsInst.smlp_and(theta_p_cand, alpha)
                #print('theta_p_cand', theta_p_cand, 'phi_cex', phi_cex, flush=True)
                l_stab_prev = l_stab # remember l_stab before it gets updated -- just fpr reporting
                p_cex, y_cex, l_stab, u_cex = self.bin_opt_min(smlp_domain, model_full_term_dict, phi_cex, objv_name, objv_expr, objv_term, l_e, u_e, epsilon, solver_logic)
                #print('p_cex', p_cex, 'l_stab', l_stab, 'u_cex', u_cex, flush=True)
                
                self._opt_logger.info('Increasing threshold lower bound for objective ' + str(objv_name) + ' from ' + str(l_stab_prev) + ' to ' + str(l_stab))
                #if objv_expr is not None:
                stable_witness_terms = p_cex | y_cex; #print('stable_witness_terms', stable_witness_terms)
                objv_witn_val_term = smlp.subst(objv_term, stable_witness_terms); #print('objv_witn_val_term', objv_witn_val_term)
                update_progress_report = l_stab_prev != -np.inf
                stable_witness_vals = self.update_optimization_reports(stable_witness_terms, l_stab, u_e, call_info, iter_count, 
                    scale_objectives, objv_name, objv_witn_val_term, objv_bounds, objv_expr, orig_objv_name, update_progress_report, sat_approx, sat_precision)
                #print('stable_witness_vals', stable_witness_vals)
                l_e = l_stab; #print('l_e', l_e, flush=True)
                assert l_e < u_e
                P.append(stable_witness_vals)
                P_eager.append((p_cand, l_cand, u_cand, l_stab))
            
            theta_p_cex_delta = self._modelTermsInst.compute_stability_formula_theta(p_cex, delta, theta_radii_dict, True)
            L = self._smlpTermsInst.smlp_and(L, self._smlpTermsInst.smlp_not(theta_p_cex_delta))
            iter_count +=  + 1
        
        # compute and return max value of l_stab within P
        # TODO !!!! what if P is empty?
        assert len(P) > 0
        #print('P', P)
        l_res = max([tup[3] for tup in P_eager]); #print('l_res', l_res, 'threshold_lo', P[-1]['threshold_lo'])
        assert l_res == P[-1]['threshold_lo_scaled']
        #print('optimize_single_objective_eager and with l_res', l_res, flush=True)
        self._opt_logger.info('Optimize single objective with eager strategy ' + str(objv_name) + ': end')
        if self._tmp_return_eager_tuples:
            return l_res
        else:
            return P[-1] # l_res
    
    # Optimization for single objective.
    # assuming in first implementation that objectives are scaled to [0,1] -- not using
    # objv_bounds, data_scaler, objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict, 
    # also not using thresholds_dict -- covering a general case
    # Arguments l0 and u0 arbitrary candidate lower and upper bounds, say one's best guess.
    # Arguments l and u are known/already proven lower and upper bounds; initialized to: -inf and inf. 
    def optimize_single_objective(self, model_full_term_dict:dict, objv_name:str, objv_expr:str, objv_term:smlp.term2, 
            epsilon:float, smlp_domain:smlp.domain, eta:smlp.form2, theta_radii_dict:dict, alpha:smlp.form2, beta:smlp.form2, delta:float, solver_logic:str, 
            scale_objectives:bool, orig_objv_name:str, objv_bounds:dict, call_info=None, sat_approx=False, sat_precision=64, save_trace=False,
            l0=None, u0=None, l=(-np.inf), u=np.inf):
        self._opt_logger.info('Optimize single objective ' + str(objv_name) + ': Start')
        
        # initial lower bound l0 and initial upper bound u0
        if u0 is None and l0 is None:
            if scale_objectives or objv_bounds is None:
                l0 = 0 
                u0 = 1 
            else:
                l0 = objv_bounds[orig_objv_name]['min']
                u0 = objv_bounds[orig_objv_name]['max']
            #print('l0', l0, 'u0', u0)
            assert l0 < u0

        self._opt_tracer.info('single_objective_u0_l0_u_l, {} : {} : {} : {} : {}'.format(str(objv_name),str(u0),str(l0),str(u),str(l)))
        #print('l0', l0, 'u0', u0, 'l', l, 'u', u)
        #TODO !!!: we assume objectives were scaled to [0,1] and l0 and u0 are initialized to 0 and 1 respectively
        #assert scale_objectives 
        P = [] # known candidates and lower bounds
        N = [] # known counter-examples and upper bounds
        
        iter_count = 0
        while True:
            #print('top of while loop: l0', l0, 'u0', u0, 'l', l, 'u', u)
            if u == np.inf:
                (T, u0) = (u0, 2*u0 - l0)
            elif l == -np.inf:
                (T, l0) = (l0, 2*l0 - u0)
            else:
                T = (l + u) / 2
            #quer_form = objv_term > smlp.Cnst(T)
            #quer_form = objv_term >= smlp.Cnst(T) 
            # TODO !!!! use the following, avoid usage of >=
            quer_form = self._smlpTermsInst.smlp_ge(objv_term, smlp.Cnst(T));
            quer_expr = '{} >= {}'.format(objv_expr, str(T)) if objv_expr is not None else None
            quer_name = objv_name + '_' + str(T)
            quer_and_beta = self._smlpTermsInst.smlp_and(quer_form, beta) if not beta == smlp.true else quer_form
            #print('quer_and_beta', quer_and_beta) 'u0_l0_u_l_T'
            self._opt_tracer.info('objective_thresholds_u0_l0_u_l_T, {} : {} : {} : {} : {}'.format(str(u0),str(l0),str(u),str(l),str(T)))
            quer_res = self._queryInst.query_condition(
                True, model_full_term_dict, quer_name, quer_expr, quer_and_beta, smlp_domain,
                eta, alpha, theta_radii_dict, delta, solver_logic, False, sat_approx, sat_precision)
            stable_witness_status = quer_res['query_status']
            stable_witness_terms = quer_res['witness']
            if stable_witness_status == 'UNSAT':
                assert T <= u
                self._opt_logger.info('Decreasing threshold upper bound for objective ' + str(objv_name) + ' from ' + str(u) + ' to ' + str(T))
                u = T
                #print('objv_bounds', objv_bounds)
                # only the last value in P is used, and we want it to contain at least one element even if lower bound
                # is not improved within this function -- that is, stable_witness_status was never 'STABLE_SAT'. 
                # For that reason we update P also in case stable_witness_status == 'UNSAT'. Unlike the case when
                # stable_witness_status == 'STABLE_SAT', stable_witness_terms here will not include values of objectives
                # or scaled objectives because these values are taken based on stable witness (stable candidate) which
                # we don't have in this case. This fact (that stable_witness_terms does not have items with objectives
                # or scaled objectives names) is used when this function is called from active_objectives_max_min_bounds():
                # the latter function will extract values (scaled_)threshold_lo/up and replaces the rest by None, which 
                # info is used to not call the reporting function report_current_thresholds() on the result of 
                # active_objectives_max_min_bounds() (because the proven lower bound has not improved).
                if len(P) == 0:
                    stable_witness_terms = {}
                    '''
                    objectives_unscaler_terms_dict = self._scalerTermsInst.feature_unscaler_terms(objv_bounds, [orig_objv_name])
                    # substitute scaled objective variables with scaled objective terms
                    # in original objective terms within objectives_unscaler_terms_dict
                    if objv_expr is not None:
                        orig_objv_const_term = smlp.subst(objectives_unscaler_terms_dict[orig_objv_name], #objv_term, 
                            {self._scalerTermsInst._scaled_name(orig_objv_name): objv_witn_val_term})
                        #print('orig_objv_const_term', orig_objv_const_term)
                        objv_name_unscaled = self._scalerTermsInst._unscaled_name(objv_name)
                        if objv_name_unscaled in self.objv_names:
                            stable_witness_terms[objv_name_unscaled] = orig_objv_const_term 
                    '''
                    if l not in [np.inf, -np.inf]:
                        unscaled_threshold_lo = self._scalerTermsInst.unscale_constant_term(objv_bounds, orig_objv_name, l)
                        #print('unscaled_threshold_lo: l', l, 'unsc', unscaled_threshold_lo)
                        stable_witness_terms['threshold_lo_scaled'] = smlp.Cnst(l)
                        stable_witness_terms['threshold_lo'] = unscaled_threshold_lo
                    if u not in [np.inf, -np.inf]:
                        unscaled_threshold_up = self._scalerTermsInst.unscale_constant_term(objv_bounds, orig_objv_name, u)
                        #print('unscaled_threshold_up: l', l, 'unsc', unscaled_threshold_up)
                        stable_witness_terms['threshold_up_scaled'] = smlp.Cnst(u)
                        stable_witness_terms['threshold_up'] = unscaled_threshold_up
                    stable_witness_vals = self._smlpTermsInst.witness_term_to_const(
                        stable_witness_terms, sat_approx, sat_precision)
                    P.append(stable_witness_vals)
            elif stable_witness_status == 'STABLE_SAT':
                #self._opt_logger.info('Increasing threshold lower bound for objective ' + str(objv_name) + ' from ' + str(l) + ' to ' + str(T))
                update_progress_report = False
                if l != -np.inf:
                    update_progress_report = True
                #print('objv_term', objv_term, flush=True); print('stable_witness_terms', stable_witness_terms, flush=True)
                l_prev = l # save the value of l, it is for reporting only.
                #if objv_expr is not None: # the objective is not a symbolic max_min term, we may need its value, at least to see search progress
                objv_witn_val_term = smlp.subst(objv_term, stable_witness_terms); #print('objv_witn_val_term', objv_witn_val_term)
                #using objective values as lower bounds is not sound since objective value in sat model is the ceneter-point value 
                # and the objective's value is not guaranteed to be a lower bound in entire stability region
                #objv_witn_val = self._smlpTermsInst.ground_smlp_expr_to_value(objv_witn_val_term, sat_approx, sat_precision)
                #assert objv_witn_val >= T
                #l = objv_witn_val
                l = T
                self._opt_logger.info('Increasing threshold lower bound for objective ' + str(objv_name) + ' from ' + str(l_prev) + ' to ' + str(l))
                #if objv_expr is not None:
                
                stable_witness_vals = self.update_optimization_reports(stable_witness_terms, l, u, call_info, iter_count, scale_objectives, objv_name, objv_witn_val_term, objv_bounds, objv_expr, orig_objv_name, update_progress_report, sat_approx, sat_precision)
                #print('adding to P', (stable_witness_vals, stable_witness_vals[objv_name]))
                #P.append((stable_witness_vals, stable_witness_vals[objv_name]))
                #if save_trace or l + epsilon > u:
                # Enhancement !!! we only use the last element of P -- could override existing last element instead of inserting.
                # Inserting is required for profiling which will be implemented soon.
                P.append(stable_witness_vals)
            else:
                raise Exception('Unsupported value ' + str(stable_witness_status) + ' received from query_conditions')
            
            if l + epsilon > u:
                # false when if l = -np.inf and u = np.inf
                #print('l', l, 'u', u, 'epsilon', epsilon, 'exit while true')
                break
            iter_count +=  + 1
        
        # make sure correct upper bound u is recorded in P{-1]. This is not used, it is just useful info for 
        # banchamrk statistics (this info is more precise when the while loop exit condition u < l + epsilon).
        if u not in [np.inf, -np.inf]:
            orig_max = objv_bounds[orig_objv_name]['max']; #print('orig_max', orig_max)
            orig_min = objv_bounds[orig_objv_name]['min']; #print('orig_min', orig_min)
            if scale_objectives:
                P[-1]['threshold_up_scaled'] = u; #print('u', u)
                P[-1]['threshold_up'] = orig_min + u * (orig_max - orig_min) ; #print( P[-1]['threshold_up'])
            else:
                P[-1]['threshold_up'] = u; #print('u', u)
            
        #print('P[-1]', P[-1])
        
        self._opt_logger.info('Optimize single objective ' + str(objv_name) + ': End')
        return P[-1]
    
    # optimize multiple objectives but each one separately -- the objectives might have optimal value
    # in different sub-spaces in the configuration space. This function essentially simply iterates in
    # the objectives and finds a stable optimum for each one using function optimize_single_objective().
    # TODO !!! strategy "eager" is not supported yet
    def optimize_single_objectives(self, feat_names:list, resp_names:list, #X:pd.DataFrame, y:pd.DataFrame, 
            model_full_term_dict:dict, objv_names:list, objv_exprs:list, objv_bounds_dict:dict, alpha:smlp.form2, beta:smlp.form2, 
            eta:smlp.form2, theta_radii_dict:dict, epsilon:float, smlp_domain:smlp.domain, delta:float, solver_logic:str, strategy:str,  scale_objv:bool,  
            data_scaler:str, sat_approx=False, sat_precision=64, save_trace=False):
        #assert X is not None
        #assert y is not None
        assert epsilon > 0 and epsilon < 1
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs)
        scale_objectives = scale_objv and data_scaler != 'none'
        #assert scale_objectives
        objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict = \
            self._modelTermsInst.compute_objectives_terms(objv_names, objv_exprs, objv_bounds_dict, scale_objectives)
        
        # TODO: set sat_approx to False once dump and load with Fractions will work
        opt_conf = {}
        for i, (objv_name, objv_term) in enumerate(list(objv_terms_dict.items())):
            objv_expr = objv_exprs[i]
            if scale_objectives:
                objv_epsn = epsilon
            else:
                objv_epsn = self.unscale_relative_constant_val(objv_bounds_dict, objv_names[i], epsilon)
            #print('objv_epsn', objv_epsn)
            if strategy == 'lazy':
                opt_conf[objv_names[i]] = self.optimize_single_objective(model_full_term_dict, objv_name, objv_expr, 
                    objv_term, objv_epsn, smlp_domain, eta, theta_radii_dict, alpha, beta, delta, solver_logic, scale_objectives, objv_names[i], 
                    objv_bounds_dict, None, sat_approx=True, sat_precision=64, save_trace=False); #print('opt_conf', opt_conf)
            elif strategy == 'eager':
                if self._tmp_return_eager_tuples:
                    assert False
                else:
                    opt_conf[objv_names[i]] = self.optimize_single_objective_eager(model_full_term_dict, objv_name, objv_expr, 
                        objv_term, objv_epsn, smlp_domain, eta, theta_radii_dict, alpha, beta, delta, solver_logic, scale_objectives, objv_names[i], 
                        objv_bounds_dict, None, sat_approx=True, sat_precision=64, save_trace=False);
            else:
                raise Exception('Unsupported optimization strategy ' + str(strategy))
                
        self.mode_status_dict['smlp_execution'] = 'completed'
        with open(self.optimization_results_file+'.json', 'w') as f:
            json.dump(opt_conf | self.mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
    
    
    # direction hi/up means we want to maximize the objectives, direction lo/dn means we want to minimize.
    # Compute smlp expression that represents the maximum amongs the minimums of the objectives that still do 
    # not have their greatest lower bound computed as part of pareto optimization algorithm, under the input
    # constraint eta and the constraints that represent already determined greatest lower bounds for the remaining
    # objectives. This max-min expression on the objectives will be passed to the single objective otimization
    # algorithm to determaine what is the best greatest lower bound that we can find and fix for these objectives.
    def active_objectives_max_min_bounds(self, model_full_term_dict:dict, objv_terms_dict:dict, t:list[float], 
            smlp_domain:smlp.domain, alpha:smlp.form2, beta:smlp.form2, eta:smlp.form2, theta_radii_dict, epsilon:float, 
            delta:float, solver_logic:str, strategy:str, direction, scale_objectives, objv_bounds, update_thresholds_dict, 
            sat_approx:bool, sat_precision:int, save_trace:bool):
        assert direction == 'up'
        eta_F_t = eta
        min_objs = None
        min_name = ''
        #print('thresholds t', t, 'objv_terms_dict', objv_terms_dict)
        for j, (objv_name, objv_term) in enumerate(objv_terms_dict.items()):
            if t[j] is not None:
                eta_F_t = self._smlpTermsInst.smlp_and(eta_F_t, objv_term > smlp.Cnst(t[j]))
            else:
                min_name = min_name + '_' + objv_name if min_name != '' else objv_name
                if min_objs is not None:
                    min_objs = smlp.Ite(objv_term < min_objs, objv_term, min_objs)
                else:
                    min_objs = objv_term
        
        # When active_objectives_max_min_bounds() is called for the first time from 
        # optimize_pareto_objectives(), the list t which represents the proven lower 
        # bounds of objectives, is composed of None's, and the proven lower
        # bound for all objectives is lower bound of empty set of reals, which is np.inf.
        # Otherwise this function tries to increase lower bounds of active objectives in
        # t (they have value None in s), and the highest already proven lower bound is
        # max(t_vals)+epsilon (this bound was proven in the inner for loop of the pareto
        # algorithm optimize_pareto_objectives() in the while loop iteration prior to this
        # call to active_objectives_max_min_bounds()); thus the expression for subset_threshold
        # below which defines/computes the proven lower bound for the active objectives in t.
        t_vals = [i for i in t if i is not None]
        #subset_threshold = max(t_vals) if len(t_vals) > 0 else np.inf
        subset_threshold = max(t_vals)+epsilon if len(t_vals) > 0 else np.inf
        #min_name = min_name + '_' + str(subset_threshold)
        #min_name = 'objectives_subset'
        if min_objs is None:
            return (np.inf, np.inf) if direction == 'up' else (-np.inf, -np.inf)
        # TODO !!! -- take care of scaling
        objv_bounds = {min_name: {'min':0, 'max' :1}}
        #print('objv_bounds', objv_bounds)
        
        #l0 = subset_threshold if len(t_vals) > 0 else 0
        # This function is called from optimize_pareto_objectives() on list of thresholds t. 
        # When t has at least not-None value, and hence len(t_vals) > 0 holds, it was already 
        # proven that threshold value subset_threshold is feasible for the active objectives
        # (under he assumption that the fixed objectoves have the value they were fixed to).
        # Therefore in this case paramer l (which is the proven lower bound for binary search in
        # optimize_single_objective() should be set to subset_threshold, otherwise l is set to -inf.
        if len(t_vals) > 0:
            l0 = subset_threshold; l = l0
        else:
            l0 = 0; l = -np.inf
        u0 = 1; u = np.inf
        '''
        if len(t_vals) > 0:
            objv_bounds = {min_name: {'min':subset_threshold, 'max' :1}}
        else:
            objv_bounds = {min_name: {'min':0, 'max' :1}}
        '''
        update_thresholds_dict['objv_thresholds'] = t
        #print('t', t, 't_vals', t)
        #print('update_thresholds_dict', update_thresholds_dict)
        if strategy == 'lazy':
            r = self.optimize_single_objective(model_full_term_dict, min_name, None, min_objs, 
                epsilon, smlp_domain, eta_F_t, theta_radii_dict, alpha, beta, delta, solver_logic,
                scale_objectives, min_name, objv_bounds, update_thresholds_dict, 
                sat_approx, sat_precision, save_trace, l0, u0, l, u)
        elif strategy == 'eager':
            if self._tmp_return_eager_tuples:
                r = {}
                r['threshold_lo'] = self.optimize_single_objective_eager(model_full_term_dict, min_name, None, min_objs, 
                    epsilon, smlp_domain, eta_F_t, theta_radii_dict, alpha, beta, delta, solver_logic,
                    scale_objectives, min_name, objv_bounds, update_thresholds_dict, 
                    sat_approx, sat_precision, save_trace, l0, u0) #, l, u)
            else:
                r = self.optimize_single_objective_eager(model_full_term_dict, min_name, None, min_objs, 
                    epsilon, smlp_domain, eta_F_t, theta_radii_dict, alpha, beta, delta, solver_logic,
                    scale_objectives, min_name, objv_bounds, update_thresholds_dict, 
                    sat_approx, sat_precision, save_trace, l0, u0)
        else:
            raise Exception('Unsupported optimization strategy ' + str(strategy))
        
        c_up = r['threshold_up'] if 'threshold_up' in r else np.inf; #print('c_up', c_up)
        c_lo = r['threshold_lo']; #print('c_lo', c_lo)
        assert c_lo != np.inf
        
        # the known lower bound has not been updated --optimize_single_objective() couldn't 
        # find a stable lower bound for the single objective (optimization in this function
        # means maximizing the objective); the known upper bound could have been tightened,
        # this info is recorded and returned as c_up.
        if min_name not in r:
            r = None
        return c_lo, c_up, r
    
    # Convert objectives best-so-far thresholds info from dictionary/json to table/csv format.
    # This function is used report_current_thresholds() to update user 
    def prog_dict_to_df(self):
        prog_dict = self.best_config_dict; #print('prog_dict', prog_dict)
        row_names = []
        new_prog_dict = {}
        prog_df = None
        for k,v in prog_dict.items():
            #print('k', k)
            row_names.append(k)
            v_new_dict = {}
            for vi, vv in v.items():
                #print('vi', vi, 'vv', vv)
                if 'threshold' in vi or vi == 'max_in_data' or vi == 'min_in_data':
                    continue
                v_new_dict[vi] = vv['value_in_config']
            v_new_df = pd.DataFrame([v_new_dict])  # v_new_dict.items()
            feat_df = v_new_df[sorted([col for col in v_new_df.columns if col in self.feat_names])]
            resp_df = v_new_df[sorted([col for col in v_new_df.columns if col in self.resp_names])]
            objv_df = v_new_df[sorted([col for col in v_new_df.columns if col in self.objv_names])]
            v_new_df = pd.concat([feat_df, resp_df, objv_df], axis=1)
            if prog_df is None:
                prog_df = v_new_df
            else:
                #print(prog_df.columns.tolist()); print(v_new_df.columns.tolist())
                assert prog_df.columns.tolist() == v_new_df.columns.tolist()
                prog_df = pd.concat([prog_df, v_new_df], axis=0, ignore_index = True)
        prog_df.insert(loc=0, column='Iteration', value=row_names)
        #print('prog_df\n', prog_df)
        return prog_df          

    # This function is called every time the thresholds of the objectives are improved during pareto or single
    # objective optimization procedure. The updated thresholds seen in current best thresholds list s as well 
    # as updated values objectves (those seen in SAT model/assignemnt witness_vals_dict) are reported in json
    # and csv/table formats, and also this info is logged in the run log file.
    def report_current_thresholds(self, s, witness_vals_dict, objv_bounds_dict, objv_names, objv_exprs, 
            completed:bool, call_n:tuple[Union[int,str]], scale_objv):
        #print('s', s, 'scale_objv', scale_objv, 'objv_bounds_dict', objv_bounds_dict)
        #print('witness_vals_dict', witness_vals_dict); print('completed', completed, 'call_n', call_n, 'scale_objv', scale_objv)
        assert s is not None
        assert s.count(None) == 0
        if scale_objv:
            s_origin = [self.unscale_constant_val(objv_bounds_dict, objv_name, b) 
                for (objv_name, b) in zip(objv_names,s)]; #print('s_unscaled', s_unscaled)
            s_scaled = s
        else:
            s_origin = s
            s_scaled = [self.scale_constant_val(objv_bounds_dict, objv_name, b) 
                for (objv_name, b) in zip(objv_names,s)]; #print('s_unscaled', s_unscaled)
        s_scaled_str = ["%.6f" % e for e in s_scaled]; #print('s_scaled_str', s_scaled_str)
        s_origin_str = ["%.6f" % e for e in s_origin]; #print('s_origin_str', s_origin_str)
        s_scaled_dict = dict(zip(objv_names, s_scaled_str))
        s_origin_dict = dict(zip(objv_names, s_origin_str))
        if completed:
            key_label = 'final'
            self._opt_logger.info('Pareto optimization completed with objectives thresholds: ' + 
                '\n    Scaled to [0,1]: {}\n    Original  scale: {}\n'.format(s_scaled_dict, s_origin_dict))
        elif call_n is None:
            key_label = 'vacuity'
            self._opt_logger.info('Pareto optimization vacuity completed with objectives thresholds: ' + 
                '\n    Scaled to [0,1]: {}\n    Original  scale: {}\n'.format(s_scaled_dict, s_origin_dict))
        else:
            key_label = 'iter'+str(call_n)
            self._opt_logger.info('Pareto optimization in progress with objectives thresholds: ' + 
                '\n    Scaled to [0,1]: {}\n    Original  scale: {}\n'.format(s_scaled_dict, s_origin_dict))
        
        # update self.best_config_dict with the improved configuration (towards the optimum)
        self.best_config_dict[key_label] = {}
        if witness_vals_dict is not None:
            objv_vals_dict = dict([(objv_name, eval(objv_expr, {},  witness_vals_dict)) 
                for objv_name, objv_expr in zip(objv_names, objv_exprs) if objv_expr is not None])
            for i, objv_name in enumerate(objv_names):
                # we should be adding objv_names[i] to self.best_config_dict[key_label] only once
                assert objv_names[i] not in self.best_config_dict[key_label]
                # objective's value in SAT model/assignment must be greater or equal to the objective's 
                # threshold in s_origin (values of objectives in the SAT model are in original scale, 
                # just like values of model interface variables -- inputs, knobs, responses).
                #print(s_origin[i],  objv_vals_dict[objv_names[i]])
                #assert objv_vals_dict[objv_names[i]] >= s_origin[i]
                self.best_config_dict[key_label][objv_names[i]] = {
                    'value_in_config': objv_vals_dict[objv_names[i]],
                    'threshold_scaled':s_scaled[i], 'threshold':s_origin[i],
                    'max_in_data': objv_bounds_dict[objv_names[i]]['max'],
                    'min_in_data': objv_bounds_dict[objv_names[i]]['min']}

            for key, val in witness_vals_dict.items():
                if key in objv_names:
                    continue
                self.best_config_dict[key_label][key] = {'value_in_config': val}
                if key in self.resp_names and self.syst_expr_dict is not None:
                    if key in self.syst_expr_dict.keys():
                        #print('key', key, 'self.syst_expr_dict[key]', self.syst_expr_dict[key])
                        # compute value of the original system function on input and knob values for witness_vals_dict
                        system_val = eval(self.syst_expr_dict[key], {},  witness_vals_dict); #print('system_val', system_val)
                        self.best_config_dict[key_label][key]['value_in_system'] = system_val
            self.best_config_df = self.prog_dict_to_df(); 
        else:
            #assert False
            for i, objv_name in enumerate(objv_names):
                assert objv_names[i] not in self.best_config_dict[key_label]
                self.best_config_dict[key_label][objv_names[i]] = {
                    'threshold_scaled':s_scaled[i], 'threshold':s_origin[i], 
                    'max_in_data': objv_bounds_dict[objv_names[i]]['max'],
                    'min_in_data': objv_bounds_dict[objv_names[i]]['min']}
        # dump the final config and/or the updated self.best_config_dict progress report
        with open(self.optimization_progress_file+'.json', 'w') as f: #json.dump(asrt_res_dict, f)
            json.dump(self.best_config_dict, f, indent='\t', cls=np_JSONEncoder)
        self.best_config_df.to_csv(self.optimization_progress_file+'.csv', index=False)
        if completed:
            self.mode_status_dict['smlp_execution'] = 'completed'
            with open(self.optimization_results_file+'.json', 'w') as f:
                json.dump(self.best_config_dict['final'] | self.mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            final_config_df = self.best_config_df.drop_duplicates(subset=self.feat_names, inplace=False)
            final_config_df.to_csv(self.optimization_results_file+'.csv', index=False)            
    
    
    def update_optimization_reports(self, stable_witness_terms, l, u, call_info, iter_count, scale_objectives, 
            objv_name, objv_witn_val_term, objv_bounds, objv_expr, orig_objv_name, update_progress_report:bool, sat_approx, sat_precision):
        stable_witness_terms[objv_name] = objv_witn_val_term
        if call_info is not None and call_info['update_thresholds'] and update_progress_report:
            witness_vals_dict = self._smlpTermsInst.witness_term_to_const(stable_witness_terms, sat_approx,  
                sat_precision); #print('witness_vals_dict', witness_vals_dict)
            #if objv_name in witness_vals_dict:
            del witness_vals_dict[objv_name]; #print('witness_vals_dict after del', witness_vals_dict)
            #print('call_info', call_info, 'iter', iter_count)
            s = call_info['objv_thresholds']
            for i in call_info['active_objv']:
                s[i] = l
            #print('s for single objv', s)
            self.report_current_thresholds(s, witness_vals_dict, self.objv_bounds_dict, self.objv_names, self.objv_exprs, 
                False, (call_info['global_iter'], iter_count), scale_objectives)

        # Enhancement: could aovid computing unscaled_threshold_lo and unscaled_threshold_up in case scale_objectives is True
        # and only add fields 'threshold_lo_scaled' and 'threshold_up_scaled' to stable_witness_terms
        #print('before updating P: l0', l0, 'u0', u0, 'l', l, 'u', u)
        if scale_objectives: 
            #print('objv_bounds', objv_bounds)
            objectives_unscaler_terms_dict = self._scalerTermsInst.feature_unscaler_terms(objv_bounds, [orig_objv_name])
            # substitute scaled objective variables with scaled objective terms
            # in original objective terms within objectives_unscaler_terms_dict
            if objv_expr is not None:
                orig_objv_const_term = smlp.subst(objectives_unscaler_terms_dict[orig_objv_name], #objv_term, 
                    {self._scalerTermsInst._scaled_name(orig_objv_name): objv_witn_val_term})
                #print('orig_objv_const_term', orig_objv_const_term)
                objv_name_unscaled = self._scalerTermsInst._unscaled_name(objv_name)
                if objv_name_unscaled in self.objv_names:
                    stable_witness_terms[objv_name_unscaled] = orig_objv_const_term 
            if l not in [np.inf, -np.inf]:
                unscaled_threshold_lo = self._scalerTermsInst.unscale_constant_term(objv_bounds, orig_objv_name, l)
                stable_witness_terms['threshold_lo_scaled'] = smlp.Cnst(l)
                stable_witness_terms['threshold_lo'] = unscaled_threshold_lo
            if u not in [np.inf, -np.inf]:
                unscaled_threshold_up = self._scalerTermsInst.unscale_constant_term(objv_bounds, orig_objv_name, u)
                #print('unscaled_threshold_up: l', l, 'unsc', unscaled_threshold_up)
                stable_witness_terms['threshold_up_scaled'] = smlp.Cnst(u)
                stable_witness_terms['threshold_up'] = unscaled_threshold_up
        else:
            #assert False
            if l not in [np.inf, -np.inf]:
                stable_witness_terms['threshold_lo'] = smlp.Cnst(l)
            if u not in [np.inf, -np.inf]:
                stable_witness_terms['threshold_up'] = smlp.Cnst(u)
        stable_witness_terms['max_in_data'] = smlp.Cnst(objv_bounds[orig_objv_name]['max'])
        stable_witness_terms['min_in_data'] = smlp.Cnst(objv_bounds[orig_objv_name]['min'])
        #print('stable_witness_terms', stable_witness_terms, flush=True)
        stable_witness_vals = self._smlpTermsInst.witness_term_to_const(
            stable_witness_terms, sat_approx, sat_precision)
        
        return stable_witness_vals
    
    
    # pareto optimization, reduced to single objective optimization and condition queries.
    def optimize_pareto_objectives(self, feat_names:list[str], resp_names:list[str], 
            model_full_term_dict:dict, objv_names:list, objv_exprs:list, objv_bounds_dict:dict, alpha:smlp.form2, 
            beta:smlp.form2, eta:smlp.form2, theta_radii_dict:dict,
            epsilon:float, smlp_domain:smlp.domain, delta:float, solver_logic:str, strategy:str, scale_objv:bool, data_scaler:str, 
            sat_approx=False, sat_precision=64, save_trace=False):
        self._opt_logger.info('Pareto optimization: Start')
        assert epsilon > 0 and epsilon < 1
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs)
        
        scale_objectives = scale_objv and data_scaler != 'none'
        if not scale_objectives:
            raise Exception('Pareto optimization requires scale_objectives to be True and data_scaler not none')
        #assert scale_objectives
        objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict = \
            self._modelTermsInst.compute_objectives_terms(objv_names, objv_exprs, objv_bounds_dict, scale_objectives)

        objv_count = len(objv_names)
        objv_enum = range(objv_count)
        
        # In this dictionary we record the achieved bounds on fixed objectives (these bounds are not
        # attampted for improvement in future iterations). This dictionary is for sanity check only,
        # implemented by fuction sanity_check_fixed_objv_thresholds().
        fixed_onjv_dict = {}
        def sanity_check_fixed_objv_thresholds(t:list[float], fixed_onjv_dict):
            #print('t', t, 'fixed_onjv_dict', fixed_onjv_dict)
            for k,v in fixed_onjv_dict.items():
                assert t[k] == v
        
        # IDs of objectives whose bounds can still be potentially improved by at least epsilon;
        # in each iteration of the while loop below, the algo aims to improves a larges subset
        # of these objectives simultaneously, and updates K by droping objectoves that cannot be improved
        # Think of objectives in K as "active'
        K = [i for i in range(objv_count)]; #print('K', K)
        # vector of length objv_count with None at indexes in K -- IDs of objectives whise bounds
        # have been fixed. They are updated at the begining of each iteration of the while loop.
        # Think of objectives that have values in s as 'fixed'
        s = [None] * objv_count; #print('s', s)
        direction = 'up' # TODO !!! add as argument to pareto or derive from bad / good value options
        eta_F_t_conj = eta
        call_n = 0 # count iterations
        witness = None # the latest sat assignmenet returned by active_objectives_max_min_bounds()
                         # or query_condition()
        
        while len(K) > 0:
            #print('start of while iteration: K =', K, 's =', s)
            call_info_dict = {'global_iter': call_n, 'update_thresholds': True, 'active_objv': K} 
            self._opt_tracer.info('pareto_iteration,{},{},{}'.format(str(call_n), '__'.join(objv_names), '__'.join([str(e) for e in s])))
            c_lo, c_up, witness = self.active_objectives_max_min_bounds(model_full_term_dict, objv_terms_dict, 
                s, smlp_domain, alpha, beta, eta, theta_radii_dict, epsilon, delta, solver_logic, strategy, direction,
                scale_objectives, objv_bounds_dict, call_info_dict, sat_approx, sat_precision, save_trace)
            #print('c_lo', c_lo, 'c_up', c_up); print('witness', witness);
            assert c_lo != np.inf
            
            # t is improvement of s with objectives bounds found in (just computed) c_lo, c_up, witness
            t = s; #print('t = s ', t)
            sanity_check_fixed_objv_thresholds(t, fixed_onjv_dict)
            for j in objv_enum:
                if j in K:
                    t[j] = c_lo
            #print('t after joint bonds increase', t)
            assert t.count(None) == 0
            sanity_check_fixed_objv_thresholds(t, fixed_onjv_dict)
            if witness is not None:
                self.report_current_thresholds(t, witness, objv_bounds_dict, objv_names, objv_exprs, 
                    False, (call_n, 'completed'), scale_objectives)
            call_n = call_n + 1
            
            # If only a single objective remains active (not fixed), its threshold cannot be increased
            # by at least epsilon (by construction of single objective optimization algorithm called
            # above from active_objectives_max_min_bounds()); hence the while loop can terminate here.
            if len(K) == 1:
                # update s to report final result below before exiting this function
                s = t
                break
            
            # At least one of the objectives whose bounds were adjusted above cannot be improved
            # further (because we improved these objectives to the joint least upper bound). We now
            # find out which of the objectives can be improved further so that they will be kept in
            # K and the rest will be dropped from K (there will be no further attempts to improve 
            # these dropped objectives); the objectives that will remain in K will be attempted to 
            # improve by epsilon or higher in the next iteration of the while loop.

            K_pr = []
            t_for_loop = t.copy(); # using copy() here to insure changes in t do not affect t_for_loop
            # Enhancement !!! when j becomes the last element in K and none of the objectives fron K got fixed
            # when we conclude that the last objective in K needs to be fixed (and skip this check in SAT)
            # this fix will affect Test 94 -- there will be one less call to SAT
            assert len(K) == len(set(K)) # K should not contain duplicates
            raised_count = 0  # count of objectives whose lower bounds have been raised so far within the 
                              # for loop below. As soon as raised_count == len(K)-1, then there is one
                              # remaining objective with index in K and it should be fixed to current threshold
            for j in K:
                K_pr.append(j)
                t = t_for_loop.copy(); # using copy() here to insure changes in t do not affect t_for_loop
                for i in K_pr:
                    t[i] = c_lo + epsilon; 
                #print('K_pr', K_pr); print('t with epsilon increased candidate at k_pr positions', t)
                assert t.count(None) == 0
                sanity_check_fixed_objv_thresholds(t, fixed_onjv_dict)
                for k in objv_enum:
                    #print('k', k)
                    if k in K_pr:
                        assert t[k] == c_lo + epsilon
                    else:
                        assert t[k] == s[k]
                #objv_bounds_in_search = dict([(objv_names, {'min':s[j], 'max':1}) for i in objv_enum])
                #print('objv_bounds_in_search', objv_bounds_in_search)
                if raised_count == len(K) - 1:
                    quer_res = {}
                    quer_res['query_status'] = 'UNSAT'
                else:    
                    self._opt_logger.info('Checking whether to fix objective {} at threshold {}...\n'.format(str(j), str(s[j])))
                    self._opt_tracer.info('activity check, objective {} threshold {}'.format(str(objv_names[j]), str(s[j])))
                    #print('objv_terms_dict', objv_terms_dict)
                    quer_form = smlp.true
                    for i in objv_enum:
                        #print('obv i', list(objv_terms_dict.keys())[i])
                        quer_form = self._smlpTermsInst.smlp_and(quer_form, list(objv_terms_dict.values())[i] > smlp.Cnst(t[i]))
                    #print('queryform', quer_form)
                    quer_and_beta = self._smlpTermsInst.smlp_and(quer_form, beta) if not beta == smlp.true else quer_form
                    opt_quer_name = 'thresholds_' + '_'.join(str(x) for x in t) + '_check'
                    quer_res = self._queryInst.query_condition(True, model_full_term_dict, opt_quer_name, 'True', quer_and_beta, 
                        smlp_domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, sat_approx, sat_precision)
                #print('quer_res', quer_res)
                if quer_res['query_status'] != 'STABLE_SAT':
                    self._opt_logger.info('Fixing objective {} at threshold {}...\n'.format(str(j), str(s[j])))
                    assert j not in fixed_onjv_dict.keys()
                    fixed_onjv_dict[j] = s[j]
                    K_pr.remove(j); #print('K_pr after reduction', K_pr)
                    t[j] = t_for_loop[j]
                else:
                    self._opt_logger.info('Not fixing objective {} at threshold {}...\n'.format(str(j), str(s[j])))
                    self._opt_logger.info('Lower bounds of objectives {} can be raised to threshold {}...\n'.
                        format(str([objv_names[i] for i in K_pr]), str(c_lo + epsilon)))
                    raised_count = raised_count + 1
                    witness = quer_res['witness'] # keeping witness up-to-date
                    self.report_current_thresholds(t, witness, objv_bounds_dict, objv_names, objv_exprs, 
                        False, (call_n, 'selection'), scale_objectives)
                    #print('update bounds report here')
                    
                #print('K_pr after inner for loop iteration', K_pr)
            #print('K_pr after inner for loop', K_pr)
            s = [None]*objv_count
            for i in objv_enum:
                if i not in K_pr:
                    s[i] = t[i]; 
            #print('s at the end of while iteration', s)
            sanity_check_fixed_objv_thresholds(s, fixed_onjv_dict)
            K = K_pr
        #print('end of while loop')
        #print('s', s)
        
        self.report_current_thresholds(s, witness, objv_bounds_dict, objv_names, objv_exprs, 
            True, (call_n, 'Final'), scale_objectives)
        
        self._opt_logger.info('Pareto optimization: End')
        return s
    
    # synthesis feasibility check of the constraints on solver instance in optimization and optsyn modes; these constraints 
    # do not include constraints imposed on the optimization / optsyn objectives as part of optimization algo
    # in these two modes.
    def check_synthesis_feasibility(self, feasibility:bool, objv_names:list[str], objv_exprs:list[str], objv_bounds_dict:dict, 
            scale_objv:bool, feat_names:list[str], resp_names:list[str], model_full_term_dict:dict, beta:smlp.form2, 
            domain:smlp.domain, eta:smlp.form2, alpha:smlp.form2, theta_radii_dict:dict, delta:float, solver_logic:str, 
            float_approx:bool, float_precision:int):
        if feasibility:
            self._opt_logger.info('Pareto optimization synthesis feasibility check: Start')
            self._opt_tracer.info('synthesis_feasibility')
            quer_res = self._queryInst.query_condition(True, model_full_term_dict, 'synthesis_feasibility', 'True', beta, 
                domain, eta, alpha, theta_radii_dict, delta, solver_logic, True, float_approx, float_precision)
            #print('beta', beta); print('quer_res', quer_res)
            if quer_res['query_status'] == 'UNSAT':
                self._opt_logger.info('Pareto optimization synthesis feasibility check: End')
                return True, None
            elif quer_res['query_status'] == 'STABLE_SAT':
                witness_vals_dict = quer_res['witness']; #print('witness_vals_dict', witness_vals_dict)
                s = []
                for objv_name, objv_expr in zip(objv_names, objv_exprs):
                    s.append(eval(objv_expr, {},  witness_vals_dict))
                #print('s', s)
                
                # here s is based on response values in the SAT model witness_vals_dict, and the latter
                # is always delivered in original scale; so we pass False as value of scale_objv independently
                # from it value obtained from user input
                self.report_current_thresholds(s, witness_vals_dict, objv_bounds_dict, objv_names, objv_exprs, 
                    False, None, False)
                
                self._opt_logger.info('Pareto optimization synthesis feasibility check: End')
                return False, s
        else:
            self._opt_logger.info('Skipping pareto optimization synthesis feasibility check')
            return False, None
    
    
    # SMLP optimization of multiple objectives -- pareto optimization or optimization per objective
    # TODO !!!: X and y are used to estimate bounds on objectives from training data, and the latter is not
    #     available in model re-run mode. Need to estimate objectove bounds in a different way and pass to this
    #     function (and to smlp_optsyn() instead of passing X,y; The bounds on objectives are not strictly necessary,
    #     any approximation may be used, but accurate approximation might reduce iterations count needed for
    #     computing optimal confoguurations (in optimize and optsyn modes)
    def smlp_optimize(self, syst_expr_dict:dict, algo:str, model:dict, X:pd.DataFrame, y:pd.DataFrame, model_features_dict:dict, 
            feat_names:list[str], resp_names:list[str], 
            objv_names:list[str], objv_exprs, pareto:bool, strategy:str, #asrt_names:list[str], asrt_exprs, 
            quer_names:list[str], quer_exprs, delta:float, epsilon:float, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool,  
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        self.objv_names = objv_names
        self.objv_exprs = objv_exprs
        self.feat_names = feat_names
        self.resp_names = resp_names
        self.syst_expr_dict = syst_expr_dict
        
        # output to user initial values of mode status
        with open(self.optimization_results_file+'.json', 'w') as f:
            json.dump(self.mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self._modelTermsInst.create_model_exploration_base_components(
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            #delta, epsilon, #objv_names, objv_exprs, None, None, None, None, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, #scale_objv, 
            float_approx, float_precision, data_bounds_json_path)

        self.mode_status_dict = self._queryInst.update_consistecy_results(self.mode_status_dict, 
            interface_consistent, model_consistent, 'optimization_status', self.optimization_results_file+'.json')
        if not interface_consistent:
            self._opt_logger.info('Input and knob interface constraints are inconsistent; aborting...')
            return
        elif not model_consistent:
            self._opt_logger.info('Input and knob interface constraints are inconsistent with model constraints; aborting...')
            return
            
        # TODO !!!: when re-using a saved model, X and y are not available, need to adapt computation of 
        # objv_bounds_dict for that case -- say simulate the model, then compute objectives' values
        # for these simulation vectors, them compute the bounds on the objectives from simulation data
        objv_bounds_dict = self.compute_objectives_bounds(X, y, objv_names, objv_exprs); #print('objv_bounds_dict', objv_bounds_dict)
        self.objv_bounds_dict = objv_bounds_dict
        
        # instance consistency check (are the assumptions contradictory?)
        contradiction, thresholds = self.check_synthesis_feasibility(vacuity, objv_names, objv_exprs, objv_bounds_dict, scale_objv, 
            feat_names, resp_names, model_full_term_dict, beta, 
            domain, eta, alpha, theta_radii_dict, delta, solver_logic, float_approx, float_precision)
        self.mode_status_dict['synthesis_feasible'] = str(not contradiction).lower()
        if contradiction:
            # instance is contradictory -- more precisely, stable witness does not exist; abort
            self._opt_logger.info('Model configuration optimization instance is inconsistent with synthesis constraints; aborting...')
            self.mode_status_dict['smlp_execution'] = 'aborted'
            with open(self.optimization_results_file+'.json', 'w') as f:
                json.dump(self.mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return
        
        # The eager strategy is only supported for problems without (free) inputs (that is, model inputs are all knobs) 
        input_features = [ft for ft in feat_names if ft in self._modelTermsInst._specInst.get_spec_inputs]
        if strategy == 'eager' and len(input_features) > 0:
            strategy = 'lazy'
        
        if pareto:
            self.optimize_pareto_objectives(feat_names, resp_names, model_full_term_dict, #X, y, 
                objv_names, objv_exprs, objv_bounds_dict, alpha, beta, eta, theta_radii_dict, 
                epsilon, domain, delta, solver_logic, strategy, scale_objv, data_scaler,
                sat_approx=True, sat_precision=64, save_trace=False)
        else:
            self.optimize_single_objectives(feat_names, resp_names, model_full_term_dict, #X, y, 
                objv_names, objv_exprs, objv_bounds_dict, alpha, beta, eta, theta_radii_dict, 
                epsilon, domain, delta, solver_logic, strategy, scale_objv, data_scaler, 
                sat_approx=True, sat_precision=64, save_trace=False)
        
    # SMLP optsyn mode that performs multi-objective optimization (pareto or per-objective) and insures
    # that with the selected configuration of knobs all assertions are also satisfied (in addition to
    # any other model interface constraints or configuration stability constraints)
    def smlp_optsyn(self, syst_expr_dict:dict, algo, model, X:pd.DataFrame, y:pd.DataFrame, model_features_dict:dict, feat_names:list[str], resp_names:list[str], 
            objv_names, objv_exprs, pareto:bool, strategy:str, asrt_names, asrt_exprs, quer_names, quer_exprs, delta:float, epsilon:float, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, 
            data_scaler:str, scale_feat:bool, scale_resp:bool, scale_objv:bool,
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        self.objv_names = objv_names
        self.objv_exprs = objv_exprs
        self.feat_names = feat_names
        self.resp_names = resp_names
        self.syst_expr_dict = syst_expr_dict
        
        domain, syst_term_dict, model_full_term_dict, eta, alpha, beta, interface_consistent, model_consistent = \
        self._modelTermsInst.create_model_exploration_base_components(
            syst_expr_dict, algo, model, model_features_dict, feat_names, resp_names, 
            #delta, epsilon, #objv_names, objv_exprs, asrt_names, asrt_exprs, None, None, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, #scale_objv, 
            float_approx, float_precision, data_bounds_json_path)

        self.mode_status_dict = self._queryInst.update_consistecy_results(self.mode_status_dict, 
            interface_consistent, model_consistent, 'optsyn_status', self.optimization_results_file+'.json')
        if not interface_consistent:
            self._opt_logger.info('Input and knob interface constraints are inconsistent; aborting...')
            return
        elif not model_consistent:
            self._opt_logger.info('Input and knob interface constraints are inconsistent with model constraints; aborting...')
            return
        
        # TODO !!!: when re-using a saved model, X and y are not available, need to adapt computation of 
        # objv_bounds_dict for that case -- say simulate the model, then compute objectives' values
        # for these simulation vectors, them compute the bounds on the objectives from simulation data
        objv_bounds_dict = self.compute_objectives_bounds(X, y, objv_names, objv_exprs); #print('objv_bounds_dict', objv_bounds_dict)
        self.objv_bounds_dict = objv_bounds_dict

        if asrt_exprs is not None:
            assert asrt_names is not None
            asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                    for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
            asrt_conj = self._smlpTermsInst.smlp_and_multi(list(asrt_forms_dict.values()))
        else:
            asrt_conj = smlp.true
        beta = self._smlpTermsInst.smlp_and(beta, asrt_conj) if beta != smlp.true else asrt_conj
        
        # instance consistency check (are the assumptions contradictory?)
        contradiction, thresholds = self.check_synthesis_feasibility(vacuity, objv_names, objv_exprs, objv_bounds_dict, scale_objv, 
            feat_names, resp_names, model_full_term_dict, beta, 
            domain, eta, alpha, theta_radii_dict, delta, solver_logic, float_approx, float_precision)
        self.mode_status_dict['synthesis_feasible'] = str(not contradiction).lower()
        if contradiction:
            # instance is contradictory -- more precisely, stable witness does not exist; abort
            self._opt_logger.info('Model configuration optimized synthesis instance is inconsistent with synthesis constraints; aborting...')
            self.mode_status_dict['smlp_execution'] = 'aborted'
            with open(self.optimization_results_file+'.json', 'w') as f:
                json.dump(self.mode_status_dict, f, indent='\t', cls=np_JSONEncoder)
            return

        # The eager strategy is only supported for problems without (free) inputs (that is, model inputs are all knobs) 
        input_features = [ft for ft in feat_names if ft in self._modelTermsInst._specInst.get_spec_inputs]
        if strategy == 'eager' and len(input_features) > 0:
            strategy = 'lazy'
        
        if pareto:
            self.optimize_pareto_objectives(feat_names, resp_names, model_full_term_dict, #X, y, 
                objv_names, objv_exprs, objv_bounds_dict, alpha, beta, eta, theta_radii_dict, 
                epsilon, domain, delta, solver_logic, strategy, scale_objv, data_scaler,
                sat_approx=True, sat_precision=64, save_trace=False)            
        else:
            self.optimize_single_objectives(feat_names, resp_names, model_full_term_dict, #X, y, 
                objv_names, objv_exprs, objv_bounds_dict, alpha, beta, eta, theta_radii_dict, 
                epsilon, domain, delta, solver_logic, strategy, scale_objv, data_scaler, 
                sat_approx=True, sat_precision=64, save_trace=False)
