import smlp
from smlp_py.smlp_terms import SmlpTerms, ModelTerms, ScalerTerms
from smlp_py.smlp_query import SmlpQuery
from smlp_py.smlp_utils import (str_to_bool, np_JSONEncoder)
            
from fractions import Fraction
from decimal import Decimal
import json
import pandas as pd
import keras
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
        
        self._DEF_APPROXIMATE_FRACTIONS:bool = True
        self._DEF_FRACTION_PRECISION:int  = 64
        
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
                'help': 'Should solver rpoblem instance vacuitr check be performed? ' + 
                    'Vacuty checks whether the the constraints are consistent and therefore at least ' +
                    'one satisfiable assignment exist to solver constraints. Relevant in "verify", "query", ' +
                    '"optimize" and "tune" modes [default: ' + str(self._DEF_VACUITY_CHECK) + ']'}
        }
    
    def set_logger(self, logger):
        self._opt_logger = logger 
        self._queryInst.set_logger(logger)
        self._modelTermsInst.set_logger(logger)
                
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
    
    @property
    def optimization_results_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_optimization_results.json'
    
    def print_result(self, res):
        if isinstance(res, smlp.sat):
            print('SAT with model')
            for var, cnst in res.model.items():
                print(' ', var, '=', smlp.Cnst(cnst))
        elif isinstance(res, smlp.unsat):
            print('UNSAT')
        else:
            print('unknown:', res.reason)
    
    
    # Optimization for single objective.
    # assuming in first implementation that objectives are scaled to [0,1] -- not using
    # objv_bounds, data_scaler, objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict, 
    # also not using thresholds_dict -- covering a general case
    def optimize_single_objective(self, model_full_term_dict:dict, objv_name:str, objv_expr:str, objv_term:smlp.term2, 
            epsilon:float, smlp_domain:smlp.domain, eta:smlp.form2, theta_radii_dict:dict, alpha:smlp.form2, beta:smlp.form2, delta:float, solver_logic:str, 
            scale_objectives:bool, orig_objv_name:str, objv_bounds:dict, sat_approx=False, sat_precision=64, save_trace=False,
            l0=None, u0=None):
        self._opt_logger.info('Optimize single objective ' + str(objv_name) + ': Start')
        
        #TODO !!!: we assume objectives were scaled to [0,1] and l0 and u0 are initialized to 0 and 1 respectively
        #assert scale_objectives 
        P = [] # known candidates and lower bounds
        N = [] # known counter-examples and upper bounds
        l = -np.inf # undefined lower bound
        u = np.inf # undefined upper bound
        
        # initial lower bound l0 and initial upper bound u0
        if u0 is None and l0 is None:
            if scale_objectives or objv_bounds is None:
                l0 = 0 
                u0 = 1 
            else:
                l0 = objv_bounds[orig_objv_name]['min']
                u0 = objv_bounds[orig_objv_name]['max']
            print('l0', l0, 'u0', u0)
            assert l0 < u0

        while True:
            if u == np.inf:
                (T, u0) = (u0, 2*u0 - l0)
            elif l == -np.inf:
                (T, l0) = (l0, 2*l0 - u0)
            else:
                T = (l + u) / 2
            
            quer_form = objv_term > smlp.Cnst(T)
            quer_expr = '{} > {}'.format(objv_expr, str(T)) if objv_expr is not None else None
            quer_name = objv_name + '_' + str(T)
            quer_and_beta = self._smlpTermsInst.smlp_and(quer_form, beta) if not beta == smlp.true else quer_form
            #print('quer_and_beta', quer_and_beta)
            quer_res = self._queryInst.query_condition(
                model_full_term_dict, quer_name, quer_expr, quer_and_beta, smlp_domain, # query_form
                eta, alpha, theta_radii_dict, delta, solver_logic, False, sat_approx, sat_precision); #print('quer_res', quer_res)  beta, 
            stable_witness_status = quer_res['status']
            stable_witness_terms = quer_res['witness']
            if stable_witness_status == 'UNSAT':
                u = T
            elif stable_witness_status == 'STABLE_SAT':
                # need to increase the bound/threshold T
                l = T
                objv_witn_val_term = smlp.subst(objv_term, stable_witness_terms); #print('objv_witn_val_term', objv_witn_val_term)
                stable_witness_terms[objv_name] = objv_witn_val_term
                
                
                # TOD0 !!!: could avid computing unscaled_threshold_lo and unscaled_threshold_up in case scale_objectives is True
                # and only add feilds 'threshold_lo_scaled' and 'threshold_up_scaled' to stable_witness_terms
                #if l + epsilon > u:
                if scale_objectives: 
                    #print('objv_bounds', objv_bounds)
                    objectives_unscaler_terms_dict = self._scalerTermsInst.feature_unscaler_terms(objv_bounds, [orig_objv_name])
                    unscaled_threshold_lo = self._scalerTermsInst.unscale_constant(objv_bounds, orig_objv_name, l)
                    #print('unscaled_threshold_lo: l', l, 'unsc', unscaled_threshold_lo)
                    unscaled_threshold_up = self._scalerTermsInst.unscale_constant(objv_bounds, orig_objv_name, u)
                    #print('unscaled_threshold_up: l', l, 'unsc', unscaled_threshold_up)
                    # substitute scaled objective variables with scaled objective terms
                    # in original objective terms within objectives_unscaler_terms_dict
                    orig_objv_const_term = smlp.subst(objectives_unscaler_terms_dict[orig_objv_name], #objv_term, 
                        {self._scalerTermsInst._scaled_name(orig_objv_name): objv_witn_val_term})
                    #print('orig_objv_const_term', orig_objv_const_term)
                    stable_witness_terms[self._scalerTermsInst._unscaled_name(objv_name)] = orig_objv_const_term 
                    stable_witness_terms['threshold_lo_scaled'] = smlp.Cnst(l)
                    stable_witness_terms['threshold_up_scaled'] = smlp.Cnst(u)
                    stable_witness_terms['threshold_lo'] = unscaled_threshold_lo
                    stable_witness_terms['threshold_up'] = unscaled_threshold_up
                else:
                    #stable_witness_terms['threshold_lo_scaled'] = smlp.Cnst(l)
                    #stable_witness_terms['threshold_up_scaled'] = smlp.Cnst(u)
                    stable_witness_terms['threshold_lo'] = smlp.Cnst(l)
                    stable_witness_terms['threshold_up'] = smlp.Cnst(u)
                stable_witness_terms['max_in_data'] = smlp.Cnst(objv_bounds[orig_objv_name]['max'])
                stable_witness_terms['min_in_data'] = smlp.Cnst(objv_bounds[orig_objv_name]['min'])    
                stable_witness_vals = self._smlpTermsInst.sat_model_term_to_const(
                    stable_witness_terms, sat_approx, sat_precision)
                #print('adding to P', (stable_witness_vals, stable_witness_vals[objv_name]))
                #P.append((stable_witness_vals, stable_witness_vals[objv_name]))
                #print('========================= l', l, 'u', u, 'epsilon', epsilon); print(l + epsilon > u);
                #if save_trace or l + epsilon > u:
                # TODO !!! we only use the last element of P -- could override existing last element instead of inserting
                P.append(stable_witness_vals)
            else:
                raise Exception('Unsupported value ' + str(stable_witness_status) + ' received from query_conditions')
                
            if l + epsilon > u:
                # false when if l = -np.inf and u = np.inf
                print('l', l, 'u',u, 'exit while true')
                break

        self._opt_logger.info('Optimize single objective ' + str(objv_name) + ': End')
        #print('P', P); print('P[-1]', P[-1])
        return P[-1]
    
    # optimize multiple objectives but each one separately -- the objectives might have optimal value
    # in different sub-spaces in the configuration space. This function essentially simply iterates in
    # the objectives and finds a stable optimum for each one using function optimize_single_objective().
    def optimize_single_objectives(self, X:pd.DataFrame, y:pd.DataFrame, feat_names:list, resp_names:list, 
            model_full_term_dict:dict, objv_names:list, objv_exprs:list, alpha:smlp.form2, beta:smlp.form2, 
            eta:smlp.form2, theta_radii_dict:dict, epsilon:float, smlp_domain:smlp.domain, delta:float, solver_logic:str, scale_objv:bool,  
            data_scaler:str, sat_approx=False, sat_precision=64, save_trace=False):
        assert X is not None
        assert y is not None
        assert epsilon > 0 and epsilon < 1
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs)
        scale_objectives = scale_objv and data_scaler != 'none'
        #assert scale_objectives
        # TODO !!!: when re-using a saved model, X and y are not available, need to adapt compute_objectives_terms
        # and computation of objv_bounds for that case -- say simulate the model, then compute objectives' values
        # for these simulation vectors, them compute the bounds on the objectives from simulation data
        objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict, objv_bounds = \
            self._modelTermsInst.compute_objectives_terms(objv_names, objv_exprs, 
                scale_objectives, feat_names, resp_names, X, y)
        #print('objv_terms_dict', objv_terms_dict); print('orig_objv_terms_dict', orig_objv_terms_dict); print('scaled_objv_terms_dict', scaled_objv_terms_dict)
        # TODO: set sat_approx to False once dump and load with Fractions will work
        opt_conf = {}
        for i, (objv_name, objv_term) in enumerate(list(objv_terms_dict.items())):
            objv_expr = objv_exprs[i]
            if scale_objectives:
                objv_epsn = epsilon
            else:
                objv_epsn = self._scalerTermsInst.unscale_constant_val(objv_bounds, objv_names[i], epsilon)
            #print('objv_epsn', objv_epsn)
            opt_conf[objv_names[i]] = self.optimize_single_objective(model_full_term_dict, objv_name, objv_expr, 
                objv_term, objv_epsn, smlp_domain, eta, theta_radii_dict, alpha, beta, delta, solver_logic, scale_objectives, objv_names[i], 
                objv_bounds, sat_approx=True, sat_precision=64, save_trace=False); #print('opt_conf', opt_conf)                                       
        with open(self.optimization_results_file, 'w') as f: #json.dump(asrt_res_dict, f)
            json.dump(opt_conf, f, indent='\t', cls=np_JSONEncoder)
    
    
    # direction hi means we want to maximize the objectives, direction lo means we want to minimize.
    # Compute smlp expression that represents the maximumum amongs the minimums of the objectives that still do 
    # not have their greatest lower bound computed as part of pareto optimization algorithm, under the input
    # constraint eta and the constraints that represent already determined greatest lower bounds for the remaining
    # objectives. This max-min expression on the objectives will be passed to the single objective otimization
    # algorithm to determaine what is the best greatest lower bound that we can fins and fix for these objectives.
    # TODO !!!: do we need theta here???
    def unbound_objectives_max_min_bounds(self, model_full_term_dict:dict, objv_terms_dict:dict, t:list, 
            smlp_domain:smlp.domain, alpha:smlp.form2, beta:smlp.form2, eta:smlp.form2, theta_radii_dict, epsilon:float, 
            delta:float, solver_logic:str, direction, scale_objectives, objv_bounds, sat_approx, sat_precision, save_trace):
        assert direction == 'up'
        eta_F_t = eta
        min_objs = None
        min_name = ''
        for j, (objv_name, objv_term) in enumerate(objv_terms_dict.items()):
            if t[j] is not None:
                eta_F_t = self._smlpTermsInst.smlp_and(eta_F_t, objv_term > smlp.Cnst(t[j]))
            else:
                min_name = min_name + '_' + objv_name if min_name != '' else objv_name
                if min_objs is not None:
                    min_objs = smlp.Ite(objv_term < min_objs, objv_term, min_objs)
                else:
                    min_objs = objv_term
        t_vals = [i for i in t if i is not None]
        subset_threshold = max(t_vals) if len(t_vals) > 0 else np.inf
        #min_name = min_name + '_' + str(subset_threshold)
        #min_name = 'objectives_subset'
        if min_objs is None:
            return (np.inf, np.inf) if direction == 'up' else (-np.inf, -np.inf)
        # TODO !!! -- take care of scaling
        objv_bounds = {min_name: {'min':0, 'max' :1}}
        print('objv_bounds', objv_bounds)
        l0 = subset_threshold if len(t_vals) > 0 else 0
        u0 = 1
        '''
        if len(t_vals) > 0:
            objv_bounds = {min_name: {'min':subset_threshold, 'max' :1}}
        else:
            objv_bounds = {min_name: {'min':0, 'max' :1}}
        print('objv_bounds', objv_bounds)
        '''
        r = self.optimize_single_objective(model_full_term_dict, min_name, None, min_objs, 
                epsilon, smlp_domain, eta_F_t, theta_radii_dict, alpha, beta, delta, solver_logic,
                scale_objectives, min_name, objv_bounds, sat_approx, sat_precision, save_trace, l0, u0)

        #print('r', r)
        c_up = r['threshold_up']; print('c_up', c_up)
        c_lo = r['threshold_lo']; print('c_lo', c_lo)
        assert c_lo != np.inf
        return c_lo, c_up
                
                       
    
    # pareto optimization
    def optimize_pareto_objectives(self, X:pd.DataFrame, y:pd.DataFrame, feat_names:list, resp_names:list, 
            model_full_term_dict:dict, objv_names:list, objv_exprs:list, alpha:smlp.form2, beta:smlp.form2, eta:smlp.form2, theta_radii_dict:dict,
            epsilon:float, smlp_domain:smlp.domain, delta:float, solver_logic:str, scale_objv:bool, data_scaler:str, 
            sat_approx=False, sat_precision=64, save_trace=False):
        assert X is not None
        assert y is not None
        assert epsilon > 0 and epsilon < 1
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs)
        scale_objectives = scale_objv and data_scaler != 'none'
        #assert scale_objectives
        # TODO !!!: when re-using a saved model, X and y are not available, need to adapt compute_objectives_terms
        # and computation of objv_bounds for that case -- say simulate the model, then compute objectives' values
        # for these simulation vectors, them compute the bounds on the objectives from simulation data
        objv_terms_dict, orig_objv_terms_dict, scaled_objv_terms_dict, objv_bounds = \
            self._modelTermsInst.compute_objectives_terms(objv_names, objv_exprs, 
                scale_objectives, feat_names, resp_names, X, y)
        #print('objv_terms_dict', objv_terms_dict); print('orig_objv_terms_dict', orig_objv_terms_dict); print('scaled_objv_terms_dict', scaled_objv_terms_dict)
        objv_count = len(objv_names)
        objv_enum = range(objv_count)
        # IDs of objectives whose bounds can still be potentially improved by at least epsilon;
        # in each iteration of the while loop below, the algo aims to improves a larges subset
        # of these objectives simultaneously, and updates K by droping objectoves that cannot be improved
        K = [i for i in range(objv_count)]; print('K', K)
        # vector of length objv_count with None at indexes in K -- IDs of objectives whise bounds
        # have been fixed. They are updated at the begining of each iteration of the while loop
        s = [None] * objv_count; print('s', s)
        direction = 'up' # TODO !!! add as argument to pareto or derive from bad / good value options
        eta_F_t_conj = eta
        while len(K) > 0:
            print('start of while iteration:')
            #  s <- {(j,s(j)) : j=1,...,k, j not in K} 
            s2 = [None] * objv_count;
            for j in objv_enum:
                if j not in K:
                    assert s[j] is not None
                    s2[j] = s[j]
            s = s2
            #print('s 1 ------------', s)        
            # /s <- s \cup {(j,b_s) : j in K}   
            #b_s = None # TODO !!!! compute b_s
            #c_lo, c_hi = self.approx_bounds(b_s, delta)
            c_lo, c_up = self.unbound_objectives_max_min_bounds(model_full_term_dict, objv_terms_dict, 
                s, smlp_domain, alpha, beta, eta, theta_radii_dict, epsilon, delta, solver_logic, direction,
                scale_objectives, objv_bounds, sat_approx, sat_precision, save_trace)
            #print('c_lo', c_lo, 'c_up', c_up)
            assert c_lo != np.inf
            for j in objv_enum:
                if j in K:
                    s[j] = c_lo
            #print('s 2 ==============', s)  
            # at least one of the objectives whose bounds there adjusted above cannot be improved
            # further (because we improved this objectives to the joint least uppur bound). We now
            # find out which of the objectoves cannot be improved frther so that they will be dropped
            # from K (there will be no further attempts to improve these objectives)
            KN = K.copy()
            for j in objv_enum:
                #print('++++++++++++++++j', j, 'K', K)
                if j in K:
                    K_pr = KN.copy()
                    K_pr.remove(j); print('K_pr', K_pr, 'K', K) 
                    t = [None] * objv_count;
                    for i in objv_enum:
                        if i in K_pr:
                            t[j] = s[j]; print('t', t)
                    #objv_bounds_in_search = dict([(objv_names, {'min':s[j], 'max':1}) for i in objv_enum])
                    #print('objv_bounds_in_search', objv_bounds_in_search)
                    self._opt_logger.info('Checking whether to fix objective {} on threshold {}...\n'.format(str(j), str(s[j])))
                    t_lo, t_up = self.unbound_objectives_max_min_bounds(model_full_term_dict, objv_terms_dict, 
                        t, smlp_domain, alpha, beta, eta, theta_radii_dict, epsilon, delta, solver_logic, direction, 
                        scale_objectives, objv_bounds, sat_approx, sat_precision, save_trace)
                    assert t_lo != np.inf
                    #print('t_lo', t_lo); print('t_up', t_up); print('epsilon', epsilon); print('s', s, 's(j)', s[j])
                    if t_lo < s[j] + epsilon:
                        self._opt_logger.info('Fixing objective {} on threshold {}...\n'.format(str(j), str(s[j])))
                        #K.remove(j); print('K after reduction', K)
                        # TODO !!! need this instead f prev line:  
                        KN.remove(j); print('KN after reduction', KN)
                    else:
                        self._opt_logger.info('Lower bounds of bjectives {} can be raised to threshold {}...\n'.
                                              format(str([i for i in KN if i != j]), str(t_lo)))
                        #assert False
                        print('break of while iteration:')
                        KN.remove(j); print('KN after reduction', KN)
                        K_prev = K
                        K = KN
                        break
                    #print('+++++++++++++++++++++K after iteration', K)
                print('end of while iteration:')
                K_prev = K
                K = KN
        print('end of while loop')
        #print('s', s)
        if scale_objectives:
            s_unscaled = [self._scalerTermsInst.unscale_constant_val(objv_bounds, objv_name, b) 
                for (objv_name, b) in zip(objv_names,s)]; #print('s_unscaled', s_unscaled)
            s_scaled_str = ["%.5f" % e for e in s]; print('s_scaled_str', s_scaled_str)
            s_orifinal_str = ["%.5f" % e for e in s_unscaled]; print('s_orifinal_str', s_orifinal_str)
            self._opt_logger.info('Pareto optimization completed with thresholds: ' + 
                '\n    Scaled to [0,1]: {}\n    Original  scale: {}\n'.format(s_scaled_str, s_orifinal_str))
            opt_conf = {}
            for i, (objv_name, objv_term) in enumerate(list(objv_terms_dict.items())):
                objv_expr = objv_exprs[i]
                if scale_objectives:
                    objv_epsn = epsilon
                else:
                    objv_epsn = self._scalerTermsInst.unscale_constant_val(objv_bounds, objv_names[i], epsilon)
                #print('objv_epsn', objv_epsn)
                opt_conf[objv_names[i]] = {'scaled':s[i], 'original':s_unscaled[i], 
                                           'max_in_data': objv_bounds[objv_names[i]]['max'],
                                           'min_in_data': objv_bounds[objv_names[i]]['min']}
                with open(self.optimization_results_file, 'w') as f: #json.dump(asrt_res_dict, f)
                    json.dump(opt_conf, f, indent='\t', cls=np_JSONEncoder)
        else:
            assert False # not supported
        
        return s
            
    # optimization of multiple objectives -- pareto optimization or optimization per objective
    # TODO !!!: X and y are used to estimate bounds on objectives from training data, and the latter is not
    #     available in model re-run mode. Need to estimate objectove bounds in a different way and pass to this
    #     function (and to smlp_tune() instead of passing X,y; The bounds on objectives are nt strictly necessary,
    #     any approximation may be used, but accurate approximation might reduce iterations count needed for
    #     computing optimal confoguurations (in optimize and tune modes)
    def smlp_optimize(self, algo, model, X, y, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, pareto, asrt_names, asrt_exprs, quer_names, quer_exprs, delta, epsilon, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        domain, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            algo, model, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, None, None, None, None, delta, epsilon, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)

        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            quer_res = self._queryInst.query_condition(model_full_term_dict, 'consistency_check', 'True', beta, 
                domain, eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision) 
            if quer_res['status'] == 'UNSAT':
                self._opt_logger.info('Model configuration optimization instance is inconsistent; aborting...')
                return
            
        if pareto:
            self.optimize_pareto_objectives(X, y, feat_names, resp_names, model_full_term_dict,
                objv_names, objv_exprs, alpha, beta, eta, theta_radii_dict, epsilon, domain, delta, solver_logic, scale_objv, data_scaler,
                sat_approx=True, sat_precision=64, save_trace=False)
        else:
            self.optimize_single_objectives(X, y, feat_names, resp_names, model_full_term_dict,
                objv_names, objv_exprs, alpha, beta, eta, theta_radii_dict, epsilon, domain, delta, solver_logic, scale_objv, data_scaler, 
                sat_approx=True, sat_precision=64, save_trace=False)

    # smlp tune mode that performs multi-objective optimization (pareto or per-objective) and einsures
    # that with the selected configuration of knobs all assertions are also satisfied (in addition to
    # any other model interface constraints or configuration stability constraints)
    def smlp_tune(self, algo, model, X, y, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, pareto, asrt_names, asrt_exprs, quer_names, quer_exprs, delta, epsilon, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict, solver_logic:str, vacuity:bool, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx=True, float_precision=64, data_bounds_json_path=None, bounds_factor=None, T_resp_bounds_csv_path=None):
        domain, model_full_term_dict, eta, alpha, beta = self._modelTermsInst.create_model_exploration_base_components(
            algo, model, model_features_dict, feat_names, resp_names, 
            objv_names, objv_exprs, asrt_names, asrt_exprs, None, None, delta, epsilon, 
            alph_expr, beta_expr, eta_expr, data_scaler, scale_feat, scale_resp, scale_objv, 
            float_approx, float_precision, data_bounds_json_path)

        if asrt_exprs is not None:
            assert asrt_names is not None
            asrt_forms_dict = dict([(asrt_name, self._smlpTermsInst.ast_expr_to_term(asrt_expr)) \
                    for asrt_name, asrt_expr in zip(asrt_names, asrt_exprs)])
            asrt_conj = self._smlpTermsInst.smlp_and_multi(list(asrt_forms_dict.values()))
        else:
            asrt_conj = smlp.true
        beta = self._smlpTermsInst.smlp_and(beta, asrt_conj) if beta != smlp.true else asrt_conj
        
        # instance consistency check (are the assumptions contradictory?)
        if vacuity:
            quer_res = self._queryInst.query_condition(model_full_term_dict, 'consistency_check', 'True', beta, 
                domain, eta, alpha, theta_radii_dict, delta, solver_logic, False, float_approx, float_precision) 
            if quer_res['status'] == 'UNSAT':
                self._opt_logger.info('Model querying instance is inconsistent; aborting...')
                return
                                                               
        if pareto:
            self.optimize_pareto_objectives(X, y, feat_names, resp_names, model_full_term_dict,
                objv_names, objv_exprs, alpha, beta, eta, theta_radii_dict, epsilon, domain, delta, solver_logic, scale_objv, data_scaler,
                sat_approx=True, sat_precision=64, save_trace=False)
        else:
            self.optimize_single_objectives(X, y, feat_names, resp_names, model_full_term_dict,
                objv_names, objv_exprs, alpha, beta, eta, theta_radii_dict, epsilon, domain, delta, solver_logic, scale_objv, data_scaler, 
                sat_approx=True, sat_precision=64, save_trace=False)