# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

# imports from SMLP modules
from smlp_py.smlp_logs import SmlpLogger
from smlp_py.smlp_utils import str_to_bool, np_JSONEncoder, model_features_sanity_check
from smlp_py.smlp_models import SmlpModels
from smlp_py.smlp_data import SmlpData
from smlp_py.smlp_subgroups import SubgroupDiscovery
from smlp_py.smlp_config import SmlpConfig
from smlp_py.smlp_doe import SmlpDoepy
from smlp_py.smlp_discretize import SmlpDiscretize
from smlp_py.smlp_terms import ModelTerms
from smlp_py.smlp_spec import SmlpSpec
from smlp_py.smlp_solver import SmlpSolver
from smlp_py.smlp_verify import SmlpVerify
from smlp_py.smlp_query import SmlpQuery
from smlp_py.smlp_optimize import SmlpOptimize

# Combining simulation results, optimization, uncertainty analysis, sequential experiments
# https://foqus.readthedocs.io/en/3.1.0/chapt_intro/index.html

class SmlpFlows:
    def __init__(self, argv):
        # data and model class instances
        self.dataInst = SmlpData()
        self.modelInst = SmlpModels()
        self.psgInst = SubgroupDiscovery()
        self.loggerInst = SmlpLogger() 
        self.configInst = SmlpConfig()
        self.doeInst = SmlpDoepy();
        self.discrInst = SmlpDiscretize()
        self.specInst = SmlpSpec()
        self.modelTernaInst = ModelTerms()
        self.modelTernaInst.set_smlp_spec_inst(self.specInst)
        self.solverInst = SmlpSolver()
        self.verifyInst = SmlpVerify()
        self.verifyInst.set_model_terms_inst(self.modelTernaInst)
        self.queryInst = SmlpQuery()
        self.queryInst.set_model_terms_inst(self.modelTernaInst)
        self.optInst = SmlpOptimize()
        self.optInst.set_model_terms_inst(self.modelTernaInst)
        self.optInst.set_smlp_query_inst(self.queryInst)
        
        # get args
        args_dict = self.modelInst.model_params_dict | \
                    self.dataInst.data_params_dict | \
                    self.loggerInst.logger_params_dict | \
                    self.doeInst.doepy_params_dict | \
                    self.discrInst.discr_params_dict | \
                    self.psgInst.get_subgroup_hparam_default_dict() |\
                    self.specInst.spec_params_dict | \
                    self.queryInst.query_params_dict | \
                    self.verifyInst.asrt_params_dict | \
                    self.optInst.opt_params_dict | \
                    self.solverInst.solver_params_dict #| \
                    
            
        self.args = self.configInst.args_dict_parse(argv, args_dict)
        self.log_file = self.configInst.report_file_prefix + '.txt'

        # set loggers
        self.logger = self.loggerInst.create_logger('smlp_logger', self.log_file, 
            self.args.log_level, self.args.log_mode, self.args.log_time)
        self.dataInst.set_logger(self.logger)
        self.modelInst.set_logger(self.logger)
        self.psgInst.set_logger(self.logger)
        self.doeInst.set_logger(self.logger)
        self.discrInst.set_logger(self.logger)
        self.specInst.set_logger(self.logger)
        self.optInst.set_logger(self.logger)
        self.verifyInst.set_logger(self.logger)
        self.queryInst.set_logger(self.logger)
        
        # set report and model files / file prefixes
        self.psgInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.dataInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.dataInst.set_model_file_prefix(self.configInst.model_file_prefix)
        self.modelInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.modelInst.set_model_file_prefix(self.configInst.model_file_prefix)
        self.optInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.optInst.set_model_file_prefix(self.configInst.model_file_prefix)
        self.verifyInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.verifyInst.set_model_file_prefix(self.configInst.model_file_prefix)
        self.queryInst.set_report_file_prefix(self.configInst.report_file_prefix)
        self.queryInst.set_model_file_prefix(self.configInst.model_file_prefix)

        # set spec file / spec
        self.modelTernaInst.set_spec_file(self.args.spec)
        self.dataInst.set_spec_inst(self.specInst)
        self.specInst.set_radii(self.args.radius_absolute, self.args.radius_relative)
        self.specInst.set_deltas(self.args.delta_absolute, self.args.delta_relative)
        #self.specInst.set_spec_witness_file(args.witness_file)
        
        
        # set external solver to SMLP
        self.solverInst.set_solver_path(self.args.solver_path)
        
        # ML model exploration modes. They require a spec file for model exploration.
        self.model_prediction_modes = ['train', 'predict']
        self.model_exploration_modes = ['optimize', 'synthesize', 'verify', 'query', 'optsyn', 'certify']
        self.supervised_modes = ['subgroups', 'discretize'] + self.model_prediction_modes + \
            self.model_exploration_modes
    
    # TODO !!!: is this the right place to define data_fname and new_data_fname and error_file ???
    @property
    def data_fname(self):
        if self.args.labeled_data is None:
            return None
        else:
            return self.args.labeled_data + '.csv'
        
    # new (unseen during training) data file name (including the directory path)
    @property
    def new_data_fname(self):
        if self.args.new_data is None:
            return None
        else:
            return self.args.new_data + '.csv'
        
    ''' not used currently
    # filename with full path for logging error message before aborting
    @property
    def error_file(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_error.txt'
    ''' 
    
    # The main function to run SMLP in all supported modes
    def smlp_flow(self):
        self.logger.info('Executing run_smlp.py script: Start')
        args = self.args
        # extract response and feature names
        if args.analytics_mode in self.supervised_modes:
            if args.response is None:
                if args.analytics_mode in self.model_exploration_modes:
                    resp_names = self.specInst.get_spec_responses
                else:
                    raise Exception('Response names should be provided')
            else:
                resp_names = args.response.split(',')
            if args.features is None:
                if args.analytics_mode in self.model_exploration_modes:
                    feat_names = self.specInst.get_spec_features
                else:
                    feat_names = None
            else:
                feat_names = args.features.split(',')
        
        # doepy usage https://doepy.readthedocs.io/en/latest/
        if args.analytics_mode == 'doe':
            self.logger.info('Running SMLP in mode "{}": Start'.format(args.analytics_mode))
            doe_out_df = self.doeInst.sample_doepy(args.doe_algo, args.doe_spec_file, args.doe_num_samples, 
                self.configInst.report_file_prefix, args.doe_prob_distribution, args.doe_design_resolution, 
                args.doe_central_composite_center, args.doe_central_composite_face, 
                args.doe_central_composite_alpha, args.doe_box_behnken_centers); #print('doe_out_df\n', doe_out_df); 
            self.logger.info('Running SMLP in mode "{}": End'.format(args.analytics_mode))
            self.logger.info('Executing run_smlp.py script: End')
            return None

        if args.analytics_mode == 'discretize':
            self.logger.info('Running SMLP in mode "{}": Start'.format(args.analytics_mode))
            X, y, feat_names, resp_names, feat_names_dict = self.dataInst.preprocess_data(self.data_fname, 
                feat_names, resp_names, None, args.impute_responses, 'training', 
                args.positive_value, args.negative_value, args.response_to_bool)
            self.discrInst.smlp_discretize_df(X, algo=args.discretization_algo, 
                bins=args.discretization_bins, labels=args.discretization_labels,
                result_type=args.discretization_type)
            self.logger.info('Running SMLP in mode "{}": End'.format(args.analytics_mode))
            self.logger.info('Executing run_smlp.py script: End')
        
        if args.analytics_mode == 'subgroups':
            self.logger.info('Running SMLP in mode "{}": Start'.format(args.analytics_mode))
            #from smlp.subgroups import SubgroupDiscovery
            #instSubgroup = SubgroupDiscovery()
            X, y, feat_names, resp_names, feat_names_dict = self.dataInst.preprocess_data(self.data_fname, 
                feat_names, resp_names, None, args.impute_responses, 'training', 
                args.positive_value, args.negative_value, args.response_to_bool)
            #data = pd.concat([X,y], axis=1); print('data\n',data)
            fs_ranking_df, fs_summary_df, results_dict = self.psgInst.smlp_subgroups(X, y, resp_names, 
                args.positive_value, args.negative_value, args.psg_quality_target, args.psg_max_dimension, 
                args.psg_top_ranked, args.interactive_plots)
            #print('fs_ranking_df\n', fs_ranking_df); 
            #for col in results_dict['num1']['psg_df'].columns.tolist():
            #    print('\ncol ', col, '\n', results_dict['num1']['psg_df'][col]) #
            self.logger.info('Running SMLP in mode "{}": End'.format(args.analytics_mode))
            self.logger.info('Executing run_smlp.py script: End')
            return None
        
        if args.analytics_mode in self.model_exploration_modes or \
            (args.model == 'system' and args.analytics_mode in self.model_prediction_modes):
            # We want to set to SmlpSpec object self.specInst expressions of assertions, queries, optimization 
            # objectives, in order to compute variables input feature names that way depend on. This is to
            # ensure that these variables are used in building models analyses in the above model exploration
            # modes. Usueally objectives, assertions, queries do not refer to inpputs and knobs but currently
            # usage of inputs and knobs are not disallowed in (objectives, assertions, and queries thus we want 
            # them varibles inputs/knobs occuring in them to be part of the trained models. Not all model 
            # xploration modes require all these components (objectives, assertions, queries) but their computation
            # is not a real overhead and we prefer to keep code readable and computate all these expressions.
            # When smlp mode is optimize, objectives must be defined. If they are not provided, the default is to use
            # the responses as objectives, and the names of objectives are names of the responses prefixed bu 'objv_'.
            alpha_global_expr, beta_expr, theta_radii_dict, delta_dict, asrt_names, asrt_exprs, quer_names, quer_exprs, \
                config_dict, witn_dict, objv_names, objv_exprs, syst_expr_dict = self.specInst.get_spec_component_exprs(
                args.alpha, args.beta, args.delta_absolute, args.delta_relative, args.assertions_names, args.assertions_expressions, 
                args.query_names, args.query_expressions, args.objectives_names, args.objectives_expressions,
                resp_names, self.dataInst.commandline_condition_separator)
            
            # make sure response names and objectives' names are different
            if args.analytics_mode in ['optimize', 'optsyn']:
                assert resp_names is not None
                for resp_name in resp_names:
                    if resp_name in objv_names:
                        raise Exception('Response (output) names must be different from names of objectives')
            self.dataInst.set_spec_inst(self.specInst)
            
        # prepare data for model training
        if args.analytics_mode in self.model_prediction_modes + self.model_exploration_modes:
            self.logger.info('Running SMLP in mode "{}": Start'.format(args.analytics_mode))
            self.logger.info('PREPARE DATA FOR MODELING')    
            X, y, X_train, y_train, X_test, y_test, X_new, y_new, mm_scaler_feat, mm_scaler_resp, \
            levels_dict, model_features_dict, feat_names, resp_names = self.dataInst.process_data(
                self.configInst.report_file_prefix,
                self.data_fname, self.new_data_fname, True, args.split_test, feat_names, resp_names, args.keep_features,
                args.train_first_n, args.train_random_n, args.train_uniform_n, args.interactive_plots, args.data_scaler,
                args.scale_features, args.scale_responses, args.impute_responses, args.mrmr_feat_count_for_prediction, 
                args.positive_value, args.negative_value, args.response_to_bool, args.save_model, args.use_model)

            # sanity check that the order of features in model_features_dict, feat_names, X_train, X_test, X is 
            # the same; this is mostly important for model exploration modes 
            model_features_sanity_check(model_features_dict, feat_names, X_train, X_test, X)
            
            # model training, validation, testing, prediction on training, labeled and new data (when available)
            if args.model == 'system':
                model = syst_expr_dict
            else:
                model = self.modelInst.build_models(args.model, X, y, X_train, y_train, X_test, y_test, X_new, y_new,
                    resp_names, mm_scaler_feat, mm_scaler_resp, levels_dict, model_features_dict, 
                    self.modelInst.get_hyperparams_dict(args, args.model), args.interactive_plots, args.seed, 
                    args.sample_weights_coef, args.save_model, args.use_model, args.model_per_response, 
                    self.configInst.model_rerun_config)
            
            # sanity check that the order of features in model_features_dict, feat_names, X_train, X_test, X is 
            # the same; this is mostly important for model exploration modes 
            model_features_sanity_check(model_features_dict, feat_names, X_train, X_test, X)
            
            if args.analytics_mode in self.model_prediction_modes:
                self.logger.info('Running SMLP in mode "{}": End'.format(args.analytics_mode))
                self.logger.info('Executing run_smlp.py script: End')
                return model
        
        # sanity check that the order of features in model_features_dict, feat_names, X_train, X_test, X is 
        # the same; this is mostly important for model exploration modes 
        model_features_sanity_check(model_features_dict, feat_names, X_train, X_test, X)
        
        if args.analytics_mode in self.model_exploration_modes:
            if args.analytics_mode == 'verify':
                if True or len(self.specInst.get_spec_knobs)> 0:
                    if config_dict is None:
                        configuration = self.specInst.sanity_check_verification_spec(); #print('configuration', configuration)
                        config_dict = dict([(asrt_name, configuration) for asrt_name in asrt_names]); #print('config_dict', config_dict)
                    self.queryInst.smlp_verify(syst_expr_dict, args.model, model, 
                        model_features_dict, feat_names, resp_names, asrt_names, asrt_exprs, config_dict,
                        delta_dict, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                        args.solver_logic, args.vacuity_check, 
                        args.data_scaler, args.scale_features, args.scale_responses,
                        args.approximate_fractions, args.fraction_precision, 
                        self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
                else:
                    self.verifyInst.smlp_verify(syst_expr_dict, args.model, model, 
                        #self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                        model_features_dict, feat_names, resp_names, asrt_names, asrt_exprs, alpha_global_expr, 
                        args.solver_logic, args.vacuity_check,
                        args.data_scaler, args.scale_features, args.scale_responses, 
                        args.approximate_fractions, args.fraction_precision,
                        self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            elif args.analytics_mode == 'certify':
                if witn_dict is None:
                    witness = self.specInst.sanity_check_certification_spec()
                    witn_dict = dict([(quer_name, witness) for quer_name in quer_names])
                #print('witness', witness, 'witn_dict', witn_dict)
                self.queryInst.smlp_certify(syst_expr_dict, args.model, model, #False, #universal
                    #self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                    model_features_dict, feat_names, resp_names, quer_names, quer_exprs, witn_dict,
                    delta_dict, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                    args.solver_logic, args.vacuity_check, 
                    args.data_scaler, args.scale_features, args.scale_responses, #args.scale_objectives, 
                    args.approximate_fractions, args.fraction_precision,
                    self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            elif args.analytics_mode == 'query':
                self.queryInst.smlp_query(syst_expr_dict, args.model, model, 
                    #self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                    model_features_dict, feat_names, resp_names, quer_names, quer_exprs, 
                    delta_dict, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                    args.solver_logic, args.vacuity_check, 
                    args.data_scaler, args.scale_features, args.scale_responses, args.scale_objectives, 
                    args.approximate_fractions, args.fraction_precision,
                    self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            elif args.analytics_mode == 'synthesize':
                self.queryInst.smlp_synthesize(syst_expr_dict, args.model, model,
                    #self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                    model_features_dict, feat_names, resp_names, asrt_names, asrt_exprs,
                    delta_dict, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                    args.solver_logic, args.vacuity_check, 
                    args.data_scaler, args.scale_features, args.scale_responses, 
                    args.sat_thresholds, args.approximate_fractions, args.fraction_precision,
                    self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            elif args.analytics_mode == 'optimize':
                self.optInst.smlp_optimize(syst_expr_dict, args.model, model,
                    self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                    model_features_dict, feat_names, resp_names, objv_names, objv_exprs, args.optimize_pareto, 
                    quer_names, quer_exprs, 
                    delta_dict, args.epsilon, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                    args.solver_logic, args.vacuity_check, 
                    args.data_scaler, args.scale_features, args.scale_responses, args.scale_objectives, 
                    args.sat_thresholds, args.approximate_fractions, args.fraction_precision,
                    self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            elif args.analytics_mode == 'optsyn':
                self.optInst.smlp_optsyn(syst_expr_dict, args.model, model, 
                    self.dataInst.unscaled_training_features, self.dataInst.unscaled_training_responses, 
                    model_features_dict, feat_names, resp_names, objv_names, objv_exprs, args.optimize_pareto, 
                    asrt_names, asrt_exprs, quer_names, quer_exprs, 
                    delta_dict, args.epsilon, alpha_global_expr, beta_expr, args.eta, theta_radii_dict, 
                    args.solver_logic, args.vacuity_check, 
                    args.data_scaler, args.scale_features, args.scale_responses, args.scale_objectives, 
                    args.sat_thresholds, args.approximate_fractions, args.fraction_precision,
                    self.dataInst.data_bounds_file, bounds_factor=None, T_resp_bounds_csv_path=None)
            self.logger.info('Running SMLP in mode "{}": End'.format(args.analytics_mode))
            self.logger.info('Executing run_smlp.py script: End')
