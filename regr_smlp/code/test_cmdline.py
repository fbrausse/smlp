#!/usr/bin/env python3

import pytest
import subprocess

from lib import *

@pytest.mark.toy
class Test1(CmdTestCase):
	'''
	basic dt_caret training and test on labeled data with single numeric response
	'''

	nr = 1
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'train', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test2(CmdTestCase):
	'''
	basic rf_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 2
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '15', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test3(CmdTestCase):
	'''
	basic poly_sklearn prediction test on labeled and new data with numeric response in training/test data only
	'''

	nr = 3
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_unlabeled'
	args = ['-mode', 'predict', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'poly_sklearn', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test4(CmdTestCase):
	'''
	basic nn_keras prediction test on labeled and new data with numeric labels and one response
	'''

	nr = 4
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_weights_precision', '2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f']

@pytest.mark.toy
class Test5(CmdTestCase):
	'''
	basic dt_caret prediction test on labeled and new data with numeric labels
	'''

	nr = 5
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-save_model', 't', '-use_model', 'f', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test6(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 6
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test7(CmdTestCase):
	'''
	basic rf_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 7
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '15', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test8(CmdTestCase):
	'''
	basic nn_keras prediction test on labeled and new data with numeric labels and two responses
	'''

	nr = 8
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-log_time', 'f']

@pytest.mark.toy
class Test9(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 9
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 't', '-model_name', 'test20_model', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-save_config', 't', '-save_model_config', 't']

@pytest.mark.toy
class Test10(CmdTestCase):
	'''
	basic et_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 10
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '15', '-et_sklearn_bootstrap', 'f', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test11(CmdTestCase):
	'''
	basic poly_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 11
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'poly_sklearn', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test12(CmdTestCase):
	'''
	EV-SI real life dt_sklearn predict test on labeled and new data with numeric labels
	'''

	nr = 12
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'train', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test13(CmdTestCase):
	'''
	EV-SI real life nn_keras prediction test on labeled and new data with numeric labels
	'''

	nr = 13
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'train', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test14(CmdTestCase):
	'''
	EV-SI real life poly_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 14
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'train', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'poly_sklearn', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test15(CmdTestCase):
	'''
	basic dt_caret prediction test from saved model on new data with numeric labels
	'''

	nr = 15
	data = 'Test5_smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test16(CmdTestCase):
	'''
	basic nn_keras prediction test from saved model on new data with numeric labels and two responses
	'''

	nr = 16
	data = 'Test8_smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test17(CmdTestCase):
	'''
	basic poly_sklearn prediction test from saved model on new data with numeric labels and two responses
	'''

	nr = 17
	data = 'Test11_smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'poly_sklearn', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test18(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels and saving model using name specified through model_name option - adapts Test6
	'''

	nr = 18
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 't', '-use_model', 'f', '-model_name', 'test19_model', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test19(CmdTestCase):
	'''
	basic dt_sklearn prediction test using a model saved under a name specified through model_name  option on new data with numeric labels
	'''

	nr = 19
	data = 'test19_model'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test20(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels
	'''

	nr = 20
	data = 'test20_model'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 'f', '-use_model', 't', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test21(CmdTestCase):
	'''
	test for illegal symbols in column names
	'''

	nr = 21
	data = 'smlp_toy_num_metasymbol_mult_reg'
	new_data = 'smlp_toy_num_metasymbol_mult_reg_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'PF ,|PF |', '-model', 'poly_sklearn', '-save_model', 't', '-use_model', 'f', '-model_name', 'test22_model', '-pred_plots', 't', '-resp_plots', 't', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test22(CmdTestCase):
	'''
	test for illegal symbols in column names
	'''

	nr = 22
	data = 'test22_model'
	new_data = 'smlp_toy_num_metasymbol_mult_reg_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'PF ,|PF |', '-model', 'poly_sklearn', '-save_model', 'f', '-use_model', 't', '-pred_plots', 't', '-resp_plots', 't', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test23(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels and saving model using name specified through model_name option - adapts Test6
	'''

	nr = 23
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 't', '-use_model', 'f', '-model_name', 'test24_model', '-model_per_response', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test24(CmdTestCase):
	'''
	basic dt_sklearn prediction test using a model saved under a name specified through model_name  option on new data with numeric labels
	'''

	nr = 24
	data = 'test24_model'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 'f', '-use_model', 't', '-model_per_response', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test25(CmdTestCase):
	'''
	basic dt_sklearn prediction test on labeled and new data with numeric labels and saving model using name specified through model_name option - adapts Test6
	'''

	nr = 25
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 't', '-use_model', 'f', '-model_name', 'test26_model', '-mrmr_pred', '2', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test26(CmdTestCase):
	'''
	basic dt_sklearn prediction test using a model saved under a name specified through model_name  option on new data with numeric labels
	'''

	nr = 26
	data = 'test26_model'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '2', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test27(CmdTestCase):
	'''
	checks nn_keras prediction with nn_keras_seq_api t
	adapts test 4 which uses nn_keras_seq_api f
	'''

	nr = 27
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't']

@pytest.mark.toy
class Test28(CmdTestCase):
	'''
	checks nn_keras prediction with sw_coef 0.8 and functional API
	adapts test 4
	'''

	nr = 28
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-sw_coef', '0.8']

@pytest.mark.toy
class Test29(CmdTestCase):
	'''
	basic  test for subgroup discovery for pass-fail responses
	'''

	nr = 29
	data = 'smlp_toy_cls_metasymbol_colnames_mult'
	new_data = ''
	args = ['-mode', 'subgroups', '-psg_dim', '3', '-psg_top', '10', '-resp', 'PF 1,PF#', '-plots', 't', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test30(CmdTestCase):
	'''
	basic  test for subgroup discovery for numric responses
	'''

	nr = 30
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'subgroups', '-psg_dim', '3', '-psg_top', '10', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-plots', 't', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test31(CmdTestCase):
	'''
	testing resp2b in subgroup discovery mode
	'''

	nr = 31
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'subgroups', '-psg_dim', '3', '-psg_top', '10', '-resp', 'y1,y2', '-resp2b', 'y1<6;y2>6', '-feat', 'x,p1,p2', '-plots', 't', '-seed', '10', '-log_time', 'f', '-save_config', 't']

@pytest.mark.toy
class Test32(CmdTestCase):
	'''
	test reusing saved model by using configuration file
	'''

	nr = 32
	data = 'test20_model'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-config', 'test20_model_rerun_model_config.json']

@pytest.mark.toy
class Test33(CmdTestCase):
	'''
	testing -config option with subgroups mode
	'''

	nr = 33
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-config', 'Test31_smlp_toy_num_resp_mult_args_config.json']

@pytest.mark.real
class Test34(CmdTestCase):
	'''
	doe test with four levels with full_factorial method
	'''

	nr = 34
	data = 'doe_four_levels_real'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'full_factorial', '-log_time', 'f']

@pytest.mark.real
class Test35(CmdTestCase):
	'''
	doe test with four levels with plackett_burman
	'''

	nr = 35
	data = 'doe_four_levels_real'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'plackett_burman', '-log_time', 'f']

@pytest.mark.real
@pytest.mark.test
class Test36(CmdTestCase):
	'''
	doe test with four levels with sukharev_grid
	'''

	nr = 36
	data = 'doe_four_levels_real'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'sukharev_grid', '-doe_samples', '125', '-log_time', 'f']

@pytest.mark.real
class Test37(CmdTestCase):
	'''
	doe test with four levels with box_behnken
	'''

	nr = 37
	data = 'doe_three_levels_real_nan'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'box_behnken', '-log_time', 'f']

@pytest.mark.real
class Test38(CmdTestCase):
	'''
	doe test with four levels with box_wilson
	'''

	nr = 38
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'box_wilson', '-doe_cc_face', 'ccc', '-doe_cc_alpha', 'r', '-doe_cc_center', '2,3', '-log_time', 'f']

@pytest.mark.real
class Test39(CmdTestCase):
	'''
	doe test with four levels with latin_hypercube
	'''

	nr = 39
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'latin_hypercube', '-doe_prob_distr', 'Exponential', '-doe_samples', '30', '-log_time', 'f']

@pytest.mark.real
class Test40(CmdTestCase):
	'''
	doe test with four levels with latin_hypercube_space_filling
	'''

	nr = 40
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'latin_hypercube_sf', '-doe_samples', '20', '-log_time', 'f']

@pytest.mark.real
class Test41(CmdTestCase):
	'''
	doe test with four levels with random_k_means
	'''

	nr = 41
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'random_k_means', '-doe_samples', '20', '-log_time', 'f']

@pytest.mark.real
class Test42(CmdTestCase):
	'''
	doe test with four levels with maximin_reconstruction
	'''

	nr = 42
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'maximin_reconstruction', '-doe_samples', '20', '-log_time', 'f']

@pytest.mark.real
class Test43(CmdTestCase):
	'''
	doe test with four levels with halton_sequence
	'''

	nr = 43
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'halton_sequence', '-doe_samples', '20', '-log_time', 'f']

@pytest.mark.real
class Test44(CmdTestCase):
	'''
	doe test with four levels with uniform_random_matrix
	'''

	nr = 44
	data = 'doe_two_levels'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'uniform_random_matrix', '-doe_samples', '20', '-log_time', 'f']

@pytest.mark.real
class Test45(CmdTestCase):
	'''
	doe test with four levels with fractional_factorial
	'''

	nr = 45
	data = 'doe_two_levels_real'
	new_data = ''
	args = ['-mode', 'doe', '-doe_algo', 'fractional_factorial', '-doe_resolution', '5', '-log_time', 'f']

@pytest.mark.toy
class Test46(CmdTestCase):
	'''
	tests options -pos_val and -neg_val
	'''

	nr = 46
	data = 'smlp_toy_pf_mult'
	new_data = 'smlp_toy_pf_mult'
	args = ['-mode', 'predict', '-resp', 'PF,PF1', '-model', 'poly_sklearn', '-save_model', 't', '-save_model_config', 'f', '-use_model', 'f', '-model_name', 'test47_model', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test47(CmdTestCase):
	'''
	tests options -pos_val and -neg_val when re-using saved model
	'''

	nr = 47
	data = 'test47_model'
	new_data = 'smlp_toy_pf_mult'
	args = ['-mode', 'predict', '-resp', 'PF,PF1', '-model', 'poly_sklearn', '-save_model', 'f', '-use_model', 't', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test48(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 48
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test49(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 49
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'quantile', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'category', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test50(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 50
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'kmeans', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'ordered', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
@pytest.mark.test
class Test51(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 51
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'jenks', '-discr_bins', '6', '-discr_labels', 'f', '-discr_type', 'integer', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test52(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 52
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'jenks', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'ordered', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test53(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 53
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'ordinals', '-discr_bins', '6', '-discr_labels', 'f', '-discr_type', 'integer', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test54(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 54
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'ordinals', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test55(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 55
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'ranks', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'category', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test56(CmdTestCase):
	'''
	tests discretization options
	'''

	nr = 56
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'discretize', '-resp', 'PF,PF1', '-discr_algo', 'ranks', '-discr_bins', '6', '-discr_labels', 'f', '-discr_type', 'object', '-data_scaler', 'none', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test57(CmdTestCase):
	'''
	basic dt_sklearn assertion verfication test with numeric labels and integer grid as domain
	'''

	nr = 57
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'y1*2+x0<=5 and y1<=10;-2*y2-1<10-x2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test58(CmdTestCase):
	'''
	basic dt_sklearn optimization test with numeric labels and integer grid as domain and without scaling objectives
	'''

	nr = 58
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult.spec', '-objv_names', 'objv_y1,objv_y2', '-objv_exprs', 'y1;y2', '-epsilon', '0.01', '-delta_rel', '0.01', '-data_scaler', 'none', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test59(CmdTestCase):
	'''
	basic nn_keras assertion verification test for functional nn_keras model
	'''

	nr = 59
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
@pytest.mark.test
class Test60(CmdTestCase):
	'''
	basic nn_keras assertion verification test for functional nn_keras model
	'''

	nr = 60
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test61(CmdTestCase):
	'''
	tests verificaion mode for NN with nn_keras_seq_api f
	'''

	nr = 61
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-nn_keras_epochs', '100', '-save_model_config', 'f', '--spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'not(x2<y1*2+x0<=(x1+1)>5 and y1<=10);-2*y2-1<10-x2 and x2>5 and x2<8', '-vacuity', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_seq_api', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test62(CmdTestCase):
	'''
	tests verificaion mode for NN with nn_keras_seq_api t
	'''

	nr = 62
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-nn_keras_epochs', '100', '-nn_keras_seq_api', 't', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'not(x2<y1*2+x0<=(x1+1)>5 and y1<=10);-2*y2-1<10-x2 and x2>5 and x2<8', '-vacuity', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test63(CmdTestCase):
	'''
	basic dt_sklearn assertion verification test on data with numeric labels
	'''

	nr = 63
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test63_model', '-spec', 'smlp_toy_num_resp_mult_y1_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x/2+y1>4.3;(y1+p2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.real
class Test64(CmdTestCase):
	'''
	basic dt_sklearn  assertion verification test on data with one numeric response
	'''

	nr = 64
	data = 'test63_model'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 't', '-spec', 'smlp_toy_num_resp_mult_y1_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x/2+y1>4.3;(y1+p2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test65(CmdTestCase):
	'''
	basic dt_sklearn assertion verification test on data with numeric labels
	'''

	nr = 65
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test65_model', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x0**2+y1>4.3;(y1+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.real
class Test66(CmdTestCase):
	'''
	basic dt_sklearn  assertion verification test on data with one numeric response
	'''

	nr = 66
	data = 'test65_model'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x0**2+y1>4.3;(y1+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test67(CmdTestCase):
	'''
	basic dt_sklearn assertion verification test on data with numeric labels
	'''

	nr = 67
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-model_per_response', 't', '-save_model', 't', '-use_model', 'f', '-model_name', 'test67_model', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x0**2+y1>4.3;(y1+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.real
class Test68(CmdTestCase):
	'''
	basic dt_sklearn  assertion verification test on data with one numeric response
	'''

	nr = 68
	data = 'test67_model'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-model_per_response', 't', '-save_model', 'f', '-use_model', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x0**2+y1>4.3;(y1+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test69(CmdTestCase):
	'''
	nn_keras verification test with model_per_response training
	'''

	nr = 69
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model', 't', '-use_model', 'f', '-model_name', 'test69_model', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '(y2**3+p2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.real
class Test70(CmdTestCase):
	'''
	nn_keras verification test with re-using saved model_per_response trained model
	'''

	nr = 70
	data = 'test69_model'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model', 'f', '-use_model', 't', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '(y2**3+p2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test71(CmdTestCase):
	'''
	nn_keras verification test with model_per_response training
	'''

	nr = 71
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model', 't', '-use_model', 'f', '-model_name', 'test71_model', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '(y1**3+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.real
class Test72(CmdTestCase):
	'''
	nn_keras verification test with re-using saved model_per_response trained model
	'''

	nr = 72
	data = 'test71_model'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model', 'f', '-use_model', 't', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '(y2**3+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test73(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	'''

	nr = 73
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test73_model', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test74(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model. with separate model for each response
	'''

	nr = 74
	data = 'test73_model'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test75(CmdTestCase):
	'''
	verification test run using model_rerun config covering the case when mrmr selcts only a subset of features specified through the command line or config file
	'''

	nr = 75
	data = 'test73_model'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-config', 'test73_model_rerun_model_config.json']

@pytest.mark.toy
class Test76(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	'''

	nr = 76
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test76_model', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.real
class Test77(CmdTestCase):
	'''
	verification test run using model_rerun config covering the case when mrmr selcts only a subset of features specified through the command line or config file
	'''

	nr = 77
	data = 'test76_model'
	new_data = ''
	args = ['-config', 'test76_model_rerun_model_config.json']

@pytest.mark.toy
class Test78(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	with one model for all responses
	'''

	nr = 78
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test78_model', '-mrmr_pred', '1', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'y1==9;y2>0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test79(CmdTestCase):
	'''
	basic test in query mode to test stability (theta) and guard (eta) constraint generation
	'''

	nr = 79
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'query', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult.spec', '-quer_names', 'query1,query2,query3', '-quer_exprs', '(y2**3+p2)/2<6;y1>=9;y2<0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.test
class Test80(CmdTestCase):
	'''
	basic dt_sklearn single objective optimization test with numeric labels and integer grid as domain and with scaling objectives
	'''

	nr = 80
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1', '-objv_exprs', '(y1+y2)/2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test81(CmdTestCase):
	'''
	basic dt_sklearn single objective optimization test with numeric labels and integer grid as domain and with scaling objectives
	'''

	nr = 81
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1', '-objv_exprs', '(y1+y2)/2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test82(CmdTestCase):
	'''
	basic dt_sklearn single objective optimization test with numeric labels and integer grid as domain and with scaling objectives
	'''

	nr = 82
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1,objv2,objv3', '-objv_exprs', '(y1+y2)/2;y1;y2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test83(CmdTestCase):
	'''
	basic dt_sklearn multi objective pareto optimization test with numeric labels and integer grid as domain and with scaling objectives
	'''

	nr = 83
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps.spec', '-data_scaler', 'min_max', '-beta', 'y1>7 and y2>6', '-objv_names', 'obj1,objv2,objv3', '-objv_exprs', '(y1+y2)/2;y1/2-y2;y2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test84(CmdTestCase):
	'''
	tests global alpha constraints specified using option -alpha on inputs
	'''

	nr = 84
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-alpha', 'x2==7.0 and x0==0 and x1==2.5', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test85(CmdTestCase):
	'''
	tests alpha and eta constraints specified in command line in optimization mode
	'''

	nr = 85
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1,objv2', '-objv_exprs', '(y1+y2)/2;y1', '-alpha', 'p2<5 and x==10 and  x<12', '-eta', 'p1==4', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test86(CmdTestCase):
	'''
	tests alpha
	beta and eta constraints specified in command line in optimization mode
	'''

	nr = 86
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1,objv2', '-objv_exprs', '(y1+y2)/2;y1', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+p2)/2<6;y1>=9;y2<0', '-alpha', 'p2<5 and x==10 and  x<12', '-eta', 'p1==4', '-epsilon', '0.05', '-delta_rel', '0.01', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test87(CmdTestCase):
	'''
	tests global alpha constraints and assertions specified in spec file
	equivalent to test 84 where the same alpha and assertions are specified in command line
	'''

	nr = 87
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_alpha_asrt_verify.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test88(CmdTestCase):
	'''
	basic dt_sklearn multi objective pareto optimization test with beta and objectives specified in spec file
	must give same results as test 83 where same beta and objectives is specified in command line
	'''

	nr = 88
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test89(CmdTestCase):
	'''
	basic test in query mode to test stability (theta) and guard (eta) constraint generation
	'''

	nr = 89
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'query', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_query.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test90(CmdTestCase):
	'''
	test to detect contradictory constraints in optsyn mode
	'''

	nr = 90
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn_vacuous.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test91(CmdTestCase):
	'''
	test to detect contradictory constraints in optimization mode due to contradictory alpha global and alpha bounds constraints on FMAX_xyx
	'''

	nr = 91
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'query', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_query_vacuous.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test92(CmdTestCase):
	'''
	test to detect contradictory constraints in verification mode
	'''

	nr = 92
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_verify_vacuous.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test93(CmdTestCase):
	'''
	basic test for mode optsyn
	'''

	nr = 93
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test94(CmdTestCase):
	'''
	basic test for rf_sklearn in model exploration mode optsyn
	'''

	nr = 94
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.test
class Test95(CmdTestCase):
	'''
	basic test for dt_caret in model exploration mode optsyn
	'''

	nr = 95
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-save_model', 'f', '-use_model', 'f', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test96(CmdTestCase):
	'''
	basic test for rf_sklearn in model exploration mode optsyn
	'''

	nr = 96
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_caret', '-save_model', 'f', '-use_model', 'f', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test97(CmdTestCase):
	'''
	basic test for rf_sklearn in model exploration mode optsyn
	'''

	nr = 97
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'query', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_query.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test98(CmdTestCase):
	'''
	basic test for et_caret in model exploration mode optsyn
	'''

	nr = 98
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_caret', '-save_model', 'f', '-use_model', 'f', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test99(CmdTestCase):
	'''
	testing that the response and feature names can be taken from spec file in model exploration modes when the responses and/or features are not specified in the command line
	'''

	nr = 99
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test100(CmdTestCase):
	'''
	basic test for sat_threshold option enabing usage of objectve values in SAT assignments that prove optimization thresholds
	'''

	nr = 100
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test101(CmdTestCase):
	'''
	basic test in certify mode to test stability (theta) and guard (eta) constraint generation
	'''

	nr = 101
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test101_model', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_witness.spec', '-quer_names', 'query1,query2,query3', '-quer_exprs', '(y2**3+p2)/2<6;y1>=9;y2<20', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.real
class Test102(CmdTestCase):
	'''
	basic test in certify mode to test stability (theta) and guard (eta) constraint generation
	'''

	nr = 102
	data = 'test101_model'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 't', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_witness.spec', '-quer_names', 'query1,query2,query3', '-quer_exprs', '(y2**3+p2)/2<6;y1>=9;y2<20', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test103(CmdTestCase):
	'''
	
	basic test in certify mode to test one valid witness and two conflicting witnesses for queries that are constant true
	'''

	nr = 103
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 't', '-use_model', 'f', '-model_name', 'test103_model', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_certify_witness.spec', '-quer_names', 'valid_candidate,grid_conflict,range_conflict', '-quer_exprs', 'True;True;True', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.test
@pytest.mark.xfail(True, reason='wrong spec due to singleton value',
                   raises=subprocess.CalledProcessError, strict=True)
class Test104(CmdTestCase):
	'''
	assertion verfication test with wrong spec that does not assign a single value using a singleton grid or range with equal max and min
	'''

	nr = 104
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'y1*2+x<=5 and y1<=10;-2*y2-1<10-p2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test105(CmdTestCase):
	'''
	basic dt_sklearn assertion verfication test with numeric labels and integer grid as domain
	'''

	nr = 105
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_stable_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'y1*2+x<=5 and y1<=10;-2*y2-1<10-p2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test106(CmdTestCase):
	'''
	test for verification mode to check that eta contraints are not contradictory and as otherwise verification problem is not well defined
	'''

	nr = 106
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_unsat_eta_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'y1*2+x<=5 and y1<=10;-2*y2-1<10-p2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.xfail(True, reason='contradictory eta contraints',
                   raises=subprocess.CalledProcessError, strict=True)
class Test107(CmdTestCase):
	'''
	test for verification mode to check that eta contraints are not contradictory and as otherwise verification problem is not well defined
	'''

	nr = 107
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_beta_verify.spec', '-asrt_names', 'asrt_y1,asrt_y2', '-asrt_expr', 'y1*2+x<=5 and y1<=10;-2*y2-1<10-p2', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test108(CmdTestCase):
	'''
	basic test for dt_sklearn in model exploration mode synthesize where synthesis succeeds
	'''

	nr = 108
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'synthesize', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_synthesize.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test109(CmdTestCase):
	'''
	basic test for mode synthesize where synthesis fails
	'''

	nr = 109
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'synthesize', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_cannot_synthesize.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test110(CmdTestCase):
	'''
	smlp toy basic example for predict mode from SMLP user manual
	'''

	nr = 110
	data = 'smlp_toy_basic'
	new_data = 'smlp_toy_basic_pred_unlabeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'poly_sklearn', '-save_model', 't', '-model_name', 'test110_model', '-save_model_config', 't', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test111(CmdTestCase):
	'''
	smlp toy basic test to rerun saved model using the model rerun config file saved during model training
	'''

	nr = 111
	data = 'test110_model'
	new_data = 'smlp_toy_basic_pred_unlabeled'
	args = ['-config', 'test110_model_rerun_model_config.json']

@pytest.mark.toy
class Test112(CmdTestCase):
	'''
	smlp toy basic test from SMLP manual
	to rerun saved model without using the model rerun config file saved during model training and directly adding required options to command that match option values used during training
	'''

	nr = 112
	data = 'test110_model'
	new_data = 'smlp_toy_basic_pred_unlabeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'poly_sklearn', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-use_model', 't', '-save_model', 'f']

@pytest.mark.toy
class Test113(CmdTestCase):
	'''
	smlp toy basic test for mode optimize from SMLP manual
	'''

	nr = 113
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '0', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model', 't', '-model_name', 'test113_model', '-save_model_config', 't', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-spec', 'smlp_toy_basic.spec']

@pytest.mark.toy
class Test114(CmdTestCase):
	'''
	smlp toy basic test for mode optimize from SMLP manual without specifying resp and feat in command line
	'''

	nr = 114
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '0', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model', 'f', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-spec', 'smlp_toy_basic.spec']

@pytest.mark.toy
class Test115(CmdTestCase):
	'''
	basic test in certify mode
	'''

	nr = 115
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_system.spec', '-quer_names', 'query1,query2', '-quer_exprs', 'y1>0;y2<=0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test116(CmdTestCase):
	'''
	basic test in certify mode when system is specified and is used as the model; p2 rel-rad needs to be 0 or very close to it the witness to first query to be stable
	'''

	nr = 116
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x1,x2,p1,p2', '-model', 'system', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_system.spec', '-quer_names', 'query1,query2', '-quer_exprs', 'y1>0;y2<=0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test117(CmdTestCase):
	'''
	certification test with knobs only where assertion is valid without stability and fails with stability
	'''

	nr = 117
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'certify', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_certify.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test118(CmdTestCase):
	'''
	verification test with knobs only where assertion is valid without stability and fails with stability
	'''

	nr = 118
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'verify', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_verify.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test119(CmdTestCase):
	'''
	query test with knobs only where query is satisfiable without stability and fails with stability
	'''

	nr = 119
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'query', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_query.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.test
class Test120(CmdTestCase):
	'''
	synthesis test with constant knob and no inputs where synthesis is not feasible because the assertion is not feasible
	'''

	nr = 120
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'synthesize', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_fail.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test121(CmdTestCase):
	'''
	synthesis test with constant knob and no inputs where synthesis is feasible
	'''

	nr = 121
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'synthesize', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test122(CmdTestCase):
	'''
	optimization test with constant knob and no inputs where synthesis is not feasible because the assertion is not feasible but beta constraint is feasible therefore optimization is performed
	'''

	nr = 122
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'lazy', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_fail.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test123(CmdTestCase):
	'''
	optimization test with constant knob and no inputs where synthesis is feasible and optimization is performed
	'''

	nr = 123
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test124(CmdTestCase):
	'''
	optimized synthesis test with constant knob and no inputs where synthesis is not feasible because while beta constraint is feasible the assertion is not feasible therefore optimization is not performed
	'''

	nr = 124
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optsyn', '-pareto', 'f', '-opt_strategy', 'lazy', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_fail.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test125(CmdTestCase):
	'''
	optimized synthesis test with constant knob and no inputs where synthesis is feasible and optimization is performed
	'''

	nr = 125
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optsyn', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test126(CmdTestCase):
	'''
	verification example with knobs only and fictitious inputs that have no effect where proparty is valid without stability and fails with stability
	'''

	nr = 126
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'verify', '-model', 'system', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_verify.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test127(CmdTestCase):
	'''
	certification example with knobs only and fictitious inputs with values fixed through their ranges
	where query is valid without stability and fails with stability
	'''

	nr = 127
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'certify', '-model', 'system', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_certify.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test128(CmdTestCase):
	'''
	Basic regression test in certify mode covering all four possible outcomes when certifying a witness for a query: the witness is stable
	the witness is valid but not stable
	the witness is invalid
	and the constraints are conflicting. The fifth query and witness capture a scenario where the polynomial model conversion to terms was missing the intercepts.
	'''

	nr = 128
	data = 'smlp_toy_ctg_num_resp'
	new_data = ''
	args = ['-mode', 'certify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'poly_sklearn', '-dt_sklearn_max_depth', '15', '-save_model', 'f', '-use_model', 'f', '-model_per_response', 'f', '-spec', 'smlp_toy_witness_certify.spec', '-quer_names', 'query_stable_witness,query_grid_conflict,query_unstable_witness,query_infeasible_witness,query_poly_intercept_sensitive', '-quer_exprs', 'y2<=90;y1>=9;y1>=(-13);y1>9;y1>=(-10)', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test129(CmdTestCase):
	'''
	verification example with demonstrating all basic result scenarious for assertions
	'''

	nr = 129
	data = 'smlp_toy_ctg_num_resp'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'poly_sklearn', '-save_model', 'f', '-use_model', 'f', '-model_per_response', 'f', '-spec', 'smlp_toy_configuration_verify.spec', '-asrt_names', 'assert_stable_config,assert_grid_conflict,assert_unstable_config,assert_infeasible', '-asrt_exprs', 'y2<=90;y1>=9;y1>=(-10);y1>20', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test130(CmdTestCase):
	'''
	captures scenario that interface constraints are consistent but model constraints are (because y2 is declared as int and not as real) -- constant input x1 is dropped as constant feature since it does not occur in constraints
	'''

	nr = 130
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x1,x2,p1,p2', '-resp', 'y1,y2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_const_input.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test131(CmdTestCase):
	'''
	test where input x1 has a constant range 0 and in data it is also constant and it is dropped before building the model because it does not occur in constraints alpha
	eta
	beta
	'''

	nr = 131
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x1,x2,p1,p2', '-resp', 'y1,y2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_const_input_const_range.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test132(CmdTestCase):
	'''
	test where input x1 has a non-constant range 0 to 1 and in data it is constant and it is not dropped before building the model because its range is constrained to constant value through alpha constraint
	'''

	nr = 132
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x1,x2,p1,p2', '-resp', 'y1,y2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_const_input_alpha.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test133(CmdTestCase):
	'''
	captures scenario that interface constraints are consistent but model constraints are not (because y2 is declared as int and not as real) -- constant inut x1 is dropped explicitly using -feat option ; uses uses dt_sklearn
	'''

	nr = 133
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x2,p1,p2', '-resp', 'y1,y2', '-model', 'dt_sklearn', '-tree_encoding', 'nested', '-compress_rules', 'f', '-spec', 'smlp_toy_const_input_dropped.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test134(CmdTestCase):
	'''
	test that model cnstraints are consistent with interface constraints when system is used as the model and output y2 is declared as int -- the model constraints are consistent because y2 is defined as p1+p2-x2 and due to constraints all these variables can assume only integer values thus u2 can also only be an integer
	'''

	nr = 134
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x2,p1,p2', '-resp', 'y1,y2', '-model', 'system', '--spec', 'smlp_toy_consistent_system.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test135(CmdTestCase):
	'''
	captures scenario where interface constraints are consistent but model constraints are not (because y2 is declared as int and not as real) -- constant inut x1 is dropped explicitly using -feat option ; uses nn_keras model which is the only difference with test 133 but which model is used does not matter as the problem is in declaration of y2 as int
	'''

	nr = 135
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x2,p1,p2', '-resp', 'y1,y2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-nn_keras_epochs', '20', '-spec', 'smlp_toy_const_input_dropped.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test136(CmdTestCase):
	'''
	tests usage of compressed data files as well as data files without .csv suffix
	'''

	nr = 136
	data = 'smlp_toy_num_resp_mult_compressed.csv.gz'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test137(CmdTestCase):
	'''
	tests usage of compressed data files as well as data files without .csv suffix
	'''

	nr = 137
	data = 'smlp_toy_num_resp_mult_compressed'
	new_data = 'smlp_toy_num_resp_mult_compressed.csv.bz2'
	args = ['-mode', 'synthesize', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_synthesize.spec', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test138(CmdTestCase):
	'''
	adapts test 134 by changing the system function for y2 to p1+p2-x2+0.01 so while p1
	p2 and x2 can only assume int values y2 can become non-integer which violates the declartion of y2 as integer -- hence the conflict of the system/model constraints with alpha and eta constraits and variable domain declarations
	'''

	nr = 138
	data = 'smlp_toy_const_input'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-feat', 'x2,p1,p2', '-resp', 'y1,y2', '-model', 'system', '--spec', 'smlp_toy_inconsistent_system.spec', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test139(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	'''

	nr = 139
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test140(CmdTestCase):
	'''
	verification example with knobs only and fictitious inputs that have no effect where proparty is valid without stability and fails with stability
	'''

	nr = 140
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'verify', '-model', 'system', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_verify.spec', '-trace_prec', '1', '-trace_anonym', 't', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test141(CmdTestCase):
	'''
	basic test for compress_rules option for dt_sklearn in optimization mode
	'''

	nr = 141
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-pareto', 'f', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 't', '-spec', 'smlp_toy_num_resp_mult.spec', '-objv_names', 'objv_y1,objv_y2', '-objv_exprs', 'y1;y2', '-epsilon', '0.01', '-delta_rel', '0.01', '-data_scaler', 'none', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test142(CmdTestCase):
	'''
	basic test for compress_rules option for rf_sklearn in optsin mode
	'''

	nr = 142
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test143(CmdTestCase):
	'''
	basic test for compress_rules for et_sklearn in mode query
	'''

	nr = 143
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'query', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'nested', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_query.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test144(CmdTestCase):
	'''
	basic test for compress_rules for dt_sklearn in mode verify and re-using saved model
	'''

	nr = 144
	data = 'smlp_toy_num_resp_noknobs'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-tree_encoding', 'nested', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2', '-asrt_exprs', 'x0**2+y1>4.3;(y1+x2)/2<6', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.real
class Test145(CmdTestCase):
	'''
	optimization test with constant knob and no inputs where synthesis is feasible and optimization is performed
	'''

	nr = 145
	data = ''
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-doe_spec', '../grids/doe_two_levels_opt.csv', '-doe_algo', 'latin_hypercube', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.real
class Test146(CmdTestCase):
	'''
	optimization test with constant knob and no inputs where synthesis is feasible and optimization is performed
	'''

	nr = 146
	data = ''
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-model', 'poly_sklearn', '-resp', 'y1,y2', '-feat', 'p1,p2,x1,x2', '-save_model', 't', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-split', '1', '-spec', 'smlp_toy_system.spec', '-doe_spec', '../grids/explore_doe_two_levels.csv', '-doe_algo', 'latin_hypercube', '-epsilon', '0.99999999', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test147(CmdTestCase):
	'''
	checks nn_keras prediction with sw_coef 0.8 and sequential API
	adapts test 28
	'''

	nr = 147
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-sw_coef', '0.8']

@pytest.mark.toy
class Test148(CmdTestCase):
	'''
	checks nn_keras prediction with sw_coef 0.8 and sequential API
	adapts test 28 to have two responses
	'''

	nr = 148
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-sw_coef', '0.8']

@pytest.mark.toy
class Test149(CmdTestCase):
	'''
	tests the mae loss function MeanAbsoluteError and sample weoghts
	'''

	nr = 149
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'mae', '-sw_coef', '0.8']

@pytest.mark.toy
class Test150(CmdTestCase):
	'''
	tests the mape loss function MeanAbsolutePercentageError and sample weights
	'''

	nr = 150
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'mape', '-sw_coef', '0.8']

@pytest.mark.toy
class Test151(CmdTestCase):
	'''
	tests msle loss function MeanSquaredLogarithmicError and and sample weoghts
	'''

	nr = 151
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'msle', '-sw_coef', '3', '-sw_exp', '10', '-sw_int', '0']

@pytest.mark.toy
class Test152(CmdTestCase):
	'''
	tests the huber loss function Huber and sample weights
	'''

	nr = 152
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'huber', '-sw_coef', '8', '-sw_exp', '5', '-sw_int', '0.5']

@pytest.mark.toy
class Test153(CmdTestCase):
	'''
	tests the logcosh loss function LogCosh and sample weights
	'''

	nr = 153
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nn_keras_loss', 'logcosh', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'mse']

@pytest.mark.toy
class Test154(CmdTestCase):
	'''
	basic nn_keras assertion verification test that uses keras tuner for functional model training
	'''

	nr = 154
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3,3,3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5']

@pytest.mark.toy
class Test155(CmdTestCase):
	'''
	basic nn_keras assertion verification test that uses keras tuner with sequrntial models for model training
	'''

	nr = 155
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3,3,3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'mae']

@pytest.mark.toy
class Test156(CmdTestCase):
	'''
	basic nn_keras assertion verification test that uses keras tuner for functional model training; adapts test 154 by consdering multiple responses
	'''

	nr = 156
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'rmse']

@pytest.mark.toy
class Test157(CmdTestCase):
	'''
	basic nn_keras assertion verification test that uses keras tuner with sequrntial models for model training; adapts test 155 by consdering multiple responses
	'''

	nr = 157
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '--model', 'nn_keras', '-nnet_encoding', 'nested', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'rmse,logcosh']

@pytest.mark.toy
class Test158(CmdTestCase):
	'''
	tests the mape loss function and sample weights with model_per_response t
	adapts test 152
	'''

	nr = 158
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'mape', '-model_per_response', 't', '-sw_coef', '8', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'rmse']

@pytest.mark.toy
class Test159(CmdTestCase):
	'''
	tests the msle loss function and sample weights with model_per_response t
	adapts test 153
	'''

	nr = 159
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nn_keras_loss', 'msle', '-model_per_response', 't', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'mae,cosine']

@pytest.mark.toy
class Test160(CmdTestCase):
	'''
	tests nn keras tuner bayesian
	'''

	nr = 160
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nn_keras_loss', 'mape', '-nn_keras_metrics', 'msle', '-nn_keras_tuner', 'bayesian', '-nn_keras_layers_grid', '2,3', '-nn_keras_losses_grid', 'mse,mae,huber', '-model_per_response', 'f', '-sw_coef', '8', '-sw_exp', '5', '-sw_int', '0.5']

@pytest.mark.toy
class Test161(CmdTestCase):
	'''
	tests nn keras tuner bayesian
	'''

	nr = 161
	data = 'smlp_toy_num_resp_mult'
	new_data = 'smlp_toy_num_resp_mult_pred_labeled'
	args = ['-mode', 'predict', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nnet_encoding', 'nested', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nn_keras_loss', 'msle', '-nn_keras_metrics', 'mape,logcosh', '-nn_keras_tuner', 'random', '-nn_keras_lrates_grid', '0.01,0.001', '-nn_keras_batches_grid', '32,64', '-model_per_response', 'f', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5']

@pytest.mark.toy
class Test162(CmdTestCase):
	'''
	tests model term construction with flat_encoding of tress and model per reponse when mrmr_pred is activated and not all features are selected for training the model
	adapts test 139
	'''

	nr = 162
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test163(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	adapts test 139
	'''

	nr = 163
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test164(CmdTestCase):
	'''
	basic flat tree encoding test for dt_sklearn multi objective pareto optimization
	'''

	nr = 164
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test165(CmdTestCase):
	'''
	basic flat tree encoding test for dt_caretin model exploration mode optsyn
	model_per_response is forced to true for caret models
	adapts test 95
	'''

	nr = 165
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-tree_encoding', 'flat', '-save_model', 'f', '-use_model', 'f', '-compress_rules', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test166(CmdTestCase):
	'''
	basic flat tree encoding test with model_per_response f for rf_sklearn in model exploration mode optsyn
	'''

	nr = 166
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '4', '-rf_sklearn_n_estimators', '3', '-tree_encoding', 'flat', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-compress_rules', 't', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test167(CmdTestCase):
	'''
	basic flat tree encoding test with model_per_response t for rf_sklearn in model exploration mode optsyn
	adapts test 94 and test 166
	'''

	nr = 167
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '4', '-rf_sklearn_n_estimators', '3', '-tree_encoding', 'flat', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test168(CmdTestCase):
	'''
	basic test for rf_caret with flat tree_encoding and modelper_response in model exploration mode optimize
	'''

	nr = 168
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_caret', '-model_per_response', 't', '-compress_rules', 't', '-tree_encoding', 'flat', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test169(CmdTestCase):
	'''
	basic test for et_sklearn with flat tree_encoding and model_per_response t in model exploration mode optimize
	'''

	nr = 169
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-rf_sklearn_n_estimators', '3', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'flat', '-model_per_response', 't', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test170(CmdTestCase):
	'''
	basic test for et_sklearn with flat tree_encoding and model_per_response f in model exploration mode optimize
	'''

	nr = 170
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-rf_sklearn_n_estimators', '3', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'flat', '-model_per_response', 'f', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test171(CmdTestCase):
	'''
	basic test for et_caret with flat tree_encoding in model exploration mode optimize
	'''

	nr = 171
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_caret', '-tree_encoding', 'flat', '-model_per_response', 't', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test172(CmdTestCase):
	'''
	basic test for nn_keras flat encoding for functional api
	i
	one response variable
	adapts test 154
	'''

	nr = 172
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nnet_encoding', 'layered', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3,3,3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-solver_path', '/nfs/iil/proj/dt/eva/smlp/external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test173(CmdTestCase):
	'''
	basic test for nn_keras flat encoding for sequential api
	one response variable
	adapts test 155
	'''

	nr = 173
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'verify', '-resp', 'y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nnet_encoding', 'layered', '-nn_keras_tuner', 'hyperband', '-nn_keras_layers_grid', '2,2;3,3,3', '-save_model_config', 'f', '-spec', 'smlp_toy_num_resp_mult_y2_verify.spec', '-asrt_names', 'asrt1', '-asrt_exprs', '2*y2>1', '-sw_coef', '4', '-sw_exp', '5', '-sw_int', '0.5', '-nn_keras_metrics', 'mae', '-solver_path', '/nfs/iil/proj/dt/eva/smlp/external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test174(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response f nn_keras_seq_api f for nn_keras in model exploration mode optsyn
	'''

	nr = 174
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test175(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response f nn_keras_seq_api t for nn_keras in model exploration mode optsyn
	'''

	nr = 175
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test176(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response t nn_keras_seq_api f for nn_keras in model exploration mode optsyn
	'''

	nr = 176
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test177(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response t nn_keras_seq_api t for nn_keras in model exploration mode optsyn
	'''

	nr = 177
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test178(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response t nn_keras_seq_api t for nn_keras in model exploration mode optsyn when features are not scaled adapts test 177
	'''

	nr = 178
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-scale_feat', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test179(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response f nn_keras_seq_api f for nn_keras in model exploration mode optsyn when resposes are not scaled adapts test 174
	'''

	nr = 179
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 'f', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 'f', '-scale_resp', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test180(CmdTestCase):
	'''
	basic layered nn_keras encoding test with model_per_response f nn_keras_seq_api t for nn_keras in model exploration mode optsyn when features and responses are not scaled adapts test 175
	'''

	nr = 180
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'nn_keras', '-nn_keras_epochs', '20', '-nn_keras_seq_api', 't', '-nnet_encoding', 'layered', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 'f', '-scale_feat', 'f', '-scale_resp', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test181(CmdTestCase):
	'''
	basic flat tree encoding test for dt_sklearn multi objective pareto optimization when features are not scaled modifies test 164
	'''

	nr = 181
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-scale_feat', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test182(CmdTestCase):
	'''
	basic flat tree encoding test for dt_sklearn multi objective pareto optimization when responses are not scaled modifies test 164
	'''

	nr = 182
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-scale_resp', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', '/nfs/iil/proj/dt/eva/smlp/external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test183(CmdTestCase):
	'''
	basic flat tree encoding test for dt_sklearn multi objective pareto optimization when features and responses are not scaled modifies test 164
	'''

	nr = 183
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'flat', '-scale_resp', 'f', '-scale_feat', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test184(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model adapts test 139
	'''

	nr = 184
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test185(CmdTestCase):
	'''
	tests model term construction with branched_encoding of tress and model per reponse when mrmr_pred is activated and not all features are selected for training the model
	adapts test 162
	'''

	nr = 185
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test186(CmdTestCase):
	'''
	tests model term formation when mrmr_pred is activated and not all features are selected for training the model
	adapts test 163
	'''

	nr = 186
	data = 'smlp_toy_num_resp_noknobs'
	new_data = 'smlp_toy_num_resp_noknobs_pred_labeled'
	args = ['-mode', 'verify', '-resp', 'y1,y2', '-feat', 'x0,x1,x2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_noknobs_verify.spec', '-asrt_names', 'asrt1,asrt2,asrt3', '-asrt_exprs', '(y2**3+x2)/2<6;y1>=9;y2<0', '-trace_anonym', 't', '-trace_prec', '3', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test187(CmdTestCase):
	'''
	basic branched tree encoding test for dt_sklearn multi objective pareto optimization adapts test 164
	'''

	nr = 187
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test188(CmdTestCase):
	'''
	basic branched tree encoding test for dt_caretin model exploration mode optsyn
	model_per_response is forced to true for caret models
	adapts test 165
	'''

	nr = 188
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_caret', '-tree_encoding', 'branched', '-save_model', 'f', '-use_model', 'f', '-compress_rules', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test189(CmdTestCase):
	'''
	basic branched tree encoding test with model_per_response f for rf_sklearn in model exploration mode optsyn adapts test 166
	'''

	nr = 189
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '4', '-rf_sklearn_n_estimators', '3', '-tree_encoding', 'branched', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-compress_rules', 't', '-mrmr_pred', '2', '-model_per_response', 'f', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test190(CmdTestCase):
	'''
	basic test for rf_caret with branched tree_encoding and modelper_response in model exploration mode optimize adapts test 168
	'''

	nr = 190
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_caret', '-model_per_response', 't', '-compress_rules', 't', '-tree_encoding', 'branched', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test191(CmdTestCase):
	'''
	basic test for et_sklearn with branched tree_encoding and model_per_response t in model exploration mode optimize adapts test 169
	'''

	nr = 191
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-et_sklearn_n_estimators', '3', '-et_sklearn_bootstrap', 't', '-tree_encoding', 'branched', '-model_per_response', 't', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test192(CmdTestCase):
	'''
	basic test for et_sklearn with branched tree_encoding and model_per_response f in model exploration mode optimize adapts test 170 !!!!!!!!! in this test z3 result differs from mathsat and yices results (the latter two give sma results
	cvc5 faild with incomparable ite tipes for if and else branches)
	'''

	nr = 192
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-et_sklearn_n_estimators', '100', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'branched', '-model_per_response', 'f', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test193(CmdTestCase):
	'''
	basic test for et_caret with branched tree_encoding in model exploration mode optimize adapts test 171
	'''

	nr = 193
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_caret', '-tree_encoding', 'branched', '-model_per_response', 't', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test194(CmdTestCase):
	'''
	basic branched tree encoding test with model_per_response t for rf_sklearn in model exploration mode optsyn
	adapts test 94 and test 167
	'''

	nr = 194
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optsyn', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'rf_sklearn', '-rf_sklearn_max_depth', '4', '-rf_sklearn_n_estimators', '3', '-tree_encoding', 'branched', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-model_per_response', 't', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test195(CmdTestCase):
	'''
	basic test for et_sklearn with branched tree_encoding and model_per_response f in model exploration mode optimize adapts test 192 by setting n_estimators 3 and then discrepancy between z3
	mathsat and yices results disappear
	'''

	nr = 195
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-et_sklearn_n_estimators', '3', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'branched', '-model_per_response', 'f', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test196(CmdTestCase):
	'''
	basic branched tree encoding test for dt_sklearn multi objective pareto optimization when features are not scaled modifies test 164 and test 181
	'''

	nr = 196
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-scale_feat', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test197(CmdTestCase):
	'''
	basic branched tree encoding test for dt_sklearn multi objective pareto optimization when responses are not scaled modifies test 164 and test 182
	'''

	nr = 197
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-scale_resp', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', '/nfs/iil/proj/dt/eva/smlp/external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test198(CmdTestCase):
	'''
	basic branched tree encoding test for dt_sklearn multi objective pareto optimization when features and responses are not scaled modifies test 164 and test 183
	'''

	nr = 198
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-tree_encoding', 'branched', '-scale_resp', 'f', '-scale_feat', 'f', '-spec', 'smlp_toy_num_resp_mult_free_inps_beta_objv.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat']

@pytest.mark.toy
class Test199(CmdTestCase):
	'''
	test to demonstrate that in pareto optimization and optsyn modes with multiple objectives when beta constraints are not present SMLP results are not consistent when different solvers are used; this is due to fact that when a subset of objectoves are exemined in pareto algo
	outputs not covered by the active objectives become don't cares (there are no contraints on then except model constraints) and this situation is likely not modeled in SMLP accurately; modifies test 192 to use z3 instead of mathsat
	'''

	nr = 199
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-et_sklearn_n_estimators', '100', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'branched', '-model_per_response', 'f', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0.05', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test200(CmdTestCase):
	'''
	basic test for et_sklearn with branched tree_encoding and model_per_response f in model exploration mode optimize adapts test 170 !!!!!!!!! in this test z3 result differs from mathsat and yices results (the latter two give sma results
	cvc5 faild with incomparable ite tipes for if and else branches)
	'''

	nr = 200
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-opt_strategy', 'lazy', '-resp', 'y1,y2', '-feat', 'x,p1,p2', '-model', 'et_sklearn', '-et_sklearn_max_depth', '2', '-et_sklearn_n_estimators', '100', '-et_sklearn_bootstrap', 'f', '-tree_encoding', 'branched', '-model_per_response', 'f', '-compress_rules', 't', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '2', '-spec', 'smlp_toy_num_resp_mult_optsyn.spec', '-epsilon', '0.1', '-delta_rel', '0', '-solver_path', 'mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test201(CmdTestCase):
	'''
	basic dt_sklearn single objective optimization with the eager algorithm when there are no inputs and there are beta constraints
	'''

	nr = 201
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'eager', '-resp', 'y1,y2', '-feat', 'p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_no_input_beta.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1', '-objv_exprs', '(y1+y2)/2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test202(CmdTestCase):
	'''
	basic dt_sklearn single objective optimization with the eager algorithm when there are no inputs and no beta constraints
	'''

	nr = 202
	data = 'smlp_toy_num_resp_mult'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'eager', '-resp', 'y1,y2', '-feat', 'p1,p2', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_num_resp_mult_no_input.spec', '-data_scaler', 'min_max', '-objv_names', 'obj1', '-objv_exprs', '(y1+y2)/2', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test203(CmdTestCase):
	'''
	optimization test with eager strategy and with constant knob and no inputs where synthesis is not feasible because the assertion is not feasible but beta constraint is feasible therefore optimization is performed adapts test 122
	'''

	nr = 203
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 'f', '-opt_strategy', 'eager', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_fail.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test204(CmdTestCase):
	'''
	optimization test with eager strategy and with constant knob and no inputs where synthesis is feasible and optimization is performed adapts test 123
	'''

	nr = 204
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'eager', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
@pytest.mark.real
class Test205(CmdTestCase):
	'''
	optimization test with eager strategy and with constant knob and no inputs where synthesis is feasible and optimization is performed adapts test 145
	'''

	nr = 205
	data = ''
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-opt_strategy', 'eager', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-doe_spec', '../grids/doe_two_levels_opt.csv', '-doe_algo', 'latin_hypercube', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test206(CmdTestCase):
	'''
	optimized synthesis test with eager strategy and with constant knob and no inputs where synthesis is feasible and optimization is performed adapts test 125
	'''

	nr = 206
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'optsyn', '-pareto', 't', '-opt_strategy', 'eager', '-model', 'system', '-resp', 'y1,y2', '-feat', 'p1,p2', '-save_model', 'f', '-use_model', 'f', '-mrmr_pred', '0', '-model_per_response', 't', '-spec', 'smlp_toy_system_stable_constant_synth_feasible.spec', '-epsilon', '0.00000001', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test207(CmdTestCase):
	'''
	basic test for mode frontier -- selecting pareto frontier directly from data without building a model
	'''

	nr = 207
	data = 'smlp_toy_frontier_beta'
	new_data = ''
	args = ['-mode', 'frontier', '-pareto', 't', '-resp', 'y', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_beta.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test208(CmdTestCase):
	'''
	testing unbounded knob and input domains for mode frontier when knob and input are of type real -- bounds inf and minus inf are specified in the spec file as null
	'''

	nr = 208
	data = 'smlp_toy_frontier_beta'
	new_data = ''
	args = ['-mode', 'frontier', '-pareto', 't', '-resp', 'y', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_real.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test209(CmdTestCase):
	'''
	testing unbounded knob and input domains for mode frontier when knob and input are of typ eint -- bounds inf and minus inf are specified in the spec file as null
	'''

	nr = 209
	data = 'smlp_toy_frontier_null_bounds_int'
	new_data = ''
	args = ['-mode', 'frontier', '-pareto', 't', '-resp', 'y', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_int.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test210(CmdTestCase):
	'''
	test for the frontier mode on data and spec such that no data points satisfy eta constrints
	as a result the pareto frontier is empty
	'''

	nr = 210
	data = 'smlp_toy_frontier_null_bounds_empty'
	new_data = ''
	args = ['-mode', 'frontier', '-pareto', 't', '-resp', 'y1,y2', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_empty_eta.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test211(CmdTestCase):
	'''
	test for the frontier mode on data and spec such that no data points satisfy eta constrints
	as a result the pareto frontier is empty
	'''

	nr = 211
	data = 'smlp_toy_frontier_null_bounds_empty'
	new_data = ''
	args = ['-mode', 'frontier', '-pareto', 't', '-resp', 'y1,y2', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_empty_alpha.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test212(CmdTestCase):
	'''
	testing unbounded knob and input domains for optimization mode -- bounds inf and minus inf are specified in the spec file as null
	'''

	nr = 212
	data = 'smlp_toy_frontier_null_bounds_int'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-resp', 'y', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_int.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test213(CmdTestCase):
	'''
	test optimize mode on data and spec such that no data points satisfy eta constrints
	'''

	nr = 213
	data = 'smlp_toy_frontier_null_bounds_empty'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-resp', 'y1,y2', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_empty_eta.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test214(CmdTestCase):
	'''
	test for the frontier mode on data and spec such that no data points satisfy eta constrints
	as a result the pareto frontier is empty
	'''

	nr = 214
	data = 'smlp_toy_frontier_null_bounds_empty'
	new_data = ''
	args = ['-mode', 'optimize', '-pareto', 't', '-resp', 'y1,y2', '-feat', 'x,p', '-model', 'dt_sklearn', '-dt_sklearn_max_depth', '15', '-compress_rules', 'f', '-spec', 'smlp_toy_frontier_null_bounds_empty_alpha.spec', '-data_scaler', 'min_max', '-epsilon', '0.05', '-delta_rel', '0.01', '-save_model_config', 'f', '-mrmr_pred', '0', '-plots', 'f', '-pred_plots', 'f', '-resp_plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test215(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type object
	'''

	nr = 215
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test216(CmdTestCase):
	'''
	basic test for correlate mode
	'''

	nr = 216
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test217(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type category
	adapts test 215
	'''

	nr = 217
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'category', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test218(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type ordered
	adapts test 215
	'''

	nr = 218
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'ordered', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test219(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features casted to integer
	adapts test 215
	'''

	nr = 219
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'integer', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test220(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type object and tests the normalized mutual information
	adapts test 215
	'''

	nr = 220
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'normalized', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test221(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type object and tests the Shannon mutual information
	adapts test 215
	'''

	nr = 221
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'shannon', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test222(CmdTestCase):
	'''
	basic test for correlate mode
	contains correlations for categorical features of type object and tests the adjusted mutual information
	adapts test 215
	'''

	nr = 222
	data = 'smlp_toy_mult_discr'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'PF,PF1', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'adjusted', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f', '-pos_val', 'fail', '-neg_val', 'pass']

@pytest.mark.toy
class Test223(CmdTestCase):
	'''
	basic test for correlate mode and tests the normalized mutual information
	adapts test 216
	'''

	nr = 223
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'normalized', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test224(CmdTestCase):
	'''
	basic test for correlate mode and tests the Shannon mutual information
	adapts test 216
	'''

	nr = 224
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'shannon', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test225(CmdTestCase):
	'''
	basic test for correlate mode and tests the adjusted mutual information
	adapts test 216
	'''

	nr = 225
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'adjusted', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test226(CmdTestCase):
	'''
	basic test for correlate mode
	'''

	nr = 226
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discret_num', 't', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'correlation', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']

@pytest.mark.toy
class Test227(CmdTestCase):
	'''
	basic test for correlate mode and tests the normalized mutual information
	adapts test 216 and 223
	'''

	nr = 227
	data = 'smlp_toy_basic'
	new_data = ''
	args = ['-mode', 'correlate', '-resp', 'y1,y2', '-discr_algo', 'uniform', '-discret_num', 't', '-discr_bins', '6', '-discr_labels', 't', '-discr_type', 'object', '-data_scaler', 'none', '-cont_est', 'pearson,spearman,kendall', '-mi_method', 'normalized', '-mrmr_pred', '0', '-plots', 'f', '-seed', '10', '-log_time', 'f']


