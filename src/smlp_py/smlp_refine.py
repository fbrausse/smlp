import numpy as np
import pandas as pd
import os


from sklearn.metrics import mean_squared_error
from smlp_py.smlp_doe import SmlpDoepy
from smlp_py.smlp_models import SmlpModels

# Assuming the SmlpDoepy class definition is already provided as shown above

from sklearn.metrics import mean_squared_error
from scipy.stats import norm

class SmlpRefine:
    def __init__(self, num_samples=100, doe_method='latin_hypercube'):
        self.doe_generator = SmlpDoepy()
        self.num_samples = num_samples
        self.doe_method = doe_method
        self._refine_logger = None
        self._report_file_prefix = None
        self._modelInst = SmlpModels()
        
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._refine_logger = logger
        self.doe_generator.set_logger(logger)
        self._modelInst.set_logger(logger)
        
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self._report_file_prefix = report_file_prefix
        self._modelInst.set_report_file_prefix(report_file_prefix)
        
    def generate_doe_samples(self, config_dict, doe_method='latin_hypercube'):
        doe_generator = self.doe_generator
        num_samples = self.num_samples
        
        large_value = 1e6  # A large number to approximate infinity
        doe_spec_dict = {}
        for k, (lower, upper) in config_dict.items():
            if lower is None: #== float('-inf'):
                lower = -large_value
            if upper is None: # float('inf'):
                upper = large_value
            doe_spec_dict[k] = [lower, upper]
        
        #print('doe_spec_dict after eliminating inf/none', doe_spec_dict)
        # Convert the config_dict to the format expected by SmlpDoepy
        #doe_spec_dict = {k: [v[0], v[1]] for k, v in doe_spec_dict.items()}
        # Generate the DOE samples using the specified method
        # ... (same as before, using self.doe_generator and self.doe_method)
        # ...
        
        # Generate the DOE samples using the specified method
        if doe_method == doe_generator.LATIN_HYPERCUBE:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',  # No file output needed
                prob_distribution= 'normal', #'uniform',  # Assuming uniform distribution for simplicity
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.FULL_FACTORIAL:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for full factorial
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.BOX_BEHNKEN:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for Box-Behnken
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=1  # Default value, adjust if needed
            )
        elif doe_method == doe_generator.CENTRAL_COMPOSITE:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for Central Composite
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center='2,2',  # Default value, adjust if needed
                central_composite_face='ccf',  # Default value, adjust if needed
                central_composite_alpha='o',  # Default value, adjust if needed
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.PLACKETT_BURMAN:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for Plackett-Burman
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.TWO_LEVEL_FRACTIONAL_FACTORIAL:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for two-level fractional factorial
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=3,  # Example value, adjust if needed
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.SUKHAREV_GRID:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == self.doe_generator.BOX_WILSON: #doe_method == self.doe_generator.CENTRAL_COMPOSITE or 
            # Handle CENTRAL_COMPOSITE and BOX_WILSON (alias)
            doe_samples_df = self.doe_generator.sample_doepy(
                doe_algo=self.doe_method,
                doe_spec=doe_spec_dict,
                num_samples=None,  # Not used for Central Composite
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center='2,2',  # Default value, adjust if needed
                central_composite_face='ccf',  # Default value, adjust if needed
                central_composite_alpha='o',  # Default value, adjust if needed
                box_behnken_centers=None
            ) 
        elif doe_method == doe_generator.LATIN_HYPERCUBE_SPACE_FILLING:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.RANDOM_K_MEANS:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.MAXMIN_RECONSTRUCTION:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.HALTON_SEQUENCE:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        elif doe_method == doe_generator.UNIFORM_RANDOM_MATRIX:
            doe_samples_df = doe_generator.sample_doepy(
                doe_algo=doe_method,
                doe_spec=doe_spec_dict,
                num_samples=num_samples,
                report_file_prefix='',
                prob_distribution=None,
                fractional_factorial_resolution=None,
                central_composite_center=None,
                central_composite_face=None,
                central_composite_alpha=None,
                box_behnken_centers=None
            )
        else:
            # Add additional elif blocks for other DOE methods here
            raise NotImplementedError(f"DOE method '{doe_method}' is not implemented in this function.")
        
        # Transform the samples using the inverse CDF of the normal distribution
        #print('config_dict', config_dict); print('doe_samples_df\n', doe_samples_df)
        ''' 
        # TODO !!!!!!!!!!!!!  fix usage of norm.ppf(doe_samples_df[k]) -- norm.ppf() expects values (probabilities) between 0 an 1 
        for k in config_dict:
            if config_dict[k][0] is None or config_dict[k][1] is None: #== float('-inf') or config_dict[k][1] == float('inf'):
                #print('before norm\n', doe_samples_df[k], 'after norm', norm.ppf(doe_samples_df[k]))
                doe_samples_df[k] = norm.ppf(doe_samples_df[k])
        assert False
        '''
        return doe_samples_df

    def evaluate_system_expr(self, system_expr_dict:dict, samples_df:pd.DataFrame):
        #print('system_expr', system_expr_dict)
        true_values_dict = {}
        for resp, syst in system_expr_dict.items():
            true_values = samples_df.apply(lambda row: eval(syst, {}, row.to_dict()), axis=1)
            true_values_dict[resp] = true_values
        return pd.DataFrame.from_dict(true_values_dict)

    #_model_predict(self, model, X:pd.DataFrame, y:pd.DataFrame, resp_names:list, algo:str, model_per_response:bool)
    def evaluate_model(self, model, algo:str, samples_df:pd.DataFrame, model_features_dict:dict, resp_names:list[str], model_per_response:bool):
        #print('model', model); print('samples_df\n', samples_df)
        if algo == 'system':
            assert list(model.keys()) == resp_names 
            pred_values_df = self.evaluate_system_expr(model, samples_df); #print('pred_values_df 1\n', pred_values_df)
            pred_values_df.rename(columns=dict(zip(pred_values_df.columns, [k+'_system' for k in pred_values_df.columns])), inplace=True)
            #print('pred_values_df 2\n', pred_values_df)
        elif isinstance(model, dict):
            assert list(model.keys()) == resp_names 
            # model.predict() required input data columns to be same and in the same order as during training.
            # Therefore we reorder features in samples_df based on model_features_dict[resp_names[0]] 
            # (which uses the first response in resp_names) and then assert other responses define the same order of the features. 
            feat_ordered = model_features_dict[resp_names[0]]
            samples_df = samples_df[feat_ordered]
            for resp in resp_names:
                assert model_features_dict[resp] == feat_ordered
            pred_values_df = self._modelInst._model_predict(model, samples_df, None, resp_names, algo, model_per_response)
        else:
            pred_values_df = self._modelInst._model_predict(model, samples_df, None, resp_names, algo, model_per_response)
            #assert False
        return pred_values_df
    
    def compute_rmse(self, config_dict:dict, model, algo:str, model_features_dict:dict, resp_names:list[str], 
            model_per_response:bool, system_expr_dict:dict, mm_scaler_resp, 
            interactive_plots:bool, prediction_plots:bool, doe_method='latin_hypercube'):
        #print('model', model); print('system', system_expr_dict); print('model_features_dict', model_features_dict); print('resp_names', resp_names)
        if system_expr_dict is not None:
            assert list(system_expr_dict.keys()) == resp_names
        else:
            return
        samples_df = self.generate_doe_samples(config_dict, doe_method); #print('samples_df\n', samples_df)
        
        model_predictions = self.evaluate_model(model, algo, samples_df, model_features_dict, resp_names, model_per_response); #print('model_predictions\n', model_predictions)
        true_values = self.evaluate_system_expr(system_expr_dict, samples_df); #print('true_values\n', true_values)
        
        #y_train_pred = self._model_predict(model, X_train, y_train, resp_names, algo, model_per_response)
        self._modelInst._report_prediction_results(algo, list(true_values.keys()), true_values, model_predictions, mm_scaler_resp,
            interactive_plots, prediction_plots, 'sampling')
        responses = zip(list(true_values.keys()), model_predictions.keys()); #print('responses', responses)
        for resp_syst, resp_modl in responses:
            #print('true_values[resp]\n', true_values[resp_syst]); print('model_predictions[resp]\n', model_predictions[resp_modl])
            rmse = np.sqrt(mean_squared_error(true_values[resp_syst], model_predictions[resp_modl])); #print('rmse', rmse)
        return rmse
'''
# Example usage:
# config_dict = {'a': [0, 10], 'b': [5, 15], 'c': [10, 20]}
# M = SomePretrainedSklearnModel()  # Replace with your actual model
# system_expr = 'a + b - c'  # Replace with your actual system expression
# smlp_refine = SmlpRefine(num_samples=100, doe_method='latin_hypercube')
# rmse = smlp_refine.compute_rmse(config_dict, M, system_expr)
# print(f"RMSE: {rmse}")
# Instantiate the SmlpDoepy class
doe_generator = SmlpDoepy()


# Number of samples you want to generate
num_samples = 10  # You can adjust this number as needed


def generate_doe_samples(config_dict, num_samples, doe_method):
    # Instantiate the SmlpDoepy class
    doe_generator = SmlpDoepy()
    
    # Convert the config_dict to the format expected by SmlpDoepy
    # For unbounded intervals, use a large finite range as a placeholder
    large_value = 1e6  # A large number to approximate infinity
    doe_spec_dict = {}
    for k, (lower, upper) in config_dict.items():
        if lower == float('-inf'):
            lower = -large_value
        if upper == float('inf'):
            upper = large_value
        doe_spec_dict[k] = [lower, upper]
    
    # Convert the config_dict to the format expected by SmlpDoepy
    doe_spec_dict = {k: [v[0], v[1]] for k, v in config_dict.items()}
    
    # Generate the DOE samples using the specified method
    if doe_method == doe_generator.LATIN_HYPERCUBE:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',  # No file output needed
            prob_distribution='uniform',  # Assuming uniform distribution for simplicity
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.FULL_FACTORIAL:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=None,  # Not used for full factorial
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.BOX_BEHNKEN:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=None,  # Not used for Box-Behnken
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=1  # Default value, adjust if needed
        )
    elif doe_method == doe_generator.CENTRAL_COMPOSITE:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=None,  # Not used for Central Composite
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center='2,2',  # Default value, adjust if needed
            central_composite_face='ccf',  # Default value, adjust if needed
            central_composite_alpha='o',  # Default value, adjust if needed
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.PLACKETT_BURMAN:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=None,  # Not used for Plackett-Burman
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.TWO_LEVEL_FRACTIONAL_FACTORIAL:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=None,  # Not used for two-level fractional factorial
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=3,  # Example value, adjust if needed
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.SUKHAREV_GRID:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.BOX_WILSON:
        # Alias for CENTRAL_COMPOSITE, handle as CENTRAL_COMPOSITE
        # ... (same as CENTRAL_COMPOSITE)
    elif doe_method == doe_generator.LATIN_HYPERCUBE_SPACE_FILLING:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.RANDOM_K_MEANS:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.MAXMIN_RECONSTRUCTION:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.HALTON_SEQUENCE:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    elif doe_method == doe_generator.UNIFORM_RANDOM_MATRIX:
        doe_samples_df = doe_generator.sample_doepy(
            doe_algo=doe_method,
            doe_spec=doe_spec_dict,
            num_samples=num_samples,
            report_file_prefix='',
            prob_distribution=None,
            fractional_factorial_resolution=None,
            central_composite_center=None,
            central_composite_face=None,
            central_composite_alpha=None,
            box_behnken_centers=None
        )
    else:
        # Add additional elif blocks for other DOE methods here
        raise NotImplementedError(f"DOE method '{doe_method}' is not implemented in this function.")
        
    # Transform the samples using the inverse CDF of the normal distribution
        for k in config_dict:
            if config_dict[k][0] == float('-inf') or config_dict[k][1] == float('inf'):
                doe_samples_df[k] = norm.ppf(doe_samples_df[k])
    # ... (handle other methods)
    return doe_samples_df


def evaluate_system_expr(system_expr, samples_df):
    # Evaluate the system expression on the samples
    true_values = samples_df.apply(lambda row: eval(system_expr, {}, row.to_dict()), axis=1)
    return true_values

def compute_rmse(config_dict, model, system_expr, num_samples=100, doe_method='latin_hypercube'):
    # Generate DOE samples
    samples_df = generate_doe_samples(config_dict, num_samples, doe_method)
    
    # Evaluate the model on the samples
    model_predictions = model.predict(samples_df)
    
    # Evaluate the system expression on the samples
    true_values = evaluate_system_expr(system_expr, samples_df)
    
    # Compute the RMSE
    rmse = np.sqrt(mean_squared_error(true_values, model_predictions))
    
    return rmse

# Example usage:
# config_dict = {'a': [0, 10], 'b': [5, 15], 'c': [10, 20]}
# M = SomePretrainedSklearnModel()  # Replace with your actual model
# system_expr = 'a + b - c'  # Replace with your actual system expression
# doe_method = 'latin_hypercube'  # Replace with your desired DOE method

# rmse = compute_rmse(config_dict, M, system_expr, num_samples=100, doe_method=doe_method)
# print(f"RMSE: {rmse}")
'''