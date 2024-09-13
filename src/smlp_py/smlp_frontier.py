# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import numpy as np
import pandas as pd
from smlp_py.smlp_spec import SmlpSpec

# functions to compute pareto frontier directly from data
class SmlpFrontier:
    def __init__(self):
        self._frontier_logger = None 
        self.report_file_prefix = None
        self._specInst = None
    
    def set_logger(self, logger):
        self._frontier_logger = logger 
    
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    def set_spec_inst(self, spec_inst):
        self._specInst = spec_inst
    
    # Function local_minimum() computes stable minimum of reponse columns resp_names for each row.
    # It performs the following steps:
    # 1. Iterate over each row of the DataFrame df.
    # 2. For each row, use the values in the feat_names columns and the feat_radii provided by func_intervals to compute the intervals (l[i], u[i]).
    # 3. Use these intervals to create a boolean mask that selects rows where the feat_names columns' values are within the respective intervals.
    # 4. For the selected subset of rows, compute the minimum values for each of the resp_names columns.
    # 5. Assign these minimum values to the corresponding resp_names columns in the new DataFrame df_min.
    # Example usage:
    # Assuming you have a DataFrame 'df' with columns 'A', 'B', 'C', 'D'
    # df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]})
    # feat_names = ['A', 'B']
    # resp_names = ['C', 'D']
    # def func_intervals(feat_vals):
    #     # Example function that returns a list of radii
    #     return [1, 1]  # Fixed radii for simplicity
    # df_min = local_minimum(df, feat_names, resp_names, func_intervals)
    # print(df_min)
    def local_minimum(self, df, feat_names, resp_names, func_intervals):
        # Initialize df_min with the same shape as df and copy feat_names columns
        df_min = pd.DataFrame(index=df.index, columns=df.columns)
        df_min[feat_names] = df[feat_names]
        #print('feat_names', feat_names, 'resp_names', resp_names)
        # Iterate over each row of the DataFrame
        for index, row in df.iterrows():
            # Get the feature values and radii for the current row
            #print('row[feat_names]', row[feat_names]); print(dict(row[feat_names]))
            #feat_vals = row[feat_names].values; print('feat_vals', feat_vals)
            intervals = func_intervals(dict(row[feat_names])); #print('intervals', intervals)
            
            # Create a boolean mask for rows within the intervals
            mask = pd.Series([True] * len(df))
            #for feat_name, lower, upper in zip(feat_names, l, u):
            #    mask &= (df[feat_name] >= lower) & (df[feat_name] <= upper)
            for feat_name in feat_names:
                mask &= (df[feat_name] >= intervals[feat_name][0]) & (df[feat_name] <= intervals[feat_name][1])
            
            # Compute the minimum values for resp_names columns within the mask
            min_values = df[mask][resp_names].min()

            # Assign the minimum values to the corresponding resp_names columns in df_min
            df_min.loc[index, resp_names] = min_values

        return df_min

    # This is an efficient implimentation that is not using df.iterrows()
    def local_minimum(self, df, feat_names, resp_names, objv_names, func_intervals):
        # Initialize df_min with the same shape as df and copy feat_names columns
        df_min = pd.DataFrame(index=df.index, columns=df.columns)
        df_min[feat_names] = df[feat_names]; #print('df_min with features\n', df_min)
        df_min[resp_names] = df[resp_names];
        
        # Precompute all intervals for all rows
        intervals_list = [func_intervals(dict(row[feat_names])) for _, row in df.iterrows()]
        #print('intervals_list', len(intervals_list), intervals_list)
        lower_bounds = np.array([[interval[feat_name][0] for feat_name in feat_names] for interval in intervals_list])
        upper_bounds = np.array([[interval[feat_name][1] for feat_name in feat_names] for interval in intervals_list])
        # None as a lower bound for interval represents -inf and as upper bound represents as inf; None is allowed in
        # specifying ranges of inputs and knobs in the spec file, thus we update lower_bounds and upper_bounds as follows:
        lower_bounds[lower_bounds == None] = (-np.inf); #print('lower_bounds\n', lower_bounds)
        upper_bounds[upper_bounds == None] = np.inf; #print('upper_bounds\n', upper_bounds)
        
        # Compute the minimum values for objv_names columns within the intervals
        for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
            #print('i, (lower, upper)', i, (lower, upper))
            # Create a boolean mask for rows within the intervals
            mask = np.ones(len(df), dtype=bool)
            for j, feat_name in enumerate(feat_names):
                mask &= (df[feat_name] >= lower[j]) & (df[feat_name] <= upper[j])

            # Compute the minimum values for objv_names columns within the mask
            min_values = df.loc[mask, objv_names].min(); #print('min_values\n', min_values)
            df_min.loc[i, objv_names] = min_values; #print('df_min i\n', df_min)

        return df_min

    # find pareto subset of data with respect to optimization objectives specified in argument 'objectives".
    # pareto subset is with respect to the maximum (optimization means finding maximum, not minimum).
    def find_pareto_front(self, df, knobs, objectives, precision=None):
        pareto_df = df.copy()
        # If precision is specified, round the objectives columns
        if precision is not None:
            pareto_df[objectives] = pareto_df[objectives].round(precision)

        # Group by the objective values and create a DataFrame with the first row from each group
        grouped = pareto_df.groupby(objectives)
        df_groupby_one = pd.DataFrame()
        for _, group_df in grouped:
            #print('group_df\n', group_df)
            df_groupby_one = pd.concat([df_groupby_one, group_df.head(1)], ignore_index=True)
        #print('df_groupby_one\n', df_groupby_one)
        # Find the Pareto front within df_groupby_one
        data = df_groupby_one[objectives].values
        is_pareto_optimal = np.ones(data.shape[0], dtype=bool)
        for i, candidate in enumerate(data):
            if is_pareto_optimal[i]:
                is_pareto_optimal[is_pareto_optimal] = np.any(data[is_pareto_optimal] > candidate, axis=1)
                is_pareto_optimal[i] = True  # Keep the current candidate

        # Select the Pareto optimal rows from df_groupby_one
        pareto_groupby_one = df_groupby_one[is_pareto_optimal]; #print('pareto_groupby_one\n', pareto_groupby_one)

        # Expand the Pareto front by including all rows from the original DataFrame
        # that correspond to the Pareto optimal rows in pareto_groupby_one
        pareto_front_indices = pareto_df.reset_index().merge(pareto_groupby_one, on=objectives, how='inner')['index']
        pareto_front = pareto_df.loc[pareto_front_indices]; #print('pareto_front\n', pareto_front)

        return pareto_front

    def find_pareto_front(self, df, knobs, objectives, precision=None):
        pareto_df = df.copy()
        # If precision is specified, round the objectives columns
        if precision is not None:
            pareto_df[objectives] = pareto_df[objectives].round(precision)

        # Create a DataFrame with the first row from each group of unique objective values
        df_groupby_one = pareto_df.drop_duplicates(subset=objectives); #print('df_groupby_one\n', df_groupby_one)

        # Find the Pareto front within df_groupby_one
        data = df_groupby_one[objectives].values
        is_pareto_optimal = np.ones(data.shape[0], dtype=bool)
        for i, candidate in enumerate(data):
            if is_pareto_optimal[i]:
                is_pareto_optimal[is_pareto_optimal] = np.any(data[is_pareto_optimal] > candidate, axis=1)
                is_pareto_optimal[i] = True  # Keep the current candidate

        # Select the Pareto optimal rows from df_groupby_one
        pareto_groupby_one = df_groupby_one[is_pareto_optimal]; #print('pareto_groupby_one\n', pareto_groupby_one)

        # Expand the Pareto front by including all rows from the original DataFrame
        # that correspond to the Pareto optimal rows in pareto_groupby_one (where "correspomd" 
        # means that all objectives values in the correponding rows must be the same)
        pareto_front_indices = pareto_df.reset_index().merge(pareto_groupby_one, on=objectives, how='inner')['index']
        pareto_front = pareto_df.loc[pareto_front_indices]; #print('pareto_front\n', pareto_front)

        return pareto_front
    
    # find sunset of data samples where response_columns have values greater or equal to thresholds: that is,
    # constraints response_columns[i] >= thresholds[i] hold for each i in range(len(response_columns)).
    def filter_by_thresholds(self, df, response_columns, thresholds):
        # Check if the lengths of the response columns and thresholds match
        if len(response_columns) != len(thresholds):
            raise ValueError("The length of response_columns and thresholds must be the same.")
        #print('df before filtering', df.shape, df.columns.tolist())
        # Create a boolean mask for each response-threshold pair
        masks = []
        for response, threshold in zip(response_columns, thresholds):
            if response not in df.columns:
                raise ValueError(f"The response column '{response}' is not in the DataFrame.")
            masks.append(df[response] >= threshold)

        # Combine all masks using the logical AND operator
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask

        # Filter the DataFrame using the combined mask
        filtered_df = df[combined_mask]
        #print('filtered_df', filtered_df.shape, filtered_df.columns.tolist())
        return filtered_df

    #Creates a boolean mask from a string expression with logical operators.
    #Parameters:
    #expression (str): The string expression with logical operators ('and', 'or', 'not', '==', '!=', '<', ...).
    #dataframe (pd.DataFrame): The pandas DataFrame to create the mask for.
    #Returns: pd.Series: The boolean mask.
    def create_boolean_mask(self, expression, dataframe):
        # Replace logical operators with bitwise operators
        mask_expression = (expression
                           .replace(' and ', '&')
                           .replace(' or ', '|')
                           .replace(' not ', '~'))  # Include a space after 'not' to avoid replacing 'not' in variable names

        # Attempt to evaluate the expression directly as a Python expression
        try:
            # If the expression evaluates to a single boolean value, create a uniform mask
            evaluated_expression = eval(mask_expression)
            if isinstance(evaluated_expression, bool):
                return pd.Series([evaluated_expression] * len(dataframe), index=dataframe.index)
        except NameError:
            # If the expression cannot be evaluated directly, proceed to use the DataFrame's eval
            pass

        # Use the DataFrame's eval method to create the boolean mask
        try:
            mask = dataframe.eval(mask_expression)
        except KeyError as e:
            raise KeyError(f"Error in expression: {e}. Make sure the expression is valid and column names are correct.")

        return mask

    #This function Creates new columns in the DataFrame df based on provided expressions and names.
    #Parameters:
    #df (pd.DataFrame): The DataFrame to extend with new objective columns.
    #objv_names (list of str): The names of the new objective columns.
    #objv_expr (list of str): The expressions to compute the values of the new columns.
    #Returns: pd.DataFrame: The extended DataFrame with new objective columns.
    def compute_objectives_columns(self, df:pd.DataFrame, objv_names:list[str], objv_exprs:list[str]):
        # Check if the lengths of objv_names and objv_exprs match
        if len(objv_names) != len(objv_exprs):
            raise ValueError("The lengths of objv_names and objv_exprs must match.")

        # Create a copy of the DataFrame to avoid modifying the original
        objv_df = df.copy()

        # Iterate over the provided names and expressions
        for name, expr in zip(objv_names, objv_exprs):
            # Evaluate the expression in the context of the DataFrame
            objv_df[name] = objv_df.eval(expr)

        return objv_df


    # Generate mask to filter datframe based on a condition specified as a python expression (string).
    # Return the filtered dataset.
    def filter_by_expression(self, df, expression:str):
        if expression is None:
            return df
        #print('condition', expression); print('df', df.shape, df.columns.tolist())
        #print('df before filtering', df.shape, df.columns.tolist())
        # Create a boolean mask for each response-threshold pair        
        mask = self.create_boolean_mask(expression, df)
        filtered_df = df[mask]; #print('filtered_df', filtered_df.shape, filtered_df.columns.tolist())
        return filtered_df
    
    def filter_beta_universal(self, df, beta_expr, knobs, resps):
        # Step 1: Find indices of df that do not satisfy beta_expr
        neg_beta_expr = f"not ({beta_expr})"
        df_neg_beta = df.query(neg_beta_expr)
        neg_beta_indices = df_neg_beta.index

        # Step 2: Find all unique tuples of values of knobs from knobs values at these row indices
        unique_knob_tuples = df_neg_beta[knobs].drop_duplicates()

        # Step 3: Drop all rows that have these tuples as values in the knobs columns
        # Create a mask to identify rows to drop
        mask_to_drop = pd.Series(False, index=df.index)
        for _, row in unique_knob_tuples.iterrows():
            mask = pd.Series(True)
            for knob in knobs:
                mask &= (df[knob] == row[knob])
            mask_to_drop |= mask

        # Step 4: Assert that the indices of rows that will be dropped is a superset of the indices computed at step (1)
        #print('neg_beta_indices', neg_beta_indices, 'df[mask_to_drop].index', df[mask_to_drop].index)
        assert set(neg_beta_indices).issubset(set(df[mask_to_drop].index)), "Assertion failed: indices to drop should include all indices that do not satisfy beta_expr"

        # Drop the rows and return the resulting DataFrame
        df_beta = df[~mask_to_drop]
        return df_beta

    # pareto subset selection directly from data, without building a model.
    def select_pareto_frontier(self, X:pd.DataFrame, y:pd.DataFrame, model_features_dict:dict, 
            feat_names:list[str], resp_names:list[str], 
            objv_names:list[str], objv_exprs, pareto:bool, strategy:str, #asrt_names:list[str], asrt_exprs, 
            quer_names:list[str], quer_exprs, delta:float, epsilon:float, 
            alph_expr:str, beta_expr:str, eta_expr:str, theta_radii_dict:dict):
        self._frontier_logger.info('Pareto frontier selection in data: Start')
        assert epsilon > 0 and epsilon < 1
        assert objv_names is not None and objv_exprs is not None
        assert len(objv_names) == len(objv_exprs)
        
        #print('X', X.shape, 'y', y.shape)
        df = pd.concat([X, y], axis=1); #print('df', df.shape, df.columns.tolist())
        df_shape = df.shape
        
        # Step 1: drop irrelevant data t-- samples that do not satisfy alpha constraints 
        df_alpha = self.filter_by_expression(df, alph_expr); #print('df_alpha', df_alpha.shape, df_alpha.columns.tolist()); print(df_alpha)
        df_alpha_shape = df_alpha.shape
        assert df_alpha_shape[0] <= df_shape[0]; assert df_alpha_shape[1] == df_shape[1]
        
        # Step 2: Compute objectives columns and then min-objectives columns by taking into account stability radii
        df_objv = self.compute_objectives_columns(df_alpha, objv_names, objv_exprs); #print('df_objv', df_objv.shape, df_objv.columns.tolist()); print(df_objv)
        df_stable_min = self.local_minimum(df_objv, feat_names, resp_names, objv_names, self._specInst.get_spec_stability_intervals_dict)
        df_stable_min_shape = df_stable_min.shape; #print('df_stable_min', df_stable_min.shape, df_stable_min.columns.tolist()); print(df_stable_min)
        
        knobs = self._specInst.get_spec_knobs; #print('knobs', knobs)
        
        # Step 3: Drop all samples that do not satisfy beta constraints. This needs to be done befre pareto subset selection
        #df_beta = self.filter_by_expression(df_stable_min, beta_expr); print('df_beta', df_beta.shape, df_beta.columns.tolist()); print(df_beta)
        df_beta = self.filter_beta_universal(df_stable_min, beta_expr, knobs, resp_names)
        
        # Step 4: Select pareto subset with respect to maximizing min-objective values
        
        pareto_frontier = self.find_pareto_front(df_beta, knobs, objv_names); #print('pareto_frontier', pareto_frontier.shape, pareto_frontier.columns.tolist()); print(pareto_frontier)
        pareto_frontier_csv_file = self.report_file_prefix + '_pareto_frontier.csv'
        pareto_frontier.to_csv(pareto_frontier_csv_file, index=False)
        self._frontier_logger.info('Pareto frontier selection in data: End')