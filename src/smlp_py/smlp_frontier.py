# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

import numpy as np
import pandas as pd
from smlp_py.smlp_spec import SmlpSpec

'''
Pareto frontier selection with respect to maximizing objectives, directly from data, without training models.
The algorithm works as follows:
1. drop from data all samples that do not satisfy alpha constraints. TODO !!! could also drop here rows
   that do not satisfy eta constraints -- while eta domain constraints are taken into account at step 3, eta
   grid constraints are never looked at. Maybe drop here also the samples that do not satisfy beta constraints?
2. compute the objectives' columns, say objv1, objv2, based on the spec that defines the objectives' expressions.
3. compute columns min-objectives columns min_onjv1, min_objv2 that will hold, in each row, the minimum values of 
respective objectives in the subset of data that falls in the stability region of the configuration defined by the
values of knobs in that row. This step is performed as follows:
    a. iterate over all rows in the data. In each step of the iteration, for a given row, the algorithm computes 
       values of min-objectives min_onjv1, min_obkv2 for that row, as follows:
       1. compute the stability region of the configuration of knob values in that row using the stability radii, 
          where stability intervals for knobs are intersected with respective domains (intervals) defined in spec file, 
          and for inputs the stability intervals are chosen based on their domains (intervals) as defined in spec file.
          At this point, the stability region (the condition/formula that define it) satisfies alpha and eta domain 
          constraints, and we further intersect this condition (take conjunction) with global alpha constraints/
          Note that doe to intersecting the stability regions of knobs with respective alpha and eta constraints, 
          the stability region might become empty (false predicate that cannot select any row in data).
       2. Apply the stability region to entire data to compute a subset of data points that fall within the stability
          region (this is for the given row in data, as part of iterations over all rows in item a).
       3. Compute minimum values of the objectives' columns for that subset of data, and assign this values as the 
          values of the respective min-objective columns min_onjv1, min_objv2, in current row. Note that the subset of
          data used to compute the min values is empty, the min values should be inf (positive infinity), while python
          returns nan values instead, and these values will be added to fill in min-objective columns in current row.
    b. after completing iterating all rows, the min-objective columns are all assigned values, and we drop from data
       all samples where at least one min-objective has value nan. In fact, in such rows values of all min-objective 
       columns must be nan, and this check is imposed as assertion in the implementation.
4. At this point we have dataset with min-objective columns and their values are all valid values (not nan). We drop
   from this dataset all samples that do not satisfy the beta constraints (the synthesis requirements), because
   these data points cannot be part of pareto optimal frontier (these samples do not represent a valid/legal 
   configuration of knobs).
5. The algorithm now selects paeto frontier from this data, with respect to maximizing the min-objective column
   values. This is done again by iterating over all rows, and for each row checking whether there is another row
   that dominates the current row (all values of min-objective columns in a dominating row are equal or higher than the
   values in the current row. If such other row exists, the current row is not added to the set of pareto frontier samples.

'''



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

    '''
    # Function to replace None with the smallest value based on the column's dtype.
    # This code snippet first determines the dtypes for each column, taking into account that if a column 
    # contains None and is numeric, it should be treated as a float. Then, it iterates over the columns and
    # replaces None with the smallest possible integer for integer columns and with -np.inf for non-integer columns. 
    # Finally, it converts the object array into a structured array with the correct dtypes for each column.
    # Please note that this code assumes that the lower_bounds array may contain columns of different dtypes, 
    # and it handles integer and non-integer columns differently. If you know in advance which columns are integers, 
    # you can optimize the code by directly replacing None values in those specific columns.
    def replace_none_with_min_value(self, lower_bounds, lower):
        # Determine the dtypes for each column
        dtypes = [col.astype(float).dtype if np.issubdtype(col.dtype, np.number) and np.any(pd.isnull(col)) else col.dtype for col in lower_bounds.T]
        #print('lower_bounds dtypes', dtypes)
        # Replace None with the smallest value for the dtype
        for i, dtype in enumerate(dtypes):
            if np.issubdtype(dtype, np.integer):
                # Get the smallest integer for the dtype
                if lower:
                    min_int = np.iinfo(dtype).min
                else:
                    min_int = np.iinfo(dtype).max
                # Replace None with the smallest integer
                lower_bounds[:, i] = np.where(lower_bounds[:, i] == None, min_int, lower_bounds[:, i])
            else:
                # For non-integer columns, replace None with -np.inf
                if lower:
                    lower_bounds[:, i] = np.where(lower_bounds[:, i] == None, -np.inf, lower_bounds[:, i])
                else:
                    lower_bounds[:, i] = np.where(lower_bounds[:, i] == None, np.inf, lower_bounds[:, i])
        return lower_bounds
#         # Convert the object array to a structured array with the correct dtypes
#         structured_array = np.empty(lower_bounds.shape[0], dtype=[('f{}'.format(i), dt) for i, dt in enumerate(dtypes)])
#         for i in range(len(dtypes)):
#             structured_array['f{}'.format(i)] = lower_bounds[:, i]

#         return structured_array
    '''
    
    # This is an efficient implimentation that is not using df.iterrows()
    # Function local_minimum() computes stable minimum of response columns resp_names for each row.
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
    def local_minimum(self, df, feat_names, resp_names, objv_names, func_intervals):
        #print('local_minimum: df', df.shape)
        # Initialize df_min with the same shape as df and copy feat_names columns
        df_min = pd.DataFrame(index=df.index, columns=df.columns)
        df_min[feat_names] = df[feat_names]; #print('df_min with features\n', df_min)
        df_min[resp_names] = df[resp_names]; #print('df_min', df_min.columns.tolist(), df_min.shape); print(df_min)
        
        # Precompute all intervals for all rows
        #for _, row in df.iterrows():
        #    print('row\n', row, '\ninterval\n', self._specInst.get_spec_stability_intervals_dict(dict(row[feat_names]), True))
        intervals_list = [func_intervals(dict(row[feat_names])) for _, row in df.iterrows()]
        #print('intervals_list', len(intervals_list), intervals_list); 
        #for col_dict in intervals_list:
        #    print("== inervals: ", col_dict.values())
        
        lower_bounds = np.array([[interval[feat_name][0] for feat_name in feat_names] for interval in intervals_list]); 
        upper_bounds = np.array([[interval[feat_name][1] for feat_name in feat_names] for interval in intervals_list])
        #print('lower_bounds', lower_bounds.shape, '\n', lower_bounds); print('upper_bounds', upper_bounds.shape, '\n', upper_bounds)

        '''
        # None as a lower bound for interval represents -inf and as upper bound represents as inf; None is allowed in
        # specifying ranges of inputs and knobs in the spec file, thus we update lower_bounds and upper_bounds as follows:
        if True or any(lower_bounds[lower_bounds == None]):
            lower_bounds = self.replace_none_with_min_value(lower_bounds, True)
            #lower_bounds[lower_bounds == None] = (-np.inf); 
            #print('lower_bounds', lower_bounds.shape, '\n', lower_bounds)
        if True or any(upper_bounds[upper_bounds == None]):   
            upper_bounds = self.replace_none_with_min_value(upper_bounds, False)
            #upper_bounds[upper_bounds == None] = np.inf; 
            #print('upper_bounds', upper_bounds.shape, '\n', upper_bounds)
        '''
        # Compute the minimum values for objv_names columns within the intervals
        for i, (lower, upper) in zip(df.index, zip(lower_bounds, upper_bounds)): #enumerate(zip(lower_bounds, upper_bounds)):
            #print('i, (lower, upper)', i, (lower, upper))
            # Create a boolean mask for rows within the intervals
            mask = np.ones(len(df), dtype=bool)
            for j, feat_name in enumerate(feat_names):
                mask &= (df[feat_name] >= lower[j]) & (df[feat_name] <= upper[j])
            
            #print('df\n', df); print('mask\n', mask); print('objv_names', objv_names)
            # Compute the minimum values for objv_names columns within the mask
            min_values = df.loc[mask, objv_names].min(); #print('min_values\n', min_values)
            df_min.loc[i, objv_names] = min_values; 
            #print('df_min i\n', df_min)
        #print('df_min (1)\n', df_min)
        df_min = self.drop_rows_with_nan_in_columns(df_min, objv_names);  #print('df_min (2)\n', df_min)
        return df_min

    
    # Generate mask to filter datframe based on a condition specified as a python expression (string).
    # Return the filtered dataset.
    # Apply beta constraint to filetr out samples that do not satisfy beta constraints as wel as samples
    # that have other samples not satisfying beta constraints within their stability region.
    def filter_beta_universal(self, df, beta_expr, knobs, resps):
        if beta_expr is None:
            return df
        # Step 1: Find indices of df that do not satisfy beta_expr
        neg_beta_expr = f"not ({beta_expr})"
        df_neg_beta = df.query(neg_beta_expr); #print('df_neg_beta\n', df_neg_beta)
        neg_beta_indices = df_neg_beta.index; #print('neg_beta_indices', neg_beta_indices)
        
        # Step 2: Find all unique tuples of values of knobs from knobs values at these row indices
        unique_knob_tuples = df_neg_beta[knobs].drop_duplicates(); #print('unique_knob_tuples\n', unique_knob_tuples)
        #intervals_list =  [func_intervals(dict(top[1])) for tup in unique_knob_tuples.iterrows()]
        intervals_list = []
        for tup in unique_knob_tuples.iterrows():
            #print('tup', tup, '\ntype', type(tup)); print('dict', dict(tup[1]))
            intervals_list.append(self._specInst.get_spec_stability_intervals_dict(dict(tup[1]), False)); 
        
        # Step 3: Compute the mask that captures samples within stability ragion of samples computed at Step 1.
        # These samples will be dropped as a result of applying beta constrains to data, by taking stability 
        # requirements into account.
        to_drop_mask = np.zeros(len(df), dtype=bool); #print('init mask\n', to_drop_mask)
        for interval in intervals_list:
            interval_mask = True
            for k, (l,h) in interval.items():
                #print('k, (l,h)', k, (l,h))
                curr_mask = (df[k] >= l) & (df[k] <= h); #print('curr_mask\n', curr_mask)
                interval_mask = interval_mask & curr_mask
            #print('interval_mask\n', interval_mask)
            to_drop_mask = to_drop_mask | interval_mask
        #print('final to_drop_mask\n', to_drop_mask)
        
        df_beta = df.loc[~to_drop_mask, : ]; #print('df_beta\n', df_beta)
        return df_beta
        
    # find pareto subset of data with respect to optimization objectives specified in argument 'objectives".
    # pareto subset is with respect to the maximum (optimization means finding maximum, not minimum).
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
        
    import pandas as pd

    def drop_rows_with_nan_in_columns(self, df, objv_names):
        """
        Drops rows where at least one of the columns in objv_names has a NaN value.
        Asserts that in every row that is dropped, all the columns in objv_names have NaN values.

        Parameters:
        df (pd.DataFrame): The DataFrame from which to drop rows.
        objv_names (list): List of column names to check for NaN values.

        Returns:
        pd.DataFrame: The DataFrame with the specified rows dropped.
        """
        # Identify rows where at least one of the columns in objv_names has a NaN value
        rows_with_nan = df[objv_names].isna().any(axis=1)

        # Assert that in every row that is dropped, all the columns in objv_names have NaN values
        rows_to_drop = df[objv_names][rows_with_nan]
        assert rows_to_drop.isna().all(axis=1).all(), "Not all columns in objv_names have NaN values in the rows to be dropped."

        # Drop the rows where at least one of the columns in objv_names has a NaN value
        df_cleaned = df[~rows_with_nan].copy()

        return df_cleaned
    
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
        
        pareto_frontier_csv_file = self.report_file_prefix + '_pareto_frontier.csv'
        
        #print('X', X.shape, 'y', y.shape)
        df = pd.concat([X, y], axis=1); #print('df', df.shape, df.columns.tolist())
        df_shape = df.shape; #print('df shape index', df_shape)
        #print(list(df.index))
        # Step 1: drop irrelevant data t-- samples that do not satisfy alpha constraints 
        df_alpha = self.filter_by_expression(df, alph_expr); #print('df_alpha', df_alpha.shape, df_alpha.columns.tolist()); print(df_alpha)
        df_alpha_shape = df_alpha.shape; #print('df_alpha shape index', df_alpha_shape); 
        #print(list(df_alpha.index))
        assert df_alpha_shape[0] <= df_shape[0]; assert df_alpha_shape[1] == df_shape[1]
        #print(df_alpha, objv_names, objv_exprs)
        # Step 2: Compute objectives columns and then min-objectives columns by taking into account stability radii
        df_objv = self.compute_objectives_columns(df_alpha, objv_names, objv_exprs); #print('df_objv', df_objv.shape, df_objv.columns.tolist()); print(df_objv)
        df_stable_min = self.local_minimum(df_objv, feat_names, resp_names, objv_names, lambda knob_dict: 
            self._specInst.get_spec_stability_intervals_dict(knob_dict, True))
        df_stable_min_shape = df_stable_min.shape; #print('df_stable_min', df_stable_min.shape, df_stable_min.columns.tolist()); print(df_stable_min)
        if df_stable_min_shape[0] == 0:
            self._frontier_logger.warning('Input data does not have rows where input and knob range constraints (alpha and eta, respectively) are satisfied')
            df_stable_min.to_csv(pareto_frontier_csv_file, index=False)
            self._frontier_logger.info('Pareto frontier selection in data: End')
            return df_stable_min
        
        knobs = self._specInst.get_spec_knobs; #print('knobs', knobs)
        
        # Step 3: Drop all samples that do not satisfy beta constraints. This needs to be done befre pareto subset selection
        #df_beta = self.filter_by_expression(df_stable_min, beta_expr); print('df_beta', df_beta.shape, df_beta.columns.tolist()); print(df_beta)
        df_beta = self.filter_beta_universal(df_stable_min, beta_expr, knobs, resp_names); #print('df_beta\n', df_beta)
        if df_beta.shape[0] == 0:
            self._frontier_logger.warning('Input data does not have rows where beta constraints are satisfied in stability regions')
            df_beta.to_csv(pareto_frontier_csv_file, index=False)
            self._frontier_logger.info('Pareto frontier selection in data: End')
            return df_beta
        
        # Step 4: Select pareto subset with respect to maximizing min-objective values
        pareto_frontier = self.find_pareto_front(df_beta, knobs, objv_names); #print('pareto_frontier', pareto_frontier.shape, pareto_frontier.columns.tolist()); print(pareto_frontier)
        
        pareto_frontier.to_csv(pareto_frontier_csv_file, index=False)
        self._frontier_logger.info('Pareto frontier selection in data: End')