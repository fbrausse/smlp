import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import random
import shutil
import os
import subprocess
import plotly.graph_objects as go
from icecream import ic
from matplotlib.colors import Normalize
import torch
import sys

# Configure icecream for debugging output
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

exp = str(round(random.random(), 4))  # Generate a random experiment identifier

class plot_exp:

    def __init__(self, exp=exp, setno='10'):

        self.exp = exp
        self.setno = setno
        self.txt_file = f'Set{self.setno}_{self.exp}_experiments.txt'
        self.Set = f'experiment_outputs/Set{self.setno}/Set_{self.setno}_'
        self.witnesses_json = f'Set{self.setno}_{self.exp}_witnesses.json'
        self.orig_csv = "smlp_toy_basic.csv"
        self.witnesses_html_path = f'Set{self.setno}_{self.exp}_witnesses.html'
        self.stable_x_original_html_path = f'Set{self.setno}_{self.exp}_stable_x_original.html'
        self.stable_x_counter_html_path = f'Set{self.setno}_{self.exp}_stable_x_counter.html'
        self.opt_out = f'Set{self.setno}_{self.exp}_optimization_output.png'
        self.source_file_1 = f'experiment_outputs/Set{self.setno}/smlp_toy_basic.csv'
        self.source_file_2 = f'experiment_outputs/Set{self.setno}/smlp_toy_basic.spec'
        self.destination_folder = '.'    
        self.spec_file = "smlp_toy_basic.spec"
        self.orig_data = pd.read_csv('smlp_toy_basic.csv')
        self.x_bounds = f"\nmin x: {self.orig_data.iloc[:, 0].min()} max x: {self.orig_data.iloc[:, 0].max()}"
        self.y_bounds = f"\nmin y: {self.orig_data.iloc[:, 1].min()} max y: {self.orig_data.iloc[:, 1].max()}"
        if len(self.orig_data.columns) > 2:
            self.z_bounds = f"\nmin z: {self.orig_data.iloc[:, 2].min()} max z: {self.orig_data.iloc[:, 2].max()}"
        self.orig_len = f"\nlength of dataset: {len(self.orig_data)}"

    def save_to_txt(self, data):
        """
        Saves data and dataset statistics to a text file.
        
        Parameters:
        - data: Data to be saved. Can be a list or a single value.
        """
        # Save data to text file
        with open(self.txt_file, 'a') as f:
            if isinstance(data, list):
                f.write(f"{self.setno}\n{self.exp}\n")
                for index, arg in enumerate(data[1:], start=1):
                    f.write(f"Argument {index}: {arg}\n")
                f.write(self.x_bounds + self.y_bounds + (self.z_bounds if len(self.orig_data.columns) > 2 else '') + self.orig_len + "\n \n")
                f.flush()
    
                resp_index = data.index('-resp')
                resp = data[resp_index + 1]
                feat_index = data.index('-feat')
                feat = data[feat_index + 1]
                resp_list = resp.split(',')
                feat_list = feat.split(',')
                with open(self.spec_file, 'r') as spec:
                    spec = json.load(spec)
                for key, value in spec.items():
                    if key == 'variables':
                        for v in value:
                            for ite, val in v.items():
                                if str(ite) == "rad-abs":
                                    f.write(f"For {v['label']} {ite}: {val}\n")
    
            else:
                f.write(f"\n{data}\n")
                f.flush()

    def param_changed(self, hparams_dict, algo, n):
        """
        Logs changed hyperparameters compared to default values.
    
        Parameters:
        - hparams_dict: Dictionary of hyperparameters to check.
        - algo: Algorithm name to filter relevant parameters.
        - n: Section in default_params.json to compare against.
        """
        # Load default parameters from JSON
        with open('default_params.json', 'r') as file:
            default_dict = json.load(file)
    
        default_dict = default_dict[n]
    
        # Check for changes in hyperparameters
        if n == 2:
            for k in hparams_dict:
                if k in hparams_dict and k in default_dict:
                    if hparams_dict[k] != default_dict[k]:
                        param = f"{k}: {hparams_dict[k]}"
                        self.save_to_txt(param)
                        ic(param)

        else: 
            for k in hparams_dict:
                if algo in k and k in default_dict:
                    if hparams_dict[k] != default_dict[k]:
                        param = f"{k}: {hparams_dict[k]}"
                        self.save_to_txt(param)

    def save_time(self, t, times=[]):
        """
        Saves the time taken for training and optimization to a text file.
    
        Parameters:
        - t: Current time.
        - times: List of times, should include start and end times for training and optimization.
        """
        times.append(t)
        if len(times) == 2:
            # Calculate training time
            train_time = f"Training time: {times[1] - times[0]}"
            # Save times to text file
            self.save_to_txt(train_time)
    
        elif len(times) == 4:
            # Calculate optimization time
            syn_time = f"Optimization synthesis feasibility check time: {times[3] - times[2]}"
            # Save times to text file
            self.save_to_txt(syn_time)
    
        elif len(times) == 6:
            # Calculate optimization time
            opt_time = f"Pareto optimization completion time: {times[5] - times[4]}"
            # Save times to text file
            self.save_to_txt(opt_time)

    # Plot and save prediction data
    def prediction_save(self, X_test, y_test_pred, mm_scaler_resp):
        """
        Plots test data against original data and saves the plot.
        
        Parameters:
        - X_test: Test features.
        - y_test_pred: Predicted values.
        - mm_scaler_resp: Scaler used to inverse transform predictions.
        """
        ind = X_test.index
        data_version = 'test'
        
        # Inverse transform the data
        y_test_pred = mm_scaler_resp.inverse_transform(y_test_pred)
        X_test = mm_scaler_resp.inverse_transform(X_test)
        
        # Convert to DataFrame
        y_test_pred = pd.DataFrame(y_test_pred)
        X_test = pd.DataFrame(X_test)
        
        # Load original data
        orig_file = pd.read_csv("smlp_toy_basic.csv")
        
        # Extract original and predicted data
        x = orig_file.iloc[ind, :-1]
        y = y_test_pred.iloc[:, 0]
        
        # Create prediction DataFrame
        prediction_df = pd.concat([x, y], axis=1)
        
        # Call the main plotting function
        self.plott(x, y, data_version)

    def copy_from(self):
        """
        Copies source files to the current directory.
        """
        if not torch.cuda.is_available():
            ic("CUDA is not available. Exiting program.")
            sys.exit(1)  # Exit with a non-zero status code

        # Your CUDA-dependent code here
        ic("CUDA is available. Proceeding with the program.")
        cuda = "CUDA is available. Proceeding with the program."
        self.save_to_txt(cuda)
        
        #ic(torch.cuda.is_available())
        #ic(torch.version.cuda)
        #ic(torch.__version__)
        shutil.copy2(self.source_file_1, self.destination_folder)
        shutil.copy2(self.source_file_2, self.destination_folder)

    def save_to_dict(self, data, data_version):
        """
        Saves model precision data to a .json file. Appends if the file exists.
    
        Parameters:
        - data: Dictionary containing data to be saved.
        - data_version: Indicates which version of data is being saved.
        """
        init_data = {}
        if os.path.exists(self.witnesses_json):
            with open(self.witnesses_json, 'r') as file:
                init_data = json.load(file)
                if data_version in init_data:
                    for key in data:
                        if key not in init_data[data_version]:
                            init_data[data_version][key] = []
                        init_data[data_version][key].append(data[key])

                else:
                    init_data[data_version] = {key: [value] for key, value in data.items()}
        else:
            init_data[data_version] = {key: value for key, value in data.items()}

        with open(self.witnesses_json, 'w') as file:
            json.dump(init_data, file, indent=4)

        if data_version == 'bounds':
            ic(init_data[data_version])

    def plott(self, x, y, data_version):
        """
        Main function to create and save plots based on data version.
        
        Parameters:
        - x: X data for plotting.
        - y: Y data for plotting.
        - data_version: Version of data to determine plot type.
        """
        # Load original data
        orig_data = pd.read_csv('smlp_toy_basic.csv')
        
        if len(orig_data.columns) > 2:
            if data_version == 'optimized':
                z_orig = orig_data.iloc[:, 2]
                y_orig = orig_data.iloc[:, 1]
                x_orig = orig_data.iloc[:, 0]
                x_additional = x.iloc[:, 0]
                y_additional = x.iloc[:, 1]
                z_additional = y
                
                # Create 3D scatter plot
                fig1 = go.Figure(data=[
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='red', opacity=0.25), name='Original data'),
                    go.Scatter3d(x=x_additional, y=y_additional, z=z_additional, mode='markers', marker=dict(color='blue'), name='Optimal value')
                ])
                
                # Update layout
                fig1.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of Optimal values on Original dataset'
                )
                
                # Save figure as HTML
                fig1.write_html(f"Set{self.setno}_{self.exp}_scatter_plot_optimized.html")
            
            else:

                z = y
                y = x.iloc[:, 1]
                x = x.iloc[:, 0]

                data = {'x': x.tolist(), 'y': y.tolist(), 'z':z.tolist()}
                self.save_to_dict(data, data_version)

                # Create 3D scatter plot
                fig0 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'))])
                
                # Update layout
                fig0.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of Predicted data on test set'
                )
                
                # Save figure as HTML
                fig0.write_html(f"Set{self.setno}_{self.exp}_scatter_plot_predicted.html")
        
        else:
            plt.scatter(x, y, color='#0000ff', marker='x', s=5, label='Original data')
            x = orig_data.iloc[:, 0]
            y = orig_data.iloc[:, 1]
            plt.scatter(x, y, color='#ff0000', marker='o', s=2, label='Model Reconstruction', alpha=0.9)
            plt.xlabel('X')
            plt.ylabel('Y', rotation=0)
            plt.title('Scatter Plot representing Original data and Model Reconstruction of Original Data/Graph')
            plt.grid(True)
            plt.legend()
            plt.savefig(f"{self.setno}{self.exp}_{data_version}.png")

    def unscale(self, b):
        data_version = 'bounds'
        min_value = self.orig_data.iloc[:, 2].min()
        max_value = self.orig_data.iloc[:, 2].max()

        # Iterate over the dictionary items
        for key, value in b.items():
            # Calculate the new value
            new_value = value * (max_value - min_value) + min_value
            # Update the dictionary with the new value
            b[key] = new_value
        ic(b)
        self.save_to_dict(b, data_version)

    def witnesses(self):
        """
        Creates and saves plots based on witnesses data and original data.
        """
        orig = pd.read_csv(self.orig_csv)
    
        if os.path.exists(self.witnesses_json):
    
            with open(self.witnesses_json, 'r') as file:
                data = json.load(file)

            num_witn = "Number of witnesses explored: " + str(len(data['witnesses']['x'][:]))
            self.save_to_txt(num_witn)

            z_orig = data['test']['z']
            y_orig = data['test']['y']
            x_orig = data['test']['x']

            if 'counter' in data and 'stable' in data:
                z_counter = data['counter']['z'][:] 
                y_counter = data['counter']['y'][:] 
                x_counter = data['counter']['x'][:] 

                z_sat = data['stable']['z'][:] 
                y_sat = data['stable']['y'][:] 
                x_sat = data['stable']['x'][:] 

                # Create 3D scatter plot
                fig1 = go.Figure(data=[
                    go.Scatter3d(x=x_sat, y=y_sat, z=z_sat, mode='markers', marker=dict(color='red'), name='stable witness'),
                    go.Scatter3d(x=x_counter, y=y_counter, z=z_counter, mode='markers', marker=dict(color='blue'), name='counter example'),
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='original data')
                    #go.Scatter3d(x=x_counter, y=y_counter, z=z_counter, mode='markers', marker=dict(color=z_counter, colorscale='Hot', colorbar=dict(title='title')), name='Optimal value')
                ])
                
                # Update layout
                fig1.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of counter examples and stable witnesses on original data'
                )
                
                # Save figure and open HTML file
                fig1.write_html(self.stable_x_counter_html_path)

                z = data['witnesses']['z'][:] 
                y = data['witnesses']['y'][:] 
                x = data['witnesses']['x'][:] 

                fig2 = go.Figure(data=[
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='Original data'),
                    go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=z, colorscale='Hot', colorbar=dict(title='title')), name='witnesses')
                ])
                
                # Update layout
                fig2.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of witnesses on Original dataset'
                )
                
                # Save figure and open HTML file
                fig2.write_html(self.witnesses_html_path)

            elif 'stable' in data:

                z_sat = data['stable']['z'][:] 
                y_sat = data['stable']['y'][:] 
                x_sat = data['stable']['x'][:] 

                z = data['witnesses']['z'][:] 
                y = data['witnesses']['y'][:] 
                x = data['witnesses']['x'][:] 

                fig2 = go.Figure(data=[
                    go.Scatter3d(x=x_sat, y=y_sat, z=z_sat, mode='markers', marker=dict(color='red'), name='stable witness'),
                    go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'), name='witnesses'),
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='original data')
                    #go.Scatter3d(x=x_counter, y=y_counter, z=z_counter, mode='markers', marker=dict(color=z_counter, colorscale='Hot', colorbar=dict(title='title')), name='Optimal value')
                ])

                # Update layout
                fig2.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of stable witnesses on Original dataset'
                )
                
                # Save figure and open HTML file
                fig2.write_html(self.stable_x_original_html_path)

            else:

                z = data['witnesses']['z'][:] 
                y = data['witnesses']['y'][:] 
                x = data['witnesses']['x'][:] 

                fig2 = go.Figure(data=[
                    go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='red', opacity=0.5), name='Original data'),
                    go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color=z, colorscale='Hot', colorbar=dict(title='title')), name='Stable value')
                ])
                
                # Update layout
                fig2.update_layout(
                    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                    title='Scatter Plot of witnesses on Original dataset'
                )
                
                # Save figure and open HTML file
                fig2.write_html(self.witnesses_html_path)

        else:
            ic("No witnesses to plot")

def copy_data(setno):
    """
    Copies relevant data files from source to destination folder.
    
    Parameters:
    - setno: Set number for paths.
    """
    source_folder = "."
    destination_folder = f'experiment_outputs/Set{self.setno}/'
    Set = f'experiment_outputs/Set{self.setno}/'
    source_folders = 'toy_out_dir'
    extensions = ['.png', '.html', '.txt']
    
    shutil.copytree(source_folders, Set + 'toy_out_dir', dirs_exist_ok=True)
    
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Copy files with specific extensions
        for file in os.listdir(source_folder):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                source_file_path = os.path.join(source_folder, file)
                destination_file_path = os.path.join(destination_folder, file)
                shutil.copy2(source_file_path, destination_file_path)
                os.remove(source_file_path)
                
    except FileNotFoundError:
        print("Source folder not found.")

