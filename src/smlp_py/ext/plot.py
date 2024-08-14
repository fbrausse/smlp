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

# Configure icecream for debugging output
ic.configureOutput(prefix=f'Debug | ', includeContext=True)

# Constants
setno = '1'
exp = str(round(random.random(), 4))  # Generate a random experiment identifier
ic(exp)

Set = f'experiment_outputs/Set{setno}/Set_{setno}_'
witnesses_csv_path = f'Set{setno}_{exp}_witnesses.csv'
witnesses_html_path = f'Set{setno}_{exp}_witnesses.html'
#txt_file = f'Set{setno}_{exp}_experiments.txt'
opt_out = f'Set{setno}_{exp}_optimization_output.png'

class plot_exp:

    def __init__(self):

        self.txt_file = f'Set{setno}_{exp}_experiments.txt'

    def save_to_txt(self, data):
        """
        Saves data and dataset statistics to a text file.
        
        Parameters:
        - data: Data to be saved. Can be a list or a single value.
        """
        global setno, exp
        
        # Load original data for bounds
        orig_data = pd.read_csv('smlp_toy_basic.csv')
        x_bounds = f"\nmin x: {orig_data.iloc[:, 0].min()} max x: {orig_data.iloc[:, 0].max()}"
        y_bounds = f"\nmin y: {orig_data.iloc[:, 1].min()} max y: {orig_data.iloc[:, 1].max()}"
        
        if len(orig_data.columns) > 2:
            z_bounds = f"\nmin z: {orig_data.iloc[:, 2].min()} max z: {orig_data.iloc[:, 2].max()}"
    
        orig_len = f"\nlength of dataset: {len(orig_data)}"
    
        # Save data to text file
        with open(self.txt_file, 'a') as f:
            if isinstance(data, list):
                f.write(f"{setno}\n{exp}\n")
                for index, arg in enumerate(data[1:], start=1):
                    f.write(f"Argument {index}: {arg}\n")
                f.write(x_bounds + y_bounds + (z_bounds if len(orig_data.columns) > 2 else '') + orig_len + "\n")
    
                resp_index = data.index('-resp')
                resp = data[resp_index + 1]
                feat_index = data.index('-feat')
                feat = data[feat_index + 1]
                resp_list = resp.split(',')
                feat_list = feat.split(',')
                #if len(feat_list) > 1:
                #    for f in feat_list:
                #        ic(f)
                #else:
                #    ic(feat_list[0])
    
                #if len(resp_list) > 1:
                #    for f in resp_list:
                #        ic(f)
                #else:
                #    ic(resp_list[0])
    
            else:
                f.write(f"\n{data}\n")

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
            ic(train_time)
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
            ic(opt_time)
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
        main(x, y, data_version)

    def witnesses(self, lower_bound, solver):
        """
        Creates and saves plots based on witnesses data and original data.
        """
        orig = pd.read_csv('/home/x/temp/smlp/smlp_toy_basic.csv')
    
        if os.path.exists(witnesses_csv_path):
    
            df = pd.read_csv(witnesses_csv_path)
            df.drop_duplicates(inplace=True)
            df.to_csv(witnesses_csv_path, index=False)
            num_witn = "Number of witnesses explored: " + str(len(df))
            self.save_to_txt(num_witn)
    
            if solver == "sat":
    
                lower_bound = float(lower_bound['objective'])
                # Check if 3D plotting is required
                if len(orig.columns) > 2:
                    
                    # Extract data for plotting
                    z_orig = orig.iloc[:, 2]
                    y_orig = orig.iloc[:, 1]
                    x_orig = orig.iloc[:, 0]
                    z_additional = df.iloc[:, 2]
                    y_additional = df.iloc[:, 1]
                    x_additional = df.iloc[:, 0]
                    
                    # Create 3D scatter plot
                    fig1 = go.Figure(data=[
                        go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='Original data'),
                        go.Scatter3d(x=x_additional, y=y_additional, z=z_additional, mode='markers', marker=dict(color=z_additional, colorscale='Hot', colorbar=dict(title='title')), name='Optimal value')
                    ])
                    
                    # Update layout
                    fig1.update_layout(
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                        title='Scatter Plot of Optimal values on Original dataset'
                    )
                    
                    # Save figure and open HTML file
                    fig1.write_html(witnesses_html_path)
                    subprocess.run(['xdg-open', os.path.abspath(witnesses_html_path)], check=True)
                
                else:
                    
                    # Extract data for plotting
                    x1 = orig.iloc[:, 0]
                    y1 = orig.iloc[:, 1]
                    x = df.iloc[:, 0]
                    y = df.iloc[:, 1]
                    
                    distances = np.abs(y - lower_bound)
                    norm = Normalize(vmin=distances.min(), vmax=distances.max())
                    # Reset index for consistency
                    x.reset_index(drop=True, inplace=True)
                    y.reset_index(drop=True, inplace=True)
                    
                    # Create scatter plot
                    plt.scatter(x1, y1, color='#8ad347', label='Objective function')
                    #plt.scatter(x, y, color='#00ffff', label='Witnesses')
                    scatter = plt.scatter(x, y, c=distances, cmap='viridis', norm=norm, label='Witnesses', edgecolor='k')
    
                    cbar = plt.colorbar(scatter, label='Distance from approximated maximum')
    
                    plt.axhline(y=lower_bound, color='0', linestyle='--', label=f'Threshold lower bound (y = {lower_bound})')
                    plt.xlabel('X')
                    plt.ylabel('Y', rotation=0)
                    plt.title('Optimization')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(opt_out)
    
            else: 
                # Check if 3D plotting is required
                if len(orig.columns) > 2:
                    # Extract data for plotting
                    z_orig = orig.iloc[:, 2]
                    y_orig = orig.iloc[:, 1]
                    x_orig = orig.iloc[:, 0]
                    z_additional = df.iloc[:, 2]
                    y_additional = df.iloc[:, 1]
                    x_additional = df.iloc[:, 0]
                    
                    # Create 3D scatter plot
                    fig1 = go.Figure(data=[
                        go.Scatter3d(x=x_orig, y=y_orig, z=z_orig, mode='markers', marker=dict(color='grey', opacity=0.5), name='Original data'),
                        go.Scatter3d(x=x_additional, y=y_additional, z=z_additional, mode='markers', marker=dict(color=z_additional, colorscale='Hot', colorbar=dict(title='title')), name='Optimal value')
                    ])
                    
                    # Update layout
                    fig1.update_layout(
                        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                        title='Scatter Plot of Optimal values on Original dataset'
                    )
                    
                    # Save figure and open HTML file
                    fig1.write_html(witnesses_html_path)
                    subprocess.run(['xdg-open', os.path.abspath(witnesses_html_path)], check=True)
                
                else:
                    # Extract data for plotting
                    x1 = orig.iloc[:, 0]
                    y1 = orig.iloc[:, 1]
                    x = df.iloc[:, 0]
                    y = df.iloc[:, 1]
                    
                    # Reset index for consistency
                    x.reset_index(drop=True, inplace=True)
                    y.reset_index(drop=True, inplace=True)
                    
                    # Create scatter plot
                    plt.scatter(x1, y1, color='#8ad347', label='Objective function')
                    plt.scatter(x, y, color='#00ffff', label='Witnesses')
                
                    plt.xlabel('X')
                    plt.ylabel('Y', rotation=0)
                    plt.title('Optimization')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(opt_out)
        else:
            ic("No witnesses to plot")


def save_to_csv(data, data_version):
    """
    Saves model precision data to a CSV file. Appends if the file exists.

    Parameters:
    - data: Dictionary containing data to be saved.
    - data_version: Indicates which version of data is being saved.
    """
    global setno, exp
    # Create DataFrame based on data version
    if data_version == 'witnesses':
        if len(data) == 3:
            df = pd.DataFrame({'x': data['x'], 'y': data['y'], 'z': data['z']}, index=[0])
        else:
            df = pd.DataFrame({'x': data['x'], 'y': data['y']}, index=[0])

        # Append to or create the CSV file
        if os.path.exists(witnesses_csv_path):
            df.to_csv(witnesses_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(witnesses_csv_path, mode='w', header=True, index=False)

def copy_from():
    """
    Copies source files to the current directory.
    """
    global setno
    source_file_1 = f'experiment_outputs/Set{setno}/smlp_toy_basic.csv'
    source_file_2 = f'experiment_outputs/Set{setno}/smlp_toy_basic.spec'
    destination_folder = '.'    
    
    shutil.copy2(source_file_1, destination_folder)
    shutil.copy2(source_file_2, destination_folder)

def copy_data(setno):
    """
    Copies relevant data files from source to destination folder.
    
    Parameters:
    - setno: Set number for paths.
    """
    source_folder = "."
    destination_folder = f'experiment_outputs/Set{setno}/'
    Set = f'experiment_outputs/Set{setno}/'
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

def main(x, y, data_version):
    """
    Main function to create and save plots based on data version.
    
    Parameters:
    - x: X data for plotting.
    - y: Y data for plotting.
    - data_version: Version of data to determine plot type.
    """
    global setno
    
    # Load original data
    orig_data = pd.read_csv('smlp_toy_basic.csv')
    Set = f'Set{setno}_'
    
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
            fig1.write_html(f"Set{setno}_{exp}_scatter_plot_optimized.html")
        
        else:
            z = y
            y = x.iloc[:, 1]
            x = x.iloc[:, 0]
            
            # Create 3D scatter plot
            fig0 = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(color='blue'))])
            
            # Update layout
            fig0.update_layout(
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                title='Scatter Plot of Predicted data on test set'
            )
            
            # Save figure as HTML
            fig0.write_html(f"Set{setno}_{exp}_scatter_plot_predicted.html")
    
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
        plt.savefig(f"{Set}{exp}_{data_version}.png")

