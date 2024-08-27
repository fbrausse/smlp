import plotly.graph_objs as go
import pandas as pd
import os
import glob
import subprocess

def plot_csv_files_in_3d(output_html="witness_plot.html"):

    # Get all CSV files in the current directory
    csv_files = glob.glob("../../../*.csv")
    
    # Initialize a list to hold the traces (each CSV will be a different trace)
    traces = []

    # Colors for the different CSV files
    colors = ['#1f77b4',  # blue
              '#ff7f0e',  # orange
              '#2ca02c',  # green
              '#d62728',  # red
              '#9467bd',  # purple
              '#8c564b',  # brown
              '#e377c2',  # pink
              '#7f7f7f',  # gray
              '#bcbd22',  # yellow-green
              '#17becf',  # teal
              '#ffbb78']  # light orange


    for i, file in enumerate(csv_files):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Ensure the CSV file has at least 3 columns
        if len(df.columns) < 3:
            print(f"Error: {file} does not have at least 3 columns.")
            continue

        # Use the first three columns for X, Y, Z
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        z = df.iloc[:, 2]

        # Create a trace for this CSV file
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],  # Cycle through colors if more CSVs than colors
                opacity=0.8
            ),
            name=os.path.basename(file)  # Use the filename as the legend label
        )

        traces.append(trace)

    # Create the layout for the plot
    layout = go.Layout(
        title="3D Scatter Plot of CSV Data",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis"
        ),
        showlegend=True
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Save the figure to an HTML file
    fig.write_html(output_html)
    print(f"3D plot saved as {output_html}")
    subprocess.run(['xdg-open', os.path.abspath(output_html)], check=True)

# Example usage:
plot_csv_files_in_3d(output_html="../../../witness_plot.html")

