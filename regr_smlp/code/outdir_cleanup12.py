#!/usr/bin/env python3

# This script cleans the code directory from SMLP execution output files
from glob import glob
import os
import shutil

code_dir = os.path.dirname(os.path.abspath(__file__))
'''
user_input = input(
    "You are deleting output files from " + code_dir + " directory, are you sure?\n(yes/no|y/n): ").lower()
while user_input not in {'yes', 'no', 'y', 'n'}:
    user_input = input('(yes/no|y/n):').lower()
if user_input in {'yes', 'y'}:
'''
if True:
    txt_ind_files = glob(code_dir + os.sep + '[tT]*_*.txt_*')
    txt_files = glob(code_dir + os.sep + '[tT]*_*.txt')
    csv_files = glob(code_dir + os.sep + '[tT]*_*.csv')
    pkl_files = glob(code_dir + os.sep + '[tT]*_*.pkl')
    png_files = glob(code_dir + os.sep + '[tT]*_*.png')
    json_files = glob(code_dir + os.sep + '[tT]*_*.json')
    h5_files =  glob(code_dir + os.sep + '[tT]*_*.h5')
    tmp_files = glob(code_dir + os.sep + 'tmp*_*log*')
    error_files = glob(code_dir + os.sep + 'eva_error.txt')
    for file in txt_ind_files + txt_files + csv_files + tmp_files + error_files + pkl_files  + png_files + json_files + h5_files:
        os.remove(file)

    plots_dirs = glob(code_dir + os.sep + '[tT]*_*_plots'); #print(plots_dirs)
    for plots_dir in plots_dirs:
        shutil.rmtree(plots_dir)
    '''
    for file in txt_files:
        os.remove(file)
    for file in csv_files:
        os.remove(file)
    for file in tmp_files:
        os.remove(file)
    for file in error_files:
        os.remove(file)
    for file in pkl_files:
        os.remove(file)
    for file in png_files:
        os.remove(file)
    for file in json_files:
        os.remove(file)
    for file in h5_files:
        os.remove(file)
    '''
