# This file is part of smlprover.
#
# Copyright 2022 Konstantin Korovin <konstantin.korovin@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# reads headed csv file with fraction entries and convers to floats

import sys
import os.path
import pandas as pd
import numpy as np

from fractions import Fraction
import csv

data_file=sys.argv[1]

def frac_to_float(fname):
    
    frac_pd = pd.read_csv(fname,dtype=str)
#    print(frac_pd.describe())
    float_pd_table = frac_pd.applymap(lambda frac: float(Fraction(frac)))
    print(float_pd_table.to_csv(index=False, sep=','))


frac_to_float(data_file)
