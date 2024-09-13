#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8

import sys, os
from icecream import ic
from smlp_py.ext import plot
from smlp_py.smlp_flows import SmlpFlows
import threading
import time
import psutil
import os

ic.configureOutput(prefix=f'Debug | ', includeContext=True)

start_time = time.time()

plot_instance = plot.plot_exp()

def main(argv):

    plot_instance.copy_from()
    plot_instance.save_to_txt(argv)
    smlpInst = SmlpFlows(argv)
    smlpInst.smlp_flow()

if __name__ == "__main__":

    main(sys.argv)

end_time = time.time()
total_time = end_time-start_time
plot_instance.witnesses()
plot_instance.save_to_txt(total_time)
