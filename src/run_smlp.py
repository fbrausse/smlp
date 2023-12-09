#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8


import sys
from smlp_py.smlp_flows import SmlpFlows


def main(argv):
    smlpInst = SmlpFlows(argv)
    smlpInst.smlp_flow()

if __name__ == "__main__":
    main(sys.argv)
    
