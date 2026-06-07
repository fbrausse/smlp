#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#

# coding: utf-8

import sys, os

def main(argv):
    smlpInst = SmlpFlows(argv)
    smlpInst.smlp_flow()

def main2():
    main(sys.argv)

if __name__ == '__main__':
    from smlp_py.smlp_flows import SmlpFlows
    main(sys.argv)
else:
    from .smlp_py.smlp_flows import SmlpFlows

