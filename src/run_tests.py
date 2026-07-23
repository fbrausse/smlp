#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run the tests for SMLP project

# coding: utf-8

from smlp_py.ra_tests import run_ra_tests

def main():
    run_ra_tests()

if __name__ == "__main__":
    main()