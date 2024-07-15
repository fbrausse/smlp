# About SMLP

SMLP is a tool for optimization, synthesis and design space exploration. It is
based on machine learning techniques combined with formal verification
approaches that allows selection of optimal configurations with respect to given
constraints on the inputs and outputs of the system under consideration.

When you use this tool, please cite our corresponding CAV 2024 tool paper,
a pre-submission version of which is provided on arXiv:
<https://arxiv.org/abs/2402.01415>


# Platform support

SMLP has successfully been run without a container or VM on Ubuntu,
Suse Linux Enterprise Server 15, and Gentoo. The following section provides
instruction for the installation on Ubuntu.


## Installation on a stock Ubuntu-22.04

	sudo apt install \
		python3-pip ninja-build z3 libz3-dev libboost-python-dev texlive \
		pkg-config libgmp-dev libpython3-all-dev python-is-python3
	# get a recent version of the meson configure tool
	pip install --user meson

	# obtain sources
	git clone https://github.com/fbrausse/kay.git
	git clone -b shai4-c https://github.com/fbrausse/smlp.git
	cd smlp/utils/poly

	# workaround <https://bugs.launchpad.net/ubuntu/+source/swig/+bug/1746755>
	echo 'export PYTHONPATH=$HOME/.local/lib/python3/dist-packages:$PYTHONPATH' >> ~/.profile
	# get $HOME/.local/bin into PATH and get PYTHONPATH
	mkdir -p $HOME/.local/bin
	source ~/.profile

	# setup, build & install libsmlp
	meson setup -Dkay-prefix=$HOME/kay --prefix $HOME/.local build
	ninja -C build install

	# tensorflow-2.16 has a change leading to the error:
	# 'The filepath provided must end in .keras (Keras model format).'
	pip install --user \
		pandas tensorflow==2.15.1 scikit-learn pycaret seaborn \
		mrmr-selection jenkspy pysubgroup pyDOE doepy


## Quick instructions on testing whether the tool works

    cd $HOME/smlp/regr_smlp/code

    # first the tool itself
    ../../src/run_smlp.py -data "../data/smlp_toy_num_resp_mult" \
    -out_dir ./ -pref Test83 -mode optimize -pareto t -sat_thresh f \
    -resp y1,y2 -feat x,p1,p2 -model dt_sklearn -dt_sklearn_max_depth 15 \
    -spec smlp_toy_num_resp_mult_free_inps -data_scaler min_max \
    -beta "y1>7 and y2>6" -objv_names obj1,objv2,objv3 \
    -objv_exprs "(y1+y2)/2;y1/2-y2;y2" -epsilon 0.05 -delta_rel 0.01 \
    -save_model_config f -mrmr_pred 0 -plots f -seed 10 -log_time f \
    -spec ../specs/smlp_toy_num_resp_mult_free_inps

    # then the regression script
    ./smlp_regr.py -w 1 -def n -t 88 -tol 5


# Running the regression suite

The regression script has to be run from inside the regression's code directory:

	cd $HOME/smlp/regr_smlp/code
	./smlp_regr.py -w 8 -def n -t all -tol 7

The above commands will execute the script, run the regression tests numbered
1 to 129 (-t all) parallely on 8 cores (-w 8), not overwriting the stored
reference output (-def n) with a tolerance of 7 decimal fractional digits
(-tol 10). Note, it needs to be run from the given directory and will place the
resulting files into that directory as well. It takes about 10 minutes on a
2.9 GHz laptop with 8 cores utilized.

The output of the command will report on the results of comparing the stored
known-good results (called 'master' in this directory tree) with those created
by running the test locally. It should form a sequence of

	comparing $FILE to master
	Passed!

In case there differences between the current run and the stored files, a diff
between the generated files will be printed instead.

The actual SMLP commands being run by the script can be obtained by appending
the parameter -p, e.g.

	./smlp_regr.py -def n -t 1 -p

will produce the SMLP command for the regression test number 1:

	../../src/run_smlp.py -data "../data/smlp_toy_num_resp_mult" \
	-out_dir ./ -pref Test1 -mode train -resp y1 -feat x,p1,p2 \
	-model dt_caret -save_model_config f -mrmr_pred 0 -plots f \
	-seed 10 -log_time f

For details about those parameters, please refer to the help messages (-h) of
both tools, src/run_smlp.py and regr_smlp/code/smlp_regr.py, as well as the
manual.

SMLP commands run in the regression can be found in ./smlp_regr.csv together
with a short description of the respective test.

## Regression tests for SMLP operating modes

The main claims of the paper are about functionality of SMLP which is
reflected in different operating modes.

The smlp_regr.py script supports parameter -m to select the subset of the
regression tests matching the given operating mode. Supported are the
following of SMLP's modes:

- certify
- query
- verify
- synthesize
- optimize
- optsyn (optimized synthesis)
- doe

Detailed outputs of each tests will be generated in the regr_smlp/code directory.
We refer to the manual for further information. 

## A note on external solvers

Some regression tests for performance reasons use external solvers, like
     MathSAT, instead of the default Z3. The list of those tests can be obtained via

    grep -- -solver_path ./smlp_regr.csv

Unfortunately, due to licensing restrictions, it is impossible for us to
include a copy of this particular external solver. However, reviewers are
free to obtain a copy and put the executable into the path expected by those
regression tests:

    $HOME/smlp/regr_smlp/code/../../../external/mathsat-5.6.8-linux-x86_64-reentrant/bin/mathsat

(or to modify the path given in the above .csv file).

It is also possible to use different SMT solvers, for details please see
<https://github.com/fbrausse/smlp/tree/master/utils/poly#external-solvers>.

These tests are thus expected to not finish successfully.

## Tests on a real-life data set

In the regr_smlp directory, we also provide two tests on a real-life data set
as have been used at Intel. To run them from inside the regr_smlp/code
directory, run the following commands:

    ../../src/run_smlp.py -out_dir ./ -pref smlp_s2_tx_dt -data ../data/smlp_s2_tx \
    -mode optimize -pareto f -sat_thresh f -resp o0 -feat \
    Byte,CH,RANK,Timing,i0,i1,i2,i3 -model dt_sklearn -dt_sklearn_max_depth 15 \
    -data_scaler min_max -epsilon 0.05 -log_time f -plots f \
    -spec ../specs/smlp_s2_tx

    ../../src/run_smlp.py -out_dir ./ -pref smlp_s2_tx_nn -data ../data/smlp_s2_tx \
    -mode optimize -pareto f -sat_thresh f -resp o0 \
    -feat Byte,CH,RANK,Timing,i0,i1,i2,i3 \
    -model nn_keras -nn_keras_epochs 20 -data_scaler min_max \
    -epsilon 0.05 -log_time f -plots f  -spec ../specs/smlp_s2_tx 

These runs will take longer than the regression tests provided earlier,
usually between one and three hours, depending on the machine.


# Source code organization

SMLP at the moment consists of parts written in Python and in C++ and tiny bits
of C. The C++ part is compiled into a shared library 'libsmlp', which exports a
Python module with the same name. Additionally, a small wrapper script around
it is provided in

- utils/poly/python/smlp

which contains the proper Python interface consisting of ways to run solvers,
to construct terms and formulas, and to deal with variable domains.

The source code for this library is located in

- utils/poly/src

and the main files in there include

- algebraics.*: support for algebraic real numbers as solutions to polynomial equations
- domain.*: definition of the search space (the "domain")
- expr2.*: definition of internal representation of terms and formulas
- libsmlp.cc: Python interface library 'libsmlp'
- reals.hh: support for computable real numbers, a superset of the algebraics
- solver.*: definition of SMT solver interface

This however is just the backend dealing with the process of solving concrete
formulas once the semantics has been put in place.

The core of the project is written in Python, and is located in

- src/smlp_py/

(and subdirectories). It defines the actual algorithms used in SMLP as described
in the tool paper, and contains code supporting all modes and configurations
laid out therein. In alphabetical order, these are the main files:

- smlp_doe.py: design of experiments
- smlp_optimize.py: optimization and optimized synthesis
- smlp_query.py: synthesis, certification, verification and query
- smlp_subgroups.py: the subgroup discovery algorithm implemented in pysubgroup package

The main entry point is the script

- src/run_smlp.py

It supports the various modes documented in the CAV submission as well as in the
manual.

SMLP comes with a set of regression tests located in the directory

- regr_smlp

It contains definitions of models, specifications, data sets used for training
and data constraints for the solving process. A script to run
the regression tests is provided in

- regr_smlp/code/smlp_regr.py

which basically just builds command lines, runs the tool and compares the
outputs to the expected results contained in

- regr_smlp/master

Documentation is provided in form of a manual and a description of the .spec
format as part of the artifact in SMLP_manual.pdf.
