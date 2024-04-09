# About SMLP

SMLP is a tool for optimization, synthesis and design space exploration. It is
based on machine learning techniques combined with formal verification
approaches that allows selection of optimal configurations with respect to given
constraints on the inputs and outputs of the system under consideration.


# CAV submission

This is the artifact accompanying the tool paper 5673 for CAV 2024. It is
provided as a VirtualBox OVA instance, which is running a stock Ubuntu 22.04.
The software has been pre-installed together with its dependencies locally into
the "smlp" user's $HOME/.local directory. The source code is located in
$HOME/smlp and is based on the git branch "cav2024" of SMLP's repository located
at

	https://github.com/fbrausse/smlp/

The VM is configured to use 8 CPU cores and 16GB of main memory. The individual
regression tests provided as part of the SMLP distribution typically finish in a
few minutes. Exceptions are noted in the section on running the regression suite.
The CPU and memory requirements can be reduced in Virtualbox to better match the
host machine's resources.

## Quick instructions on running the smoke test

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


# VM Login

- Hostname: smlp-ubuntu
- User: smlp
- Pass: start

It is easiest to connect via ssh. For this the network settings of the virtual
machine should be set to "network bridge" with one of the host's network devices
that has access to a DHCP server.

After that,

	ssh smlp@smlp-ubuntu

should find the virtual host and login via the above credentials will be
possible.


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

The following section outlines how the regression script can be used to check
that SMLP is behaving as intended.

Documentation is provided in form of a manual and a description of the .spec
format as part of the artifact in SMLP_manual.pdf.

As this project has been growing since more than 4 years, the following parts
represent state that is irrelevant to the tool described for CAV 2024 and
should be ignored:

- include
- src/*.*
- utils/*.*


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

    ../../src/run_smlp.py -out_dir ./ -pref Test130 -data ../data/smlp_s2_tx \
    -mode optimize -pareto f -sat_thresh f -resp o0 -feat \
    Byte,CH,RANK,Timing,i0,i1,i2,i3 -model dt_sklearn -dt_sklearn_max_depth 15 \
    -data_scaler min_max -epsilon 0.05 -log_time f -plots f \
    -spec ../specs/smlp_s2_tx

    ../../src/run_smlp.py -out_dir ./ -pref Test131 -data ../data/smlp_s2_tx \
    -mode optimize -pareto f -sat_thresh f -resp o0 \
    -feat Byte,CH,RANK,Timing,i0,i1,i2,i3 \
    -model nn_keras -nn_keras_epochs 20 -data_scaler min_max \
    -epsilon 0.05 -log_time f -plots f  -spec ../specs/smlp_s2_tx 

These runs will take longer than the regression tests provided earlier,
usually between one and three hours, depending on the machine.

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
	git clone -b cav2024 https://github.com/fbrausse/smlp.git
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


# Badges

## Functional

We provide an extensive manual with SMLP in this artifact, found in
SMLP_manual.pdf on the top-level of the published .zip archive.

The paper describes the functionality offered by SMLP in form of operating
modes, all of which are supported by SMLP. This can be checked by running the
regression tests appropriate for each mode and examining their outputs,
see the corresponding section above.

The regression tests are performed against a known-good baseline located in
the regr_smlp/master directory.

## Reusable

SMLP is open-source licensed under the Apache-2.0 license.

The library dependencies of SMLP are common in machine learning applications.
For a list of dependencies, see the installation instructions for Ubuntu-22.04
above.

The manual distributed in this artifact contains extensive documentation of
the functionality implemented in SMLP and on their use.

SMLP has successfully been run without a container or VM on Ubuntu,
Suse Linux Enterprise Server 15, and Gentoo.
 