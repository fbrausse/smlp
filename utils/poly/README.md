# smlp

smlp is a program to solve and optimize polynomials and NNs over integers and reals.

## Usage

```
usage: smlp [-OPTS] [--] { DOMAIN EXPR | H5-NN SPEC GEN IO-BOUNDS } OP [CNST]

Options [defaults]:
  -1           use single objective from GEN instead of all H5-NN outputs [no]
  -a ALPHA     additional ALPHA constraints restricting candidates *and*
               counter-examples (only points in regions satsifying ALPHA
               are considered counter-examples to safety) [true]
  -b BETA      additional BETA constraints restricting only candidates
               (all points in safe regions satisfy BETA) [true]
  -c           clamp inputs (only meaningful for NNs) [no]
  -C COMPAT    use a compatibility layer, can be given multiple times; supported
               values for COMPAT:
               - python: reinterpret floating point constants as python would
                         print them
  -d DELTA     increase radius around counter-examples by factor (1+DELTA) [0]
  -F IFORMAT   determines the format of the EXPR file; can be one of: 'infix',
               'prefix' [infix]
  -h           displays this help message
  -n           dry run, do not solve the problem [no]
  -O OUT-BNDS  scale output according to min-max output bounds (.csv, only
               meaningful for NNs) [none]
  -p           dump the expression in Polish notation to stdout [no]
  -P PREC      maximum precision to obtain the optimization result for [0.05]
  -Q QUERY     answer a query about the problem; supported QUERY:
               - vars: list all variables
  -r           re-cast bounded integer variables as reals with equality
               constraints
  -R LO,HI     optimize threshold in the interval [LO,HI] [0,1]
  -s           dump the problem in SMT-LIB2 format to stdout [no]
  -t TIMEOUT   set the solver timeout in seconds, 0 to disable [0]

The DOMAIN is a text file containing the bounds for all variables in the
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.
Empty lines are skipped.

The EXPR file contains a polynomial expression in the variables specified by the
DOMAIN-FILE. The format is either an infix notation or the prefix notation also
known as Polish notation. The expected format can be specified through the -F
switch.

The problem to be solved is specified by the two parameters OP CNST where OP is
one of '<=', '<', '>=', '>', '==' and '!='. Remember quoting the OP on the shell
to avoid unwanted redirections. CNST is a rational constant in the same format
as those in the EXPR-FILE (if any).

Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.
License: Apache 2.0; part of SMLP.
```

Assuming files `domain` and `expression` are in the current directory, this would
be a sample to execute poly:

	$ build/src/smlp domain expression '>=' 0.42

## Build instructions

To build the program, meson <https://mesonbuild.com> is required.
Run:

	$ mkdir build && meson setup build && meson compile -C build

The resulting binary will be located in `build/src/smlp`.

A compiler supporting C++20 is required as well as:

- the kay library <https://github.com/fbrausse/kay>

  Use the parameter `-Dkay-prefix=PATH` for the `meson setup` command to
  specify the location of the kay library.
- GMP <https://gmplib.org> and either
  - its C++ bindings called `gmpxx`, or
  - Flint <http://flintlib.org>

    Use `-Dflint-prefix=PATH` for the `meson setup` command to specify a
    non-standard location of the flint library.

  Use one of `-Dflint=(enabled|disabled|auto)` to prefer Flint over `gmpxx`.
  The default is `auto`.
- Z3 and its C++ bindings <https://github.com/Z3Prover/z3>

### NN support

The usage output shown above is for a build that includes NN-support. This
requires more dependencies:

- kjson <https://github.com/fbrausse/kjson>
- hdf5 and its C++ bindings <https://www.hdfgroup.org/HDF5>
- sources of libiv (*not published, yet*)

## Legal info

All source files in this distribution are:
```
Copyright 2022 Franz Brauße <franz.brausse@manchester.ac.uk>
Copyright 2022 The University of Manchester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
See the LICENSE file for more details.
