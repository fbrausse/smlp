# smlp

smlp is a program to solve and optimize polynomials and NNs over integers and reals.

## Usage

```
usage: smlp [-OPTS] [--] { DOMAIN EXPR | H5-NN SPEC GEN IO-BOUNDS } OP [CNST]

Options [defaults]:
  -a ALPHA     additional ALPHA constraints restricting candidates *and*
               counter-examples (only points in regions satisfying ALPHA
               are considered counter-examples to safety); can be given multiple
               times, the conjunction of all is used [true]
  -b BETA      additional BETA constraints restricting candidates and safe
               regions (all points in safe regions satisfy BETA); can be given
               multiple times, the conjunction of all is used [true]
  -c COLOR     control colored output: COLOR can be one of: on, off, auto [auto]
  -C COMPAT    use a compatibility layer, can be given multiple times; supported
               values for COMPAT:
               - python: reinterpret floating point constants as python would
                         print them
               - bnds-dom: the IO-BOUNDS are domain constraints, not just ALPHA
               - clamp: clamp inputs (only meaningful for NNs) [no]
               - gen-obj: use single objective from GEN instead of all H5-NN
                          outputs [no]
  -d DELTA     increase radius around counter-examples by factor (1+DELTA) or by
               the constant DELTA if the radius is zero [0]
  -e ETA       additional ETA constraints restricting only candidates, can be
               given multiple times, the conjunction of all is used [true]
  -F IFORMAT   determines the format of the EXPR file; can be one of: 'infix',
               'prefix' (only EXPR) [infix]
  -h           displays this help message
  -i SUBDIVS   use interval evaluation with SUBDIVS subdivisions and fall back
               to the critical points solver before solving symbolically [no]
  -I EXT-INC   optional external incremental SMT solver [value for -S]
  -n           dry run, do not solve the problem [no]
  -o OBJ-SPEC  specify objective explicitely (only meaningful for NNs), an
               expression using the labels from SPEC or 'Pareto(E1,E2,...)'
               where E1,E2,... are such expressions [EXPR]
  -O OBJ-BNDS  scale objective(s) according to min-max output bounds (only
               meaningful for NNs, either .csv or .json) [none]
  -p           dump the expression in Polish notation to stdout (only EXPR) [no]
  -P PREC      maximum precision to obtain the optimization result for [0.05]
  -Q QUERY     answer a query about the problem; supported QUERY:
               - vars: list all variables
               - out : list all defined outputs
  -r           re-cast bounded integer variables as reals with equality
               constraints (requires -C bnds-dom); cvc5 >= 1.0.1 requires this
               option when integer variables are present
  -R LO,HI     optimize threshold in the interval [LO,HI] [interval-evaluation
               of the LHS]
  -s           dump the problem in SMT-LIB2 format to stdout [no]
  -S EXT-CMD   invoke external SMT solver instead of the built-in one via
               'SHELL -c EXT-CMD' where SHELL is taken from the environment or
               'sh' if that variable is not set []
  -t TIMEOUT   set the solver timeout in seconds, 0 to disable [0]
  -T THRESHS   instead of on an interval perform binary search among the
               thresholds in the list given in THRESHS; overrides -R and -P;
               THRESHS is either a triple LO:INC:HI of rationals with INC > 0 or
               a comma-separated list of rationals
  -v[LOGLVL]   increases the verbosity of all modules or sets it as specified in
               LOGLVL: comma-separated list of entries of the form [MODULE=]LVL
               where LVL is one of none, error, warn, info, note, debug [note];
               see below for values of the optional MODULE to restrict the level
               to; if LOGLVL is given there must not be space between it and -v
  -V           display version information

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
as those in the EXPR file (if any).

For log detail setting -v, MODULE can be one of:
  cand, coex, crit, ext, ival, nn, poly, prob, smlp, z3

Options are first read from the environment variable SMLP_OPTS, if set.

Exit codes are as follows:
  0: normal operation
  1: invalid user input
  2: unexpected SMT solver output (e.g., 'unknown' on interruption)
  3: unhandled SMT solver result (e.g., non-rational assignments)
  4: partial function applicable outside of its domain (e.g., 'Match(expr, .)')

Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.
License: Apache 2.0; part of SMLP.
```

Assuming files `domain` and `expression` are in the current directory, this would
be a sample to execute smlp:

	$ build/src/smlp domain expression '>=' 0.42

### External solvers

The following SMT solvers are known to work with smlp:
| Solver | Version(s) | param `-S` | param `-I` |
|--------|------------|------------|------------|
| [Z3](https://github.com/Z3Prover/z3) | 4.8.12, 4.11.2 | `z3 -in` | not required |
| [ksmt](http://informatik.uni-trier.de/~brausse/ksmt) | 0.1.8 | `ksmt` | not required |
| [Yices](http://yices.csl.sri.com) | 2.6.1 | `yices-smt2` | `yices-smt2 --incremental` |
| [CVC4](https://cvc4.github.io) | 1.8 | `cvc4 -L smt2` | `cvc4 -L smt2 --incremental` |
| [CVC5](https://cvc5.github.io) | 1.0.1 | `cvc5` | `cvc5 --incremental` |
| [MathSAT](https://mathsat.fbk.eu) | 5.6.3, 5.6.8 | `mathsat` | not required |

## Build instructions

To build the program, meson <https://mesonbuild.com> is required.
Run:

	$ mkdir build && meson setup build && meson compile -C build

The resulting binary will be located in `build/src/smlp`.

A compiler supporting C++20 is required (GCC >= 11, clang >= 11) as well as:

- The kay library <https://github.com/fbrausse/kay>.

  Use the parameter `-Dkay-prefix=PATH` for the `meson setup` command to
  specify the location of the kay library.

- GMP <https://gmplib.org> and its C++ bindings called `gmpxx`.

- Optionally Flint <http://flintlib.org> to speed up integer and rational
  arithmetic.

  Use `-Dflint-prefix=PATH` for the `meson setup` command to specify a
  non-standard location of the flint library.

  Use one of `-Dflint=(enabled|disabled|auto)` to prefer Flint.
  The default is `auto`: Use flint if found, otherwise fall back to `gmpxx`.

### Internal solver support

smlp can make use of a built-in solver. So far, API usage of the following SMT
solvers is implemented:

- Z3, requires its C++ bindings <https://github.com/Z3Prover/z3>

  Use one of `-Dz3=(enabled|disabled|auto)` to control the build behaviour.
  Default is `auto`: link against Z3 if found, otherwise disable this feature.

### NN support

The usage output shown above is for a build that includes NN-support. This
requires more dependencies:

- kjson <https://github.com/fbrausse/kjson>
- hdf5 and its C++ bindings <https://www.hdfgroup.org/HDF5>

### Python API

SMLP offers a Python API that allows to easily compose problems and solve them,
e.g., the same example as above could be written in Python as
```
import smlp
pp = smlp.parse_poly('domain', 'expression')
slv = smlp.solver(False)
slv.declare(pp.dom)
slv.add(pp.obj > smlp.Cnst(0.42))
r = slv.check()
if isinstance(r, smlp.sat):
     print('sat: ' + str(r.model))
elif isinstance(r, smlp.unsat):
     print('unsat')
else:
     print('unknown: ' + r.reason)
```

The Python module `smlp` will be built by the above `meson compile` command if
the following dependencies are satisfied:

- CPython development libraries and headers
- Boost.Python <https://www.boost.org/doc/libs/1_79_0/libs/python/doc/html/index.html>

Using `meson install -C build`, the `smlp` Python module will be installed into
the Python module directory under PREFIX, which can be specified at `meson setup`
time with the option `-Dprefix=PATH` and defaults to the standard system
prefix. On Unix this is usually `/usr/local`.

## SMLP Traces

As part of the optimization procedure, SMLP produces on `stdout` a so-called
trace. A trace is a sequence of records, one per line, that can be used to
reconstruct the steps taken by SMLP and to reconstruct its intermediate
results.

Each record consists of comma-separated fields. The first field denotes the
type of the record and it identifies the number and format of the remaining
fields. The following record types exist:

| Type | Format | Correspondence to SMLP |
|------|--------|------------------------|
| d | `d,p` | `p` is the working directory where `smlp` has been executed; issued once at the beginning |
| c | `c,n,a` | command line `smlp` was invoked with; `n` is the number of arguments and `a` is the `\0`-delimited string resulting from concatenating the `n` arguments; issued once at the beginning |
| r | `r,l,h,T` | search range, `l` is the lower bound, `h` is the upper bound and `T` corresponds to the current threshold |
| u | `u,l,h,c` | search space exhausted, `l` and `h` are as for the `r` type and `c` specifies whether the threshold on the safe regions is inside the initial objective range, `c` can take values `in`, `out` and `maybe` |
| a | `a,r,T,s,as...` | candidate search result; `r` is either `sat`, `unsat` or `unknown`; `T` is the threshold this result holds for, `s` is the time in seconds needed to solve this problem and in case `r` is `sat`, `as...` contains the satisfying assignment as a sequence of comma-separated pairs `var,value` |
| b | `b,r,T,s,as...` | counter-example search result; same fields as `a` records; it corresponds to the preceding found candidate for threshold `T` |

Note that these traces are designed to be easily processed with Unix tools,
though with Python >= 3.11 they also are, e.g.
```
>>> import csv
>>> r = csv.reader(open('test.trace'))
>>> list(r)
```
Python < 3.11 will [not process](https://bugs.python.org/issue27580) the `c`
record due to embedded `'\0'` characters and Pandas will by default refuse
traces since the number of "columns" is not constant.

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
