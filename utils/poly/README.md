# poly

poly is a program to solve polynomial inequalities over the reals.

## Usage

```
usage: ./poly [-OPTS] [--] DOMAIN-FILE EXPR-FILE OP CNST

Options [defaults]:
  -C COMPAT   use a compatibility layer, can be given multiple times; supported
              values for COMPAT:
              - python: reinterpret floating point constants as python would
                        print them
  -F IFORMAT  determines the format of the EXPR-FILE; can be one of: 'infix',
              'prefix' [infix]
  -h          displays this help message
  -n          dry run, do not solve the problem [no]
  -p          dump the expression in Polish notation to stdout [no]
  -s          dump the problem in SMT-LIB2 format to stdout [no]
  -t TIMEOUT  set the solver timeout in seconds, 0 to disable [0]

The DOMAIN-FILE is a text file containing the bounds for all variables in the
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.
Empty lines are skipped.

The EXPR-FILE contains a polynomial expression in the variables specified by the
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

## Build instructions

To build the program, meson <https://mesonbuild.com> is required.
Run:

	$ mkdir build && meson setup build && meson compile -C build

The resulting binary will be located in build/src/poly.

A compiler supporting C++20 is required as well as:

- the kay library <https://github.com/fbrausse/kay>
- GMP <https://gmplib.org> and either
  - its C++ bindings, or
  - Flint <http://flintlib.org>
- Z3 and its C++ bindings <https://github.com/Z3Prover/z3>

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
