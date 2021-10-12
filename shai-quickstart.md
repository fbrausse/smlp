
# SMLP details and usage

Prepare the shell:
```
$ bash
$ export SMLP=/nfs/iil/proj/dt/eva/smlp
$ source $SMLP/root/venv/bin/activate
(venv) $
```

This sets the `SMLP` envvar to SMLP's root path and loads the venv.
`$SMLP/repo` contains the distributed SMLP code and
`$SMLP/root/venv` contains the Python virtualenv preset to CPython-3.7.


## Shai

As modelling Shai's setting changed a few times, some preprocessing is required
to solve his problem. It is performed the steps detailed below, the second of
which either results in a single- or multi-objective optimization problem.


### Preprocessing, step 1

Shai's raw data contains a few columns that are not required for the task,
as well as `Up` and `Down`, the difference of which makes the task's objective
called `delta`.

The required preprocessing is performed by the script
`$SMLP/repo/src/shai-preproc.py`, which is invoked for `rx` and `tx` separately
like so:

```
(venv) $ $SMLP/repo/src/shai-preproc.py -i RX-RAW.csv -o RX-PP.csv -s RX-PP.spec ADLS_new rx
(venv) $ $SMLP/repo/src/shai-preproc.py -i TX-RAW.csv -o TX-PP.csv -s TX-PP.spec ADLS_new tx
```

Here, `*-RAW.csv` denote the input datasets, `*-PP.csv` are paths the
preprocessed outputs are written to and the argument `*-SPEC.spec` to the
(optional) parameter `-s` is the path the corresponding pre-generated .spec
file is written to. It contains the description of the domain and the problem
specification for SMLP. `ADLS_new` finally denotes the product the given
dataset refers to and has so far been described by Shai directly. To add new
products, just add another such description to `$SMLP/repo/src/smlp/mrc/defs.py`
akin the existing one in the definitions of `spec['ADLS_new']` and
`joint['ADLS_new']`. If 'special' columns (or their labels) changed, it might
be necessary to also adjust `shai_data_desc` in the same file.


### Preprocessing, step 2 (single objective version)

In the single objective version, the 2 new features
- `trad` denoting the Time window RADius
- `area` denoting the non-linear objective function depending on `min_delta` and
  `trad`
are added to the existing datasets for RX and TX, respectively.
The intermediate `min_delta` is computed by taking the minimum `delta` in the
time window around each sample. The 7 default values for `trad` are
defined as `DEF_TIME_WINDOW_RADII` in `$SMLP/repo/src/smlp/mrc/preparea.py`.

This 2nd preprocessing step is thus executed like so:
```
(venv) $ env PYTHONPATH=$SMLP/repo/src:$PYTHONPATH python3 -m smlp.mrc.shai -o pp1 RX-PP.csv RX-PP.spec TX-PP.csv TX-PP.spec
```

This will create the directory `pp1` and the files `pp1/rx.spec`, `pp1/rx.csv`,
`pp1/tx.spec` and `pp1/tx.csv`.

If "joint parameters" are part of the problem, these can be passed to the above
command via the `-j` parameter. See the help output (when invoked with `-h`) for
details. In this case, another file `pp1/joint` will be created containing the
information about all joint features.


### Preprocessing, step 2 (multi objective version)

In the multi objective version, the 2 new features `eye_w` and `eye_h` denoting
eye width and height, respectively, are added to the existing datasets.
`eye_h` corresponds to `min_delta` from the single objective preprocessing
version and `eye_w` to `trad`.

As a second step, these 2 objectives are combined across all `Byte` values
resulting in features labelled `eye_w_0`, `eye_w_1`, `eye_w_2`, etc. and
correspondingly for `eye_h`.

The command to generate these is:
```
(venv) $ env PYTHONPATH=$SMLP/repo/src:$PYTHONPATH python3 -m smlp.mrc.shai -2 -o pp2 RX-PP.csv RX-PP.spec TX-PP.csv TX-PP.spec
```

As before, this will create the directory `pp2` and in it the files
`pp2/rx.csv`, `pp2/rx.spec`, `pp2/tx.csv` and `pp2/tx.spec`. Again, if joint
parameters are specified via `-j`, the `pp2/joint` file will be created.


### Running SMLP for single objective version

Assuming `pp1` was created for the single-objective problem according to the
instructions above, the following commands will, in order, prepare the directory
structure, train neural networks, solve the optimization problem and collect
the results from the multiple `Byte`s for each `MC`.

First, we create a temporary working directory on the local machine and store
its path in the variable WD for easier reference:
```
(venv) $ # create temporary working directory on the local machine
(venv) $ export WD=/localdisk/smlp-work-dir
(venv) $ mkdir -p $WD
```

Next is to execute `smlp-mrc.sh` on RX and TX separately like this:

```
(venv) $ $SMLP/repo/smlp-mrc.sh -s pp1/rx.spec -i pp1/rx.csv -t $WD/rx -r area run collect
(venv) $ $SMLP/repo/smlp-mrc.sh -s pp1/tx.spec -i pp1/tx.csv -t $WD/tx -r area run collect
```

`smlp-mrc.sh` internally executes the stages `prepare`, `train`, `search` before
the last one called `collect`, as necessary. Thus, for example for RX, the above
command is equivalent to manually running the following steps, which are listed
here in case one of the intermediate stages encounters a problem:
```
(venv) $ # prepare directory structure
(venv) $ $SMLP/repo/smlp-mrc.sh -s pp1/rx.spec -i pp1/rx.csv -t $WD/rx -r area run prepare

(venv) $ # train the networks for RANK=0, and each MC and Byte combination
(venv) $ $SMLP/repo/smlp-mrc.sh -r area -t $WD/rx run train

(venv) $ # run the searches for a threshold on 'area'
(venv) $ $SMLP/repo/smlp-mrc.sh -r area -t $WD/rx run search

(venv) $ # collect the results
(venv) $ $SMLP/repo/smlp-mrc.sh -r area -t $WD/rx run collect
```

The result of the `collect` stage is the file `$WD/rx/rank0/shared1.csv` for RX
and `$WD/tx/rank0/shared1.csv` for TX.

Again, for all supported options (like `-r` used in the above commands to denote
the response feature to optimize, see also preprocessing step 1), run
`smlp-mrc.sh -h`.
