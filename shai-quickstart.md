
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

### Preprocessing, step 1

Shai's raw data contains a few columns that are not required for the task,
as well as `Up` and `Down`, the difference of which makes the task's objective
called `delta`.

The required preprocessing is performed by the script
`$SMLP/repo/src/shai-preproc.py`, which is invoked for `rx` and `tx` separately
like so:

```
(venv) $ $SMLP/repo/src/shai-preproc.py -i RX-RAW.csv -o RX-PP.csv -s RX-SPEC.spec ADLS_new rx
(venv) $ $SMLP/repo/src/shai-preproc.py -i TX-RAW.csv -o TX-PP.csv -s TX-SPEC.spec ADLS_new tx
```

Here, `*-RAW.csv` denote the input datasets, `*-PP.csv` are paths the
preprocessed outputs are written to and the parameter to the (optional) `-s`
argument `*-SPEC.spec` is the path the corresponding pre-generated .spec file
is written to. It contains the description of the domain and the problem
specification for SMLP. `ADLS_new` finally denotes the product the given dataset
refers to and has so far been described by Shai directly. To add new products,
just add another such description to `$SMLP/repo/src/shai-preproc.py` akin the
existing one in the definition of `spec['ADLS_new']`.

### Preprocessing, step 2


