#!/usr/bin/env bash
export DRYRUN=1
for t in $@; do
    pytest -s test_cmdline.py::"Test${t}" |& egrep "^smlp|DRYRUN: smlp"
    if [[ $t != "${!#}" ]]; then
        printf '=%.0s' {1..80}
        echo ""
    fi
done
