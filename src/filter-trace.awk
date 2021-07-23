#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

BEGIN { n=0; v=0 }
/./ { if (verb) print }
/^v,/ {
        if (!v) {
                printf("n,%s,status,thresh,elapsed", $1)
                for (i=2; i<=NF; i++)
                        printf(",%s", $i)
                printf(",objval\n")
                v = 1
        }
}
/^[aA],/ { n++ }
function rpy(x) {
	py " " x | getline flt
	return flt
}
/^[abAB],/ {
        printf("%d,%s,%s", n, $1, $2);
        for (i=3; i<=NF; i++) {
                split($i, a, "/")
                printf(",%.4g", a[2] ? a[2] + 0 ? a[1]/a[2] : rpy($i) : a[1])
        }
        printf("\n");
}
