#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

shared1.csv:
	$(MAKE) -C rank0 $@ && ln -s rank0/$@ $@
