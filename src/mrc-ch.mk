#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

ENV := /usr/bin/env
SHELL := $(ENV) LC_ALL=C LC_NUMERIC=C bash
BYTES = 0 1 2 3

shared1.csv: $(patsubst %,shared1.m%.csv,$(BYTES))
	{ head -n1 $< && tail -qn+2 $^; } > $@.tmp && mv $@.tmp $@

shared1.m%.csv: shared1.b%.csv
	sed -r '1{s/(th-b.)-b$*,/\1,/g;s/^/Byte,/;b};s/^/$*,/' $< > $@.tmp && mv $@.tmp $@

define shared1b =
byte/$(1)/%:
	$$(MAKE) -C $$(dir $$@) $$*

$$(patsubst %,byte/%/searchmax$(1).srch.ren,$(BYTES)): byte/$(1)/lock.mk

shared1.b$(1).csv: $$(addsuffix /searchmax$(1).srch.ren,$(addprefix byte/,$(BYTES)))
	all() { while [ $$$$# -gt 0 ]; do test 0 -eq $$$$1 || exit $$$$?; shift; done; }; \
	cat byte/0/searchmax$(1).srch.ren | sort -t\; -k2 -g | \
	join --nocheck-order -t\; -j2 -o '1.1 2.1 0' - <(sort -t\; -k2 -g < byte/1/searchmax$(1).srch.ren) | sed 's/;/,/' | \
	join --nocheck-order -t\; -j2 -o '1.1 2.1 0' - <(sort -t\; -k2 -g < byte/2/searchmax$(1).srch.ren) | sed 's/;/,/' | \
	join --nocheck-order -t\; -j2 -o '1.1 2.1 0' - <(sort -t\; -k2 -g < byte/3/searchmax$(1).srch.ren) | sed 's/;/,/g' > $$@.tmp && \
	all $$$${PIPESTATUS[@]} && mv $$@.tmp $$@
endef

$(foreach b,$(BYTES),$(eval $(call shared1b,$(b))))
