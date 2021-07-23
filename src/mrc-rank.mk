#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

CHS   = ch0 ch1
BYTES = $(foreach c,$(CHS),$(addprefix $(c)/byte/,0 1 2 3 4 5 6 7))

.PHONY: all train search collect

all: collect

define delegate =
$(1)/%:
	$$(MAKE) -C $(1) $$*
endef
$(foreach d,$(CHS) $(BYTES),$(eval $(call delegate,$(d))))

train: $(addsuffix /train,$(BYTES))
search: $(addsuffix /lock.mk,$(BYTES))
collect: shared1.csv

shared1.csv: $(patsubst %,shared1.%.csv,$(CHS))
	{ head -n1 $< && tail -qn+2 $^; } > $@.tmp && mv $@.tmp $@

#ch%/shared1.csv:
#	$(MAKE) -C ch$* shared1.csv

shared1.ch%.csv: ch%/shared1.csv
	sed -r '1{s/^/CH,/;b};s/^/$*,/' $< > $@.tmp && mv $@.tmp $@
