#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

TRAIN := env -u DISPLAY $(TRAIN)

CH_BOUNDS := $(me)/rank$(RANK)/ch$(CH)/bounds.csv
VERIFY_FLAGS += -r $(CH_BOUNDS)
SEARCH_FLAGS += -r $(CH_BOUNDS)
CHECK_DATA_FLAGS += -B $(CH_BOUNDS)
PREDICT_GRID_FLAGS += -B $(CH_BOUNDS)

N = 100

CATEGORIES = categories

categories: category
	echo '{ "RANK": $(RANK), "CH": $(CH) }' | paste - $^ > $@.tmp && mv $@.tmp $@

include $(wildcard ../*/lock.mk)


srch: SAFE = $(VFY).srch
srch: VERIFY_FLAGS = $(VERIFY_FLAGS-$(VNAME)) \
        -d $(DATA) \
        -g $(GEN) \
        -b -B $(BOUNDS) \
        -C 0 \
        -n $(N) \
        $(if $(OBJT),-O '$(OBJT)') \
        -s $(SPEC) \
        -S $(SAFE).csv \
        $(MODEL) \
	-r $(CH_BOUNDS)
srch: verify

vfy%.srch.ren: vfy%.srch.csv
	c=`echo $< | sed -r 's/vfy.*max(.)\.srch.csv/th-b$(Byte)-b\1/'` && \
	sed -r "1s/,thresh/,$$c/;s/(.*),([^,]*)/\\2;\\1/" $< > $@.tmp && \
	mv $@.tmp $@

vfy%max$(Byte).srch.ren: vfy%.safe.csv
	sed -r '1{s/^/th-b$(Byte)-b$(Byte);/;b};s/^/0.$(call t_from_tany,$*);/' $< > $@.tmp && \
	mv $@.tmp $@

.PHONY: srch-ren

srch-ren: $(patsubst vfy%.srch.csv,vfy%.srch.ren,$(wildcard vfy*.srch.csv))
srch-ren: $(patsubst ../$(Byte)/vfy%.safe.csv,vfy%max$(Byte).srch.ren,$(lastword $(VERIFY_FLAGS-max$(Byte))))

searchmax%.safe.csv: ../%/lock.mk
	$(MAKE) VNAME=max$* search

#searchmax$(Byte).srch.ren: search.safe.csv lock.mk
#	awk -F, '{printf("%s;",$$NF);for(i=1; i<NF; i++)printf("%s%s", i>1 ? "," : "", $$i);print ""}' $< | \
#	sed '1s/^thresh;/th-b$(Byte)-b$(Byte);/' > $@.tmp && mv $@.tmp $@

searchmax%.srch.ren: searchmax%.safe.csv ../%/lock.mk
	awk -F, '{printf("%s;",$$NF);for(i=1; i<NF; i++)printf("%s%s", i>1 ? "," : "", $$i);print ""}' $< | \
	sed '1s/^thresh;/th-b$(Byte)-b$*;/' > $@.tmp && mv $@.tmp $@

../%/lock.mk:
	$(MAKE) -C $(dir $@) lock.mk

lock.mk: search.safe.csv
	echo "SEARCH_FLAGS-max$(Byte) = -NG ../$(Byte)/$<" > $@.tmp && mv $@.tmp $@

.PRECIOUS: lock.mk ../%/lock.mk searchmax%.safe.csv
