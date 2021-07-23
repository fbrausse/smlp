#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

BASE_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(BASE_DIR)/scripts.mk

.PHONY: all train verify check collect safe.any clean-nn clean-vfy clean-check clean-collect clean-grid clean grid.istar

all: train verify check collect

clean: clean-train clean-vfy clean-safe clean-check clean-collect clean-grid

-include params.mk

############
# Training #
############

train: $(MODEL) $(CONFIG) $(BOUNDS) $(GEN)

.PRECIOUS: %_12_eval-Test.png %_12_eval-Train.png %_12_eval-Test-obj.png %_12_eval-Train-obj.png %_12_resp-distr.png %_12_train-reg.png

# we assume GNU make
model_complete_%_12.h5 model_config_%_12.json data_bounds_%_12.json model_gen_%_12.json \
%_12_eval-Test.png %_12_eval-Train.png %_12_resp-distr.png %_12_train-reg.png: %.csv $(SPEC)
	$(NICE) /usr/bin/time -v $(UNBUFFER) $(TRAIN) $(TRAIN_FLAGS) $(basename $<) |& \
	bash -c "tee >(sed -r 's/^([^\r]*\r)*//' > train$(NAME).log) $(DISP)" && \
	exit $${PIPESTATUS[0]}

clean-train:
	$(RM) $(MODEL) $(BOUNDS) $(CONFIG) $(GEN) train$(NAME).log $(PNGS)

###################
# Grid prediction #
###################

t_from_tany = $(shell printf "%s" '$1' | sed -r 's/^([0-9]+).*/\1/')

#grid.predict.csv.xz: $(MODEL) $(BOUNDS) $(GEN)
#	$(PREDICT_GRID) $(PREDICT_GRID_FLAGS) | xz -vT0 >$@.tmp && mv $@.tmp $@

.PRECIOUS: grid.T%.predict grid.T%.domain grid.T%.check

grid.T%.predict grid.T%.domain: $(MODEL) $(BOUNDS) $(GEN)
	$(PREDICT_GRID) $(PREDICT_GRID_FLAGS) -t 0.$(call t_from_tany,$*) \
		-p grid.T$*.predict -o grid.T$*.domain

grid.T%.check: grid.T%.domain
	($(CHECK_DATA) $(CHECK_DATA_FLAGS) -S $< -t 0.$(call t_from_tany,$*) \
		$(DATA) > $@.tmp || test $$? -eq 8) && mv $@.tmp $@

grid.T%.csv: grid.T%.domain grid.T%.predict
	$(CHECK_DATA) $(CHECK_DATA_FLAGS) -S $< -t 0.$(call t_from_tany,$*) \
		-v $(DATA) | paste -d, - $^ > $@

grid.T%.istar: grid.T%.check grid.T%.domain grid.T%.predict
	paste -d, $^ | grep -Ev '^0,0,' | grep -Ev '^[1-9][^,]*,' > $@

grid.istar: grid.T$(T)$(VNAME).istar

clean-grid:
	$(RM) grid.predict.csv.xz \
		$(wildcard grid.T*.domain grid.T*.predict grid.T*.check grid.T*.istar)

################
# Verification #
################

verify: train
	touch $(TRACE) && \
	$(NICE) $(UNBUFFER) sh -c '/usr/bin/time -v $(VERIFY) -vv $(VERIFY_FLAGS) 2>>$(VFY).log' | \
	tee -a $(TRACE) | $(UNBUFFER_P) $(FILTER_TRACE) -r | tr , '\t' $(DISP)

#$(SAFE).csv: $(SPEC) $(MODEL) $(BOUNDS) $(GEN)
#	$(NICE) $(UNBUFFER) sh -c '/usr/bin/time -v $(VERIFY) -vv $(VERIFY_FLAGS) 2>$(VFY).log' | \
#	tee $(TRACE) | $(UNBUFFER_P) $(FILTER_TRACE) -r | tr , '\t' $(DISP)

clean-vfy:
	$(RM) $(TRACE) $(VFY).test.smt2 $(VFY).log

clean-safe: clean-check
	$(RM) $(SAFE).csv

####################
# Search threshold #
####################

$(SEARCH).safe.csv: $(MODEL) $(CONFIG) $(BOUNDS) $(GEN)
	touch $(SEARCH).trace && \
	$(NICE) $(UNBUFFER) sh -c '/usr/bin/time -v $(VERIFY) -vv $(subst ','\'',$(SEARCH_FLAGS)) 2>>$(SEARCH).log' | \
	tee -a $(SEARCH).trace | $(UNBUFFER_P) $(FILTER_TRACE) -r | tr , '\t' $(DISP) && \
	exit $${PIPESTATUS[0]}

search: $(SEARCH).safe.any

clean-search:
	$(RM) $(SEARCH).{trace,log}

clean-search-safe:
	$(RM) $(SEARCH).safe.csv

#########################
# Checking against data #
#########################

#$(SAFE).check: $(SAFE).csv $(SPEC) $(DATA)
vfy%.safe.check: vfy%.safe.csv $(SPEC) $(DATA)
	($(CHECK_DATA) $(CHECK_DATA_FLAGS) -S $< -t 0.$(call t_from_tany,$*) $(DATA) || test $$? -eq 8) >$@.tmp && mv $@.tmp $@

%.safe.any: %.safe.csv
	if [ `wc -l < $<` -gt 1 ]; then touch $@; else exit 0; fi

.PHONY: safe.n

%.safe.n$(N): %.safe.csv
	if [ `wc -l < $<` -gt $(N) ]; then touch $@; else exit 1; fi

safe.n: $(SAFE).n$(N)

safe.any: $(wildcard vfy*.safe.csv)
	for f in $(patsubst %.csv,%.any,$^); do $(MAKE) "$$f" && break; done

check: $(SAFE).check $(SAFE).any
	@echo "in ball: `head -n1 $<`: `tail -n+2 $<`"

clean-check:
	$(RM) $(SAFE).check $(SAFE).any

######################
# Collecting results #
######################

#collect: $(subst .any,.results,$(wildcard vfy*.safe.any)) #$(SAFE).results
collect: $(SAFE).results

#%.safe.results: %.safe.csv %.safe.check category
vfy%.safe.results: vfy%.safe.csv vfy%.safe.check $(CATEGORIES)
	$(COLLECT) $^ 0.$(call t_from_tany,$*) > $@.tmp && mv $@.tmp $@

clean-collect:
	$(RM) $(SAFE).results

COLLECT_VFY = $(addsuffix $(VNAME),\
	vfy70ST75 \
	vfy75ST80 \
	vfy80ST85 \
	vfy85ST90 \
	vfy90ST95 \
)

collect.mk: $(foreach f,$(COLLECT_VFY),$(wildcard $(f).log $(f).trace)) $(BOUNDS) train$(NAME).log $(MODEL)
	echo 'FILES := $$(FILES) $$(addprefix $$(REL_PATH)/,$^)' > $@.tmp && mv $@.tmp $@


############################
# Various
############################

%.h5.dump: %.h5
	h5dump $< > $@.tmp && mv $@.tmp $@
%.h5.dump.a: %.h5
	h5dump -m %a $< > $@.tmp && mv $@.tmp $@
%.h5.dump.g: %.h5
	h5dump -m %g $< > $@.tmp && mv $@.tmp $@
