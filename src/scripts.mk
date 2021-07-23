#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

ENV := /usr/bin/env
SHELL := $(ENV) LC_ALL=C LC_NUMERIC=C bash
me = $(dir $(lastword $(MAKEFILE_LIST)))

DATA = data.csv
NAME = $(basename $(DATA))
SPEC = $(NAME).spec
MODEL = model_complete_$(NAME)_12.h5
CONFIG = model_config_$(NAME)_12.json
BOUNDS = data_bounds_$(NAME)_12.json
GEN = model_gen_$(NAME)_12.json
NICE = nice -n 5
TRACE = $(VFY).trace
CATEGORIES = category

PNGS = \
	$(NAME)_12_eval-Test.png \
	$(NAME)_12_eval-Train.png \
	$(NAME)_12_resp-distr.png \
	$(NAME)_12_train-reg.png \

#UNBUFFER = $(shell command -v unbuffer)

ifneq ($(UNBUFFER),)
UNBUFFER_P = $(UNBUFFER) -p
endif

TRAIN_FLAGS = \
	-f $(TRAIN_FILTER) \
	-R $(TRAIN_SEED) \
	-s $(SPEC) \
	-r $(RESP) \
	-b $(TRAIN_BATCH) \
	-e $(TRAIN_EPOCHS) \
	-l $(TRAIN_LAYERS) \
	$(if $(OBJT),-O '$(OBJT)') \

TRAIN := $(BASE_DIR)/train-nn.py

SAFE = $(VFY).safe

VERIFY_FLAGS = $(VERIFY_FLAGS-$(VNAME)) \
	-d $(DATA) \
	-g $(GEN) \
	-b -B $(BOUNDS) \
	-C 0 \
	-n $(N) \
	-O '$(OBJT)' \
	-s $(SPEC) \
	-S $(SAFE).csv \
	-t 0.$(T) \
	-T 0.$(ST) \
	-x $(TRACE) \
	$(MODEL)
#	-o $(VFY).test.smt2 \

VERIFY := $(BASE_DIR)/prove-nn.py

SEARCH_FLAGS = $(SEARCH_FLAGS-$(VNAME)) \
	-d $(DATA) \
	-g $(GEN) \
	-b -B $(BOUNDS) \
	-C 0 \
	-n $(N) \
	-O '$(OBJT)' \
	-s $(SPEC) \
	-S $(SEARCH).safe.csv \
	-t $(T) \
	-U $(COFF) \
	-x $(SEARCH).trace \
	$(MODEL)

FILTER_TRACE := $(BASE_DIR)/filter-trace.sh

CHECK_DATA := $(BASE_DIR)/check-data
CHECK_DATA_FLAGS = \
	-r $(OBJT) \
	-s $(SPEC) \

COLLECT := $(BASE_DIR)/collect-results.sh

PREDICT_GRID := $(BASE_DIR)/nn_predict-grid.py
PREDICT_GRID_FLAGS = \
	-s $(SPEC) \
	-b $(BOUNDS) \
	-g $(GEN) \
	$(MODEL)

VFY = vfy$(T)ST$(ST)$(VNAME)
SEARCH = search$(VNAME)

ifeq ($(V),)
DISP = >/dev/null
else
DISP =
endif

include $(BASE_DIR)/defaults.mk
