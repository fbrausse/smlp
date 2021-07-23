
ENV ?= /usr/bin/env
SHELL := $(ENV) LC_ALL=C LC_NUMERIC=C $(SHELL)
LATEXMK = $(shell command -v latexmk)

ifneq ($(LATEXMK),)
%.pdf: %.tex
	$(LATEXMK) -pdf $<
endif
