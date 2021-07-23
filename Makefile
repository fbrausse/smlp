# default compiler flags
CSTD = -std=c11
CXXSTD = -std=c++17
OPTS = -O2
WARN = -Wall

.PHONY: build all clean

build: ext
	$(MAKE) all

# files appending to BINS, DLIBS and SLIBS, setting %-srcs, %-ldflags and %-ldlibs as needed
INCLUDES = src/build.mk

DIRS = doc ext

.PHONY: $(DIRS)

clean: clean-dirs

all-dirs: TGT=all
all-dirs: $(DIRS)

clean-dirs: TGT=clean
clean-dirs: $(DIRS)

$(DIRS):
	+$(MAKE) -C $@ $(TGT)

###################################
# no need to change below this line
###################################

# paths
BIN = bin
LIB = lib
OBJ = obj
DESTDIR ?= /usr/local

dir = $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
obj = $(patsubst src/%,$(OBJ)/%,$(addsuffix .o,$(1)))
objs = $(call obj,$(foreach b,$(1),$($(b)-srcs)))

# file name patterns
BINPAT  = $(BIN)/%$(BINSUF)
DLIBPAT = $(LIB)/lib%$(DLIBSUF)
SLIBPAT = $(LIB)/lib%$(SLIBSUF)

# paths
BINP  = $(patsubst %,$(BINPAT),$(BINS))
DLIBP = $(patsubst %,$(DLIBPAT),$(DLIBS))
SLIBP = $(patsubst %,$(SLIBPAT),$(SLIBS))

binp = $(patsubst %,$(BINPAT),$(1))
dlibp = $(patsubst %,$(DLIBPAT),$(1))
slibp = $(patsubst %,$(SLIBPAT),$(1))

ifeq ($(OS),Windows_NT)
  OS = Windows
else
  OS = $(shell uname)
endif

# suffixes
ifeq ($(OS),Windows)
  BINSUF = .exe
  DLIBSUF = .dll
  SLIBSUF = .a
else ifeq ($(OS),Darwin)
  BINSUF =
  DLIBSUF = .dylib
  SLIBSUF = .a
else
  BINSUF =
  DLIBSUF = .so
  SLIBSUF = .a
endif

# make-4.1 circumvents calling $(SHELL) -c if it is set to /bin/sh
ENV := /usr/bin/env
SHELL := $(ENV) LC_ALL=C LC_NUMERIC=C $(SHELL)

PKG_CONFIG = $(shell command -v pkg-config)
pkg-config = $(ENV) PKG_CONFIG_PATH=$(abspath $(LIB)/pkgconfig):$(PKG_CONFIG_PATH) $(PKG_CONFIG)
ifeq ($(PKG_CONFIG),)
 pkg-cflags =
 pkg-libs   = $(addprefix -l,$(1))
 pkg-slibs  = $(addprefix -l,$(1))
else
 pkg-cflags = $(shell $(pkg-config) --cflags $(1))
 pkg-libs   = $(shell $(pkg-config) --libs $(1))
 pkg-slibs  = $(shell $(pkg-config) --static --libs $(1))
endif

LD_STATIC  = -Wl,-Bstatic
LD_DYNAMIC = -Wl,-Bdynamic

BINS ?=
DLIBS ?=
SLIBS ?=

include $(INCLUDES)

all: $(BINP) $(DLIBP) $(SLIBP)

clean:
	$(RM) $(BINP) $(DLIBP) $(SLIBP) $(call objs,$(BINS) $(DLIBS) $(SLIBS)) $(subst .o,.d,$(call objs,$(BINS) $(DLIBS) $(SLIBS)))

ifeq ($(OS),Darwin)
  $(forall l,$(DLIBP),$(eval $(l): override private LDFLAGS += \
	-dynamiclib -install_name $(abspath $(DESTDIR)$(libdri))/$(l)))
else
#  $(OBJS)/%.c.o: override private CFLAGS += -pthread
#  $(OBJS)/%.cc.o: override private CXXFLAGS += -pthread
#  $(BINP) $(DLIBP): override private LDFLAGS += -pthread
endif

clinker = $(if $(filter %.cc.o,$(1)),C,c)
linker = $(call clinker,$(1))
ccompile = $(if $(filter %.cc,$(1)),C,c)
compile = $(call ccompile,$(1))

.SECONDEXPANSION:
$(BINP) $(DLIBP): private override LDFLAGS += -Llib -Wl,-rpath,'$$ORIGIN/../lib'
$(DLIBP): private override LDFLAGS += -shared
$(DLIBP): override CFLAGS += -fPIC
$(DLIBP): override CXXFLAGS += -fPIC
$(BINP): $(BINPAT): $$(call objs,$$*) $$(call slibp,$$($$*-dep-slibs)) | $$(call dlibp,$$($$*-dep-dlibs)) $$(dir $$@)
$(DLIBP): $(DLIBPAT): $$(call objs,$$*) $$(call slibp,$$($$*-dep-slibs)) | $$(call dlibp,$$($$*-dep-dlibs)) $$(dir $$@)
$(SLIBP): $(SLIBPAT): $$(call objs,$$*) $$(call slibp,$$($$*-dep-slibs)) | $$(dir $$@)
$(BINP) $(DLIBP):
	+$(LINK.$(call linker,$^)) $($*-ldflags) $(OPTS) $(OUTPUT_OPTION) $^ $($*-ldlibs)

$(SLIBP):
	$(AR) rcs $@ $^

$(OBJ)/%.o: override private CPPFLAGS += -Iinclude -MD $(OPTS) $(WARN)
$(OBJ)/%.o: override private CFLAGS += $(CSTD)
$(OBJ)/%.o: override private CXXFLAGS += $(CXXSTD)

$(OBJ)/%.o: src/% Makefile | $$(dir $$@)
	+$(COMPILE.$(call compile,$<)) $< $(OUTPUT_OPTION)

%/:
	mkdir -p $@

-include $(subst .o,.d,$(call objs,$(BINS) $(DLIBS) $(SLIBS)))
