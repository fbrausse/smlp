
PYTHON ?= $(shell command -v python3)
PYTHON_VERSION_MAJOR_MINOR := $(wordlist 2,3,$(subst ., ,$(shell $(PYTHON) --version)))
PYTHON_VERSION_MAJOR := $(word 1,$(PYTHON_VERSION_MAJOR_MINOR))
PYTHON_VERSION_MINOR := $(word 2,$(PYTHON_VERSION_MAJOR_MINOR))
NUMPY_INCLUDE_PATH := $(shell $(PYTHON) -c 'import numpy; print(numpy.get_include())')

BINS += \
	check-data-cmd \
	shai-prep-cmd \

DLIBS += \
	check-data \
	spec \
	shai-prep \

check-data-cmd-srcs := \
	$(dir)/check-data.c \

check-data-cmd-dep-dlibs = \
	check-data \

check-data-cmd-ldlibs = \
	-lcheck-data $(call pkg-libs,kjson) \

check-data-srcs := \
	$(dir)/check-data-lib.c \
	$(dir)/table.c \
	$(dir)/response.c \
	$(dir)/cbuf.c \

check-data-ldlibs = \
	$(call pkg-libs,kjson) -lm

$(call objs,$(BINS) $(DLIBS)): override CPPFLAGS += $(pkg-cflags,kjson)

shai-prep-srcs := \
	$(dir)/shai-prep.cc \

shai-prep-dep-dlibs = \
	spec \
	check-data \

shai-prep-ldlibs = \
	-lspec \
	-lcheck-data \
	#$(call pkg-libs,python-$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)-embed) \

ifneq ($(NUMPY_INCLUDE_PATH),)
$(call objs,shai-prep): override CPPFLAGS += -I$(NUMPY_INCLUDE_PATH) -DSMLP_PY $(call pkg-cflags,python-$(PYTHON_VERSION_MAJOR).$(PYTHON_VERSION_MINOR)-embed)
endif

shai-prep-cmd-srcs := \
	$(dir)/shai-prep-cmd.cc \

shai-prep-cmd-dep-dlibs = \
	spec \
	check-data \

shai-prep-cmd-ldlibs = \
	-lspec \
	-lcheck-data \

spec-srcs := \
	$(dir)/spec.c \
	$(dir)/spec.cc \

spec-ldlibs = \
	$(call pkg-libs,kjson)

# cc -O -std=c2x -I ../ext/kjson -Wall -Wextra -Wpedantic shai-prep.c -L ../ext/kjson -Wl,-rpath,../ext/kjson -lkjson
