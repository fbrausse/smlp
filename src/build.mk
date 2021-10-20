
BINS += \
	check-data-cmd \
	shai-prep \

DLIBS += \
	check-data \
	spec

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

spec-srcs := \
	$(dir)/spec.c \
	$(dir)/spec.cc \

spec-ldlibs = \
	$(call pkg-libs,kjson)

# cc -O -std=c2x -I ../ext/kjson -Wall -Wextra -Wpedantic shai-prep.c -L ../ext/kjson -Wl,-rpath,../ext/kjson -lkjson
