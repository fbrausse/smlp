
BINS += check-data-cmd
DLIBS += check-data

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
