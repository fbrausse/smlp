
EXT = $(wildcard */)

.PHONY: all clean $(EXT)

all: TGT=install
all: $(EXT)

$(EXT):
	+$(MAKE) -C $@ DESTDIR=$(realpath ../) $(TGT)

clean: TGT=uninstall clean
clean: $(EXT)
