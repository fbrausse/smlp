
LINKS = doc.pdf spec.pdf

.PHONY: all clean

all: $(LINKS)
clean:
	$(RM) $(LINKS)

.SECONDEXPANSION:

$(LINKS): $$(basename $$@)/$$@
	ln -sf $< $@

%:
	make -C $(@D) $(@F)
