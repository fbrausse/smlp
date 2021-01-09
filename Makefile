
.PHONY: all

all: protocol.pb
	protoc --python_out=. $<
