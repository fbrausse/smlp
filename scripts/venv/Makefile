SHELL           := /bin/bash
IMAGE           ?= readme_venv
DOCKERFILE      := Dockerfile.$(IMAGE)

.PHONY: build
.DEFAULT_GOAL := build

build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build --progress=plain -f $(DOCKERFILE) -t $(IMAGE)  . && \
	docker run --rm -it $(IMAGE) ./z3_version.py

help:
	@echo    ""
	@echo    "Usage: make [target] [IMAGE=<IMAGE_name>]" 
	@echo    ""
	@echo    "Targets:"
	@echo    "  build                               Normal build using Dockerfile.<IMAGE>  (default target)"
	@echo    "  help                                Show this message"
	@echo    ""
	@echo    "Example:"
	@echo    "  make IMAGE=readme_system"
	@echo    ""
