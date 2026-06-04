SHELL           := /bin/bash
IMAGE           ?= smlp-wheel_2_28-python311
IMAGES          := $(shell ls  Dockerfile.* | cut -d. -f2- | sort -r)
space           := $(subst x, ,x)
IMAGES_DISPLAY  := $(subst $(space),\n                                        ,$(IMAGES))
DOCKERFILE      := Dockerfile.$(IMAGE)

GIT_BRANCH      := $(shell git branch --show-current)
SMLP_COMMIT     := $(shell git ls-remote https://github.com/SMLP-Systems/smlp.git refs/heads/$(GIT_BRANCH) | cut -f1)
RELEASE_TYPE    ?=
INSTALL_MODE    ?=
WHEEL_DST       := ../dist

ifeq ($(RELEASE_TYPE),test)
    RELEASE_BUILD_ARG := --build-arg RELEASE_TYPE=-test
else
    RELEASE_BUILD_ARG := --build-arg RELEASE_TYPE=
endif

ifeq ($(INSTALL_MODE),system)
    INSTALL_MODE_ARG := --build-arg INSTALL_MODE=system
    IMAGE_TAG        := $(IMAGE)-system
else ifeq ($(INSTALL_MODE),mac)
    INSTALL_MODE_ARG := --platform linux/arm64 --load
    IMAGE_TAG        := $(IMAGE)-mac
else
    INSTALL_MODE_ARG := --build-arg INSTALL_MODE=venv
    IMAGE_TAG        := $(IMAGE)
endif

define copy_out
	@if [[ "$(IMAGE_TAG)" == smlp-wheel* ]]; then \
		echo "Copying wheel from container IMAGE_TAG $(IMAGE_TAG)..."; \
		mkdir -p $(WHEEL_DST); \
		docker create --name smlp_tmp $(IMAGE_TAG); \
		docker cp smlp_tmp:/app/smlp/dist/. /tmp/smlp_dist_tmp; \
		docker rm smlp_tmp; \
                mv /tmp/smlp_dist_tmp/smlptech-*manylinux*.whl $(WHEEL_DST)/ ; \
                rm -rf /tmp/smlp_dist_tmp; \
	else \
		echo "Skipping copy_out: IMAGE_TAG=$(IMAGE_TAG) does not start with 'smlp-wheel'"; \
	fi
endef

.PHONY: build rebuild_smlp clean_build help
.DEFAULT_GOAL := build

build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		$(INSTALL_MODE_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE_TAG) \
		.
	$(copy_out)

rebuild_smlp: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--build-arg CACHE_BUST_SMLP=$(SMLP_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		$(INSTALL_MODE_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE_TAG) \
		.
	$(copy_out)

clean_build: $(DOCKERFILE)
	env DOCKER_BUILDKIT=1 docker build \
		--no-cache \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		$(RELEASE_BUILD_ARG) \
		$(INSTALL_MODE_ARG) \
		--progress=plain \
		-f $(DOCKERFILE) \
		-t $(IMAGE_TAG) \
		.
	$(copy_out)

help:
	@echo    ""
	@echo    "Usage: make [target] [IMAGE=<image_name>] [RELEASE_TYPE=test] [SMLP_COMMIT=<tagname>] [INSTALL_MODE=<system|mac>]"
	@echo    ""
	@echo    "Targets:"
	@echo    "  build :                               Incremental build using Dockerfile.<image_name> (default target). Docker cache is used"
	@echo -e "    supported images:                   $(IMAGES_DISPLAY)"
	@echo    "    default image:                      $(IMAGE)"
	@echo    "  rebuild_smlp :                        Force smlp clone after commit pushed to remote branch"
	@echo -e "  rebuild_smlp SMLP_COMMIT=<tagname> :  Force smlp clone after commit or \033[1mtag pushed to remote branch\033[0m"
	@echo    "  clean_build :                         Full rebuild. Docker cache is ignored"
	@echo    "  RELEASE_TYPE=test  :                  The latest version installed on https://test.pypi.org/project/smlptech/#history will be used"
	@echo    "                                          By default version installed on https://pypi.org/project/smlptech/#history will be used"
	@echo    "  INSTALL_MODE=system :                 Package installation for ubuntu_24.04 will be done using sudo"
	@echo -e "  INSTALL_MODE=mac :                    Image with be built for Mac Apple silicon. \033[1mSupported only for IMAGE=python311-dev\033[0m"
	@echo    "                                          By default package installation for ubuntu_24.04 will be done for Intel/AMD silicon in virtual environment"
	@echo    "  help :                                Show this message"
	@echo    ""
	@echo    "Example:"
	@echo    "  make build IMAGE=python311-dev"
	@echo    ""
