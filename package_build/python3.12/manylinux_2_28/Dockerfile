FROM quay.io/pypa/manylinux_2_28_x86_64

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
RUN dnf install -y \
        wget git make m4 pkg-config && \
    dnf clean all

# ---------------------------------------------------------------------------
# 2. Set Python 3.12 from pre-installed pypa versions
# ---------------------------------------------------------------------------
ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"
ENV PYTHON=/opt/python/cp312-cp312/bin/python3.12
ENV PIP=/opt/python/cp312-cp312/bin/pip

# ---------------------------------------------------------------------------
# 3. Python build tools
# ---------------------------------------------------------------------------
RUN /opt/python/cp312-cp312/bin/pip install --upgrade pip && \
    /opt/python/cp312-cp312/bin/pip install setuptools==80.10.1 && \
    /opt/python/cp312-cp312/bin/pip install meson ninja z3-solver==4.8.12 auditwheel patchelf && \
    ln -sf /opt/python/cp312-cp312/bin/python3.12 /usr/local/bin/python3.12 && \
    ln -sf /opt/python/cp312-cp312/bin/pip /usr/local/bin/pip3.12 && \
    ln -sf /opt/python/cp312-cp312/bin/meson /usr/local/bin/meson && \
    ln -sf /opt/python/cp312-cp312/bin/ninja /usr/local/bin/ninja

# ---------------------------------------------------------------------------
# 4. Add ~/.local/bin to PATH
# ---------------------------------------------------------------------------
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------------------------
# 5. Clone smlp and optionaly switch to branch
# ---------------------------------------------------------------------------
# Point setup.py to the correct z3 location for this Python installation
ENV Z3_PREFIX=/opt/python/cp312-cp312/lib/python3.12/site-packages/z3

WORKDIR /app
ARG GIT_BRANCH=smlp_python312
ARG CACHE_BUST_SMLP
COPY run_git_clone .
RUN ./run_git_clone $GIT_BRANCH

# ---------------------------------------------------------------------------
# 6. Build wheel
# ---------------------------------------------------------------------------
WORKDIR smlp
RUN /opt/python/cp312-cp312/bin/python3.12 -m pip -v wheel . -w dist/ && \
    /opt/python/cp312-cp312/bin/python3.12 repair_wheel.py dist/ --plat manylinux_2_28_x86_64

CMD ["/usr/bin/bash"]
# The manylinux wheel is in /app/smlp/dist/
