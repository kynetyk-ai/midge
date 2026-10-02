# Image content for the midge sandbox workload. Runtime settings (network,
# credentials, skills) are in midge.yaml; plain `docker run` does not apply them.

# Docker's agent-less template: Ubuntu, the `agent` user (UID 1000, sudo),
# git, gh, curl, ripgrep, jq, Node, Python 3, uv, Docker CLI, and the
# certificate/BASH_ENV setup the sandbox proxy expects.
FROM docker/sandbox-templates:shell

USER root

# Compilers and headers for Python packages that build from source.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

# HTTPS from the sandbox goes through a TLS-inspecting proxy whose CA the
# sandbox adds to the system bundle at startup. Tools that ship their own CA
# list (uv, conda, requests, httpx, Node) must be pointed at that bundle.
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    CONDA_SSL_VERIFY=/etc/ssl/certs/ca-certificates.crt

# Shared, root-owned locations so tools are usable by `agent` and survive
# the home directory being populated at sandbox creation.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin

# Poetry, installed as an isolated uv tool.
ARG POETRY_VERSION=2.2.1
RUN uv tool install "poetry==${POETRY_VERSION}"

# Conda via Miniforge (conda-forge channel by default). Only condabin is put
# on PATH, so `python` still resolves to the system interpreter unless an
# environment is activated.
ARG MINIFORGE_VERSION=latest
RUN arch="$(uname -m)" \
 && if [ "$MINIFORGE_VERSION" = latest ]; then \
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"; \
    else \
      url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${arch}.sh"; \
    fi \
 && curl -fsSL "$url" -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/conda \
 && rm /tmp/miniforge.sh \
 && /opt/conda/bin/conda clean -afy \
 && chown -R agent:agent /opt/conda
ENV PATH="/opt/conda/condabin:${PATH}"

# midge itself, from the lock file, in its own Python 3.13 virtualenv so the
# agent's `pip install` or `uv pip install` in a project cannot break it.
ARG MIDGE_REPO=https://github.com/kynetyk-ai/midge.git
# Set by `args.ref` in midge.yaml.
ARG MIDGE_REF
RUN git clone "$MIDGE_REPO" /opt/midge \
 && git -C /opt/midge checkout "$MIDGE_REF" \
 && uv venv --python 3.13 /opt/midge/.venv \
 && cd /opt/midge \
 && POETRY_VIRTUALENVS_IN_PROJECT=true POETRY_NO_INTERACTION=1 \
    poetry install --only main --extras tui
 && ln -s /opt/midge/.venv/bin/midge /usr/local/bin/midge

# Sandbox-environment skill, exposed to midge by agent-skill@1 in midge.yaml.
COPY skills/sandbox-env /usr/share/midge-kit/skills/sandbox-env

USER agent
ENV COLORTERM=truecolor
WORKDIR /home/agent/workspace
ENTRYPOINT ["midge"]
CMD []
