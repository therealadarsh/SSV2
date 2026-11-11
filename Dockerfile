# syntax=docker/dockerfile:1
#
# Stages, so the final image holds only what runs in production:
#   uv       -> the uv binary (an image of its own, so Dependabot can bump its digest)
#   builder  -> installs the locked dependencies, then the ssv2 package, into /opt/venv
#   runtime  -> a fresh slim Python with just /opt/venv; runs as a non-root user
# Phase 1 adds an `index` stage that downloads the pinned model and builds the index.
#
# Every base image is pinned by digest: a tag like `3.12-slim` can be re-pointed upstream,
# a digest can't. Dependabot opens PRs to bump both together.

# ---------------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 AS uv

# ---------------------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Use the image's Python; never download one.
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    # Compile .pyc files now: the runtime filesystem is read-only, and startup is faster.
    UV_COMPILE_BYTECODE=1 \
    # The cache is a mount, so copy files into the venv instead of hard-linking to it.
    UV_LINK_MODE=copy

WORKDIR /src

# Layer 1: third-party dependencies only, from uv.lock. This layer (PyTorch included) is
# rebuilt only when pyproject.toml or uv.lock change, not when the code changes.
# The optional `extra_ca` secret is for building behind a TLS-intercepting proxy:
#   docker build --secret id=extra_ca,src=/path/to/ca-bundle.crt .
# It is mounted only for this command and never stored in a layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --locked --no-dev --no-install-project

# Layer 2: the ssv2 package itself, installed as a regular (non-editable) package, so the
# runtime image needs the venv only, not the source tree.
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export SSL_CERT_FILE=/run/secrets/extra_ca; fi; \
    uv sync --locked --no-dev --no-editable

# ---------------------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Nothing may be downloaded at runtime: the image must contain everything it needs.
    HF_HUB_OFFLINE=1 \
    PORT=8000

# Install Debian security fixes released since the base image was built (the Trivy scan in CI
# fails on fixable HIGH/CRITICAL CVEs). Once upstream rebuilds the base image, Dependabot
# bumps the digest and this becomes a no-op. apt uses HTTPS so the build needs only port 443;
# package signatures are checked either way. (mode=0444: apt downloads as the `_apt` user.)
RUN --mount=type=secret,id=extra_ca,required=false,mode=0444 \
    sed -i 's|http://deb.debian.org|https://deb.debian.org|' /etc/apt/sources.list.d/debian.sources \
 && if [ -f /run/secrets/extra_ca ]; then \
      echo 'Acquire::https::CAInfo "/run/secrets/extra_ca";' > /etc/apt/apt.conf.d/99extra-ca; \
    fi \
 && apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99extra-ca

RUN useradd --create-home --uid 10001 app
# Owned by root and read-only to `app`: a compromised process can't modify the code it runs.
COPY --from=builder /opt/venv /opt/venv

USER app
WORKDIR /home/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=4)"

# `sh -c` so ${PORT} is expanded at runtime (many hosts inject PORT); `exec` so uvicorn
# replaces the shell as PID 1 and receives SIGTERM for a graceful shutdown.
CMD ["sh", "-c", "exec uvicorn ssv2.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
