FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        curl \
        gnupg \
        make \
        gcc \
        libc6-dev \
        libelf1 \
        libnuma1 \
        libdrm2 \
    && rm -rf /var/lib/apt/lists/*

# gcc/libc6-dev: so core/ can be compiled inside this container rather
# than only outside it -- this is the only container with GPU access,
# so a build here is a build that can then be run against the device.
#
# libelf1/libnuma1/libdrm2: ROCm's own runtime dependencies. ROCm is
# NOT installed in this image -- the host's /opt/rocm is bind-mounted
# read-only instead (see docker-compose.yml), which avoids a multi-GB
# image layer and keeps the container's ROCm version identical to the
# host's by construction. These three are what the mounted binaries
# need from the image side; verified by running rocminfo in a bare
# ubuntu container with the same mounts and adding exactly what it
# reported missing.

# gh CLI, official repo per current docs (verified 2026-08-06,
# https://github.com/cli/cli/blob/trunk/docs/install_linux.md) --
# Debian's own package is community-maintained and can lag behind
# fixes to deprecated GitHub API behavior, per that same doc.
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Auth split (Arunas's own design, 2026-08-06): `git push`/pull over
# ssh uses a read-only-mounted deploy key (see docker-compose.yml);
# everything else (PRs, issues, releases) uses `gh`'s own token via
# GH_TOKEN. Pin GitHub's published ed25519 host key here rather than
# trust-on-first-use -- verified live against
# https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints
# (fingerprint SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU,
# confirmed by decoding the key below with ssh-keygen, 2026-08-06).
# Deliberately only the ed25519 entry -- HostKeyAlgorithms below
# restricts negotiation to match, so there's no ambiguity about which
# host key type the connection could fall back to.
RUN mkdir -p -m 700 /root/.ssh \
    && echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" \
        > /root/.ssh/known_hosts \
    && chmod 600 /root/.ssh/known_hosts \
    && printf 'Host github.com\n    User git\n    IdentityFile /root/.ssh/github_deploy_key\n    IdentitiesOnly yes\n    HostKeyAlgorithms ssh-ed25519\n    UserKnownHostsFile /root/.ssh/known_hosts\n    StrictHostKeyChecking yes\n' \
        > /root/.ssh/config \
    && chmod 600 /root/.ssh/config

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ lib/
COPY bash_mcp_server.py .

WORKDIR /workspace
