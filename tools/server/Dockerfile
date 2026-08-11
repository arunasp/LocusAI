FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openssh-client \
        curl \
        gnupg \
        make \
        gcc \
        g++ \
        libc6-dev \
        util-linux \
        libxml2 \
        libelf1 \
        libnuma1 \
        libdrm2 \
    && rm -rf /var/lib/apt/lists/*

# libxml2: ROCm's bundled lld links against libxml2.so.2, so hipcc fails
# at the amdgcn-link step without it -- found by running `make -C core gpu`
# here, which got past compilation and died in the linker.
#
# util-linux: supplies setpriv, which entrypoint.sh uses to drop from
# root to the invoking host uid. Chosen over su/sudo/gosu because it is
# already standard in a Debian image and its --reset-env re-initialises
# HOME, SHELL, USER and PATH from the target uid's own passwd entry
# rather than leaving root's environment behind under another uid.
#
# gcc/libc6-dev: so core/ can be compiled inside this container rather
# than only outside it -- this is the only container with GPU access,
# so a build here is a build that can then be run against the device.
#
# g++: HIP is C++ and hipcc is a clang++ driver, so the GPU probe and any
# later kernel need the C++ standard library headers. gcc alone is not
# enough -- found by running `make -C core gpu` here and reading the
# error, which was a missing <cmath> after hipcc had already correctly
# auto-detected --offload-arch=gfx1100.
#
# libelf1/libnuma1/libdrm2: ROCm's own runtime dependencies. ROCm is
# NOT installed in this image -- the host's /opt/rocm is bind-mounted
# read-only instead (see docker-compose.yml), which avoids a multi-GB
# image layer and keeps the container's ROCm version identical to the
# host's by construction. These three are what the mounted binaries
# need from the image side; verified by running rocminfo in a bare
# ubuntu container with the same mounts and adding exactly what it
# reported missing.

# The gh CLI and the ssh deploy-key setup that used to live here were
# removed 2026-08-10. Both belonged to the Aug 6 auth split, which
# cicd_runner has since superseded: deployment goes through that runner,
# and this container's scope is now testing project code only. Removing
# them is also what makes the uid drop below possible -- /root/.ssh is
# mode 700, so a process running as the host user could not traverse it,
# and leaving the mount in place would have looked live while being
# unreadable. The pinned ed25519 host key and IdentityFile config remain
# in git history if the capability is ever wanted back.

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ lib/
COPY bash_mcp_server.py .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /workspace
