FROM python:3.12-slim

# git binary itself, for mcp-server-git to shell out to
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lib/ lib/
COPY bash_mcp_server.py .

WORKDIR /workspace

# No default CMD -- the two Desktop config entries in README.md override
# this per tool, so one image serves both roles.
