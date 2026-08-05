#!/usr/bin/env python3
"""Allowlisted command-execution MCP server -- streamable-HTTP transport.

Runs as a persistent service (docker compose up -d, not run --rm),
listening on TCP so a Windows-side Claude Desktop can reach it via
WSL2's automatic localhost port forwarding -- confirmed real, not
assumed: a container publishing a port inside WSL2 is reachable from
Windows as localhost:<port> with zero extra configuration. The only
thing crossing the OS boundary is a plain HTTP request (via
mcp-remote on the Desktop side) -- no wsl.exe, no WSLENV, no bash
script execution across OSes at all.
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from lib.common import run_allowlisted

WORKDIR = Path("/workspace")
TIMEOUT_SECONDS = 30

ALLOWED_BINARIES = frozenset({
    "ls", "cat", "grep", "find", "wc", "head", "tail",
    "python3", "pip", "pytest",
})

mcp = FastMCP(
    "bash-tools",
    host="0.0.0.0",
    port=int(os.environ.get("MCP_PORT", "1443")),
)


@mcp.tool()
def run_command(binary: str, args: list[str]) -> str:
    """Run one allowlisted binary with the given argv inside the mounted
    project directory (/workspace). Returns combined stdout+stderr and the
    exit code as text. No shell is invoked; args are passed literally.

    Args:
        binary: executable name, must be in the server's allowlist.
        args: argv list for the binary (no shell operators apply).
    """
    return run_allowlisted(binary, args, ALLOWED_BINARIES, WORKDIR, TIMEOUT_SECONDS)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
