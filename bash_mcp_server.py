#!/usr/bin/env python3
"""Allowlisted command-execution MCP server.

Thin service script: this file owns only what's specific to the
bash-tools service (the allowlist, timeout, tool schema). Execution
logic lives in lib/common.py, shared with any future Python MCP
service -- same relationship as a sysvinit script calling into
/lib/lsb/init-functions instead of reimplementing start/stop logic.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from lib.common import run_allowlisted

WORKDIR = Path("/workspace")
TIMEOUT_SECONDS = 30

# Extend only with binaries you have actually reviewed. Anything not
# listed here is refused before subprocess ever runs.
ALLOWED_BINARIES = frozenset({
    "ls", "cat", "grep", "find", "wc", "head", "tail",
    "python3", "pip", "pytest",
})

mcp = FastMCP("bash-tools")


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
    mcp.run()  # stdio transport
