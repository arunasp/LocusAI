#!/usr/bin/env python3
"""Allowlisted command-execution MCP server.

Design constraints (deliberate, not defaults):
- No shell string is ever accepted or passed to a shell. Callers supply a
  binary name plus an argv list; subprocess.run always runs with
  shell=False, so shell metacharacters (;, |, &&, $(), backticks) have no
  special meaning and cannot chain or inject additional commands.
- Only binaries in ALLOWED_BINARIES may run at all.
- All execution is pinned to WORKDIR (the container's bind-mounted
  project directory) via cwd=, regardless of what the caller passes.
- Every call has a hard timeout so a hung process cannot stall the server.
"""

import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

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
    if binary not in ALLOWED_BINARIES:
        return f"REFUSED: '{binary}' is not in the allowlist {sorted(ALLOWED_BINARIES)}"

    try:
        result = subprocess.run(
            [binary, *args],
            cwd=WORKDIR,
            shell=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: '{binary}' exceeded {TIMEOUT_SECONDS}s"
    except FileNotFoundError:
        return f"ERROR: '{binary}' is allowlisted but not installed in this image"

    return (
        f"exit_code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    mcp.run()  # stdio transport
