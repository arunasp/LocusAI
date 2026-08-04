"""Shared functions for LocusAI's local MCP service scripts.

Mirrors the sysvinit /lib/lsb/init-functions convention: one common
library holding logic every service script needs, imported rather than
duplicated. bash_mcp_server.py is the first caller; any future
Python-based MCP service (not just git/bash) should import from here
too instead of reimplementing execution or logging.
"""

import subprocess
from pathlib import Path


class ExecutionResult:
    """Structured result from run_allowlisted, instead of a raw string."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def as_text(self) -> str:
        return (
            f"exit_code: {self.returncode}\n"
            f"stdout:\n{self.stdout}\n"
            f"stderr:\n{self.stderr}"
        )


def run_allowlisted(
    binary: str,
    args: list[str],
    allowed: frozenset[str],
    workdir: Path,
    timeout_seconds: int,
) -> str:
    """Run one allowlisted binary with the given argv, no shell involved.

    Shared by any service script that needs allowlisted execution —
    the allowlist itself and the timeout are supplied by the caller,
    this function only owns the "how", not the "what's permitted".
    """
    if binary not in allowed:
        return f"REFUSED: '{binary}' is not in the allowlist {sorted(allowed)}"

    try:
        result = subprocess.run(
            [binary, *args],
            cwd=workdir,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: '{binary}' exceeded {timeout_seconds}s"
    except FileNotFoundError:
        return f"ERROR: '{binary}' is allowlisted but not installed in this image"

    return ExecutionResult(result.returncode, result.stdout, result.stderr).as_text()
