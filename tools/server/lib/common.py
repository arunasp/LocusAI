"""Shared functions for LocusAI's local MCP service scripts.

Mirrors the sysvinit /lib/lsb/init-functions convention: one common
library holding logic every service script needs, imported rather than
duplicated. bash_mcp_server.py is the first caller; any future
Python-based MCP service (not just git/bash) should import from here
too instead of reimplementing execution or logging.
"""

import subprocess
from collections.abc import Iterable
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


class FileAllowlist:
    """A binary allowlist backed by a text file, refreshed on change.

    One binary name per line; blank lines and '#' comments are
    ignored. The file is only re-read when its mtime moves, so a hot
    edit on the host (this path is expected to live under the
    /workspace bind mount, not the image's COPY'd /app layer) takes
    effect on the next request with no container restart -- while a
    request that doesn't touch the file at all pays only a stat(),
    not a full re-parse.
    """

    __slots__ = ("_path", "_mtime", "_binaries")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float | None = None
        self._binaries: frozenset[str] = frozenset()
        self._reload_if_changed()

    def _reload_if_changed(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            # Fail closed, not open: a missing allowlist file means
            # nothing is permitted, not everything.
            self._mtime = None
            self._binaries = frozenset()
            return

        if mtime == self._mtime:
            return

        entries = set()
        for line in self._path.read_text().splitlines():
            name = line.split("#", 1)[0].strip()
            if name:
                entries.add(name)

        self._binaries = frozenset(entries)
        self._mtime = mtime

    def __contains__(self, binary: str) -> bool:
        self._reload_if_changed()
        return binary in self._binaries

    def __iter__(self):
        self._reload_if_changed()
        return iter(self._binaries)


def run_allowlisted(
    binary: str,
    args: list[str],
    allowed: Iterable[str],
    workdir: Path,
    timeout_seconds: int,
) -> str:
    """Run one allowlisted binary with the given argv, no shell involved.

    Shared by any service script that needs allowlisted execution —
    the allowlist itself and the timeout are supplied by the caller,
    this function only owns the "how", not the "what's permitted".
    `allowed` only needs to support `in` and iteration -- a frozenset
    and a FileAllowlist both satisfy that.
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
