#!/usr/bin/env python3
"""Repository delivery steps, each one call with its checks built in.

  digests FILE...                      size and sha256 per file
  commit-verified --msg F FILE...      pipeline, stage exactly FILEs, commit
  squash --msg F [--upstream REF]      squash commits ahead of REF into one

Commit messages are read from a file (``--msg``); keep it under .git/
(for example .git/locus-msg.txt) so writing it never dirties the tree.

commit-verified runs the pipeline first (PIPELINE, default
`make -C core all`) and commits nothing unless it exits 0. It refuses
when anything outside FILEs is already staged, and reports tracked
files it leaves modified.

squash DESTROYS the branch's per-commit history. Before moving anything
it creates backup/pre-squash-<UTC stamp> at HEAD; that branch is the
surviving copy. It refuses on a dirty tracked tree, does nothing with
fewer than two commits ahead, and after committing checks that the new
commit's tree equals the old HEAD's tree and that its parent is REF. On
any failure after the reset it resets back to the backup and exits 1.
"""

import hashlib
import os
import shlex
import subprocess
import sys
import time


def git(*args, check=True):
    r = subprocess.run(("git",) + args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s: %s" % (" ".join(args),
                                           r.stderr.strip()))
    return r.stdout.strip()


def fail(msg, code=1):
    print("FAIL: %s" % msg)
    return code


def digests(files):
    bad = 0
    for f in files:
        try:
            with open(f, "rb") as fh:
                data = fh.read()
        except OSError as e:
            print("MISSING %s (%s)" % (f, e.strerror))
            bad += 1
            continue
        print("%8d %s %s" % (len(data), hashlib.sha256(data).hexdigest(), f))
    return 1 if bad else 0


def read_msg(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return text if text.strip() else None


def commit_verified(msg_path, files):
    if not files:
        return fail("no FILES given")
    if read_msg(msg_path) is None:
        return fail("message file missing or empty: %s" % msg_path)
    staged = set(git("diff", "--cached", "--name-only").splitlines())
    extra = staged - set(files)
    if extra:
        return fail("already staged outside FILES: %s" % " ".join(
            sorted(extra)))
    cmd = shlex.split(os.environ.get("PIPELINE", "make -C core all"))
    print("pipeline: %s" % " ".join(cmd))
    sys.stdout.flush()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return fail("pipeline exited %d; nothing staged or committed" % rc)
    git("add", "--", *files)
    staged = git("diff", "--cached", "--name-only").splitlines()
    if not staged:
        return fail("FILEs have no changes to commit")
    git("commit", "-q", "-F", msg_path)
    print(git("log", "-1", "--stat", "--format=committed %h %s"))
    left = git("status", "--porcelain=v1", "--untracked-files=no")
    if left:
        print("tracked files still modified:\n%s" % left)
    return 0


def squash(msg_path, upstream):
    if read_msg(msg_path) is None:
        return fail("message file missing or empty: %s" % msg_path)
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        return fail("tracked changes present; commit or stash first")
    try:
        base = git("rev-parse", "--verify", upstream + "^{commit}")
    except RuntimeError as e:
        return fail(str(e))
    ahead = int(git("rev-list", "--count", upstream + "..HEAD"))
    if ahead < 2:
        print("nothing to squash: %d commit(s) ahead of %s"
              % (ahead, upstream))
        return 0
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    backup = "backup/pre-squash-" + time.strftime("%Y%m%d-%H%M%S",
                                                  time.gmtime())
    git("branch", backup, head)
    print("backup: %s at %s (%d commits)" % (backup, head[:7], ahead))
    try:
        git("reset", "--soft", base)
        git("commit", "-q", "-F", msg_path)
        new_tree = git("rev-parse", "HEAD^{tree}")
        parent = git("rev-parse", "HEAD^")
        if new_tree != tree or parent != base:
            raise RuntimeError("tree or parent mismatch after squash")
    except RuntimeError as e:
        git("reset", "--soft", head, check=False)
        return fail("%s; HEAD restored to %s" % (e, head[:7]))
    print("squashed %d commits into %s on %s; tree identical" % (
        ahead, git("rev-parse", "--short", "HEAD"), upstream))
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    msg = None
    upstream = "@{upstream}"
    files = []
    i = 0
    while i < len(rest):
        if rest[i] == "--msg" and i + 1 < len(rest):
            msg = rest[i + 1]
            i += 2
        elif rest[i] == "--upstream" and i + 1 < len(rest):
            upstream = rest[i + 1]
            i += 2
        elif rest[i] == "--":
            files.extend(rest[i + 1:])
            break
        else:
            files.append(rest[i])
            i += 1
    if cmd == "digests":
        return digests(files)
    if cmd == "commit-verified":
        return commit_verified(msg, files)
    if cmd == "squash":
        return squash(msg, upstream)
    print("unknown command: %s" % cmd)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except RuntimeError as e:
        raise SystemExit(fail(str(e)))
