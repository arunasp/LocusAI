"""Scenario driver for gitops.py against throwaway repos with an upstream."""
import os, subprocess, sys, tempfile, shutil
G = os.path.abspath("gitops.py")
results = []

def sh(*a, cwd, check=True, env=None):
    r = subprocess.run(a, cwd=cwd, capture_output=True, text=True, env=env)
    if check and r.returncode:
        raise SystemExit("setup failed: %s\n%s" % (a, r.stderr))
    return r.stdout.strip()

def repo(ahead):
    root = tempfile.mkdtemp()
    remote, work = os.path.join(root, "r.git"), os.path.join(root, "w")
    sh("git", "init", "-q", "--bare", remote, cwd=root)
    sh("git", "clone", "-q", remote, work, cwd=root)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        sh("git", "config", k, v, cwd=work)
    open(os.path.join(work, "a.txt"), "w").write("base\n")
    sh("git", "add", "a.txt", cwd=work); sh("git", "commit", "-qm", "base", cwd=work)
    sh("git", "push", "-q", "-u", "origin", "HEAD", cwd=work)
    for i in range(ahead):
        open(os.path.join(work, "f%d.txt" % i), "w").write("%d\n" % i)
        sh("git", "add", ".", cwd=work); sh("git", "commit", "-qm", "c%d" % i, cwd=work)
    msg = os.path.join(work, ".git", "locus-msg.txt")
    open(msg, "w").write("squashed\n\nbody\n")
    return root, work, msg

def run(work, *args, env=None):
    e = dict(os.environ, **(env or {}))
    r = subprocess.run((sys.executable, G) + args, cwd=work, capture_output=True, text=True, env=e)
    return r.returncode, r.stdout + r.stderr

def check(name, cond, out=""):
    results.append((name, bool(cond)))
    print("%-58s %s" % (name, "ok" if cond else "FAILED\n" + out))

# S1 clean, 3 ahead
root, w, msg = repo(3)
head, tree = sh("git", "rev-parse", "HEAD", cwd=w), sh("git", "rev-parse", "HEAD^{tree}", cwd=w)
rc, out = run(w, "squash", "--msg", msg)
check("S1 squash: exit 0", rc == 0, out)
check("S1 squash: one commit ahead", sh("git", "rev-list", "--count", "@{upstream}..HEAD", cwd=w) == "1", out)
check("S1 squash: tree identical", sh("git", "rev-parse", "HEAD^{tree}", cwd=w) == tree, out)
bk = [b for b in sh("git", "branch", "--list", "backup/*", "--format=%(refname:short) %(objectname)", cwd=w).splitlines()]
check("S1 squash: backup branch at old HEAD", len(bk) == 1 and bk[0].split()[1] == head, out)
check("S1 squash: message used", sh("git", "log", "-1", "--format=%s", cwd=w) == "squashed", out)
shutil.rmtree(root)

# S2 dirty tracked
root, w, msg = repo(3)
head = sh("git", "rev-parse", "HEAD", cwd=w)
open(os.path.join(w, "a.txt"), "a").write("dirty\n")
rc, out = run(w, "squash", "--msg", msg)
check("S2 dirty: refused (exit 1)", rc == 1 and "tracked changes" in out, out)
check("S2 dirty: HEAD unchanged, no backup", sh("git", "rev-parse", "HEAD", cwd=w) == head
      and not sh("git", "branch", "--list", "backup/*", cwd=w), out)
shutil.rmtree(root)

# S3 one ahead
root, w, msg = repo(1)
head = sh("git", "rev-parse", "HEAD", cwd=w)
rc, out = run(w, "squash", "--msg", msg)
check("S3 one ahead: no-op exit 0", rc == 0 and "nothing to squash" in out, out)
check("S3 one ahead: HEAD unchanged", sh("git", "rev-parse", "HEAD", cwd=w) == head, out)
shutil.rmtree(root)

# S4 untracked file survives
root, w, msg = repo(2)
open(os.path.join(w, "untracked.log"), "w").write("keep\n")
rc, out = run(w, "squash", "--msg", msg)
check("S4 untracked: squash proceeds", rc == 0, out)
check("S4 untracked: file survives", os.path.exists(os.path.join(w, "untracked.log")), out)
shutil.rmtree(root)

# S5 commit fails after the reset -> restored
root, w, msg = repo(3)
head, tree = sh("git", "rev-parse", "HEAD", cwd=w), sh("git", "rev-parse", "HEAD^{tree}", cwd=w)
hook = os.path.join(w, ".git", "hooks", "commit-msg")
open(hook, "w").write("#!/bin/sh\necho hook-rejects >&2\nexit 1\n"); os.chmod(hook, 0o755)
rc, out = run(w, "squash", "--msg", msg)
check("S5 commit fails: exit 1", rc == 1 and "restored" in out, out)
check("S5 commit fails: HEAD restored", sh("git", "rev-parse", "HEAD", cwd=w) == head, out)
check("S5 commit fails: index/tree intact", sh("git", "status", "--porcelain", "--untracked-files=no", cwd=w) == "", out)
shutil.rmtree(root)

# S6 missing message
root, w, msg = repo(3)
head = sh("git", "rev-parse", "HEAD", cwd=w)
rc, out = run(w, "squash", "--msg", "/nonexistent")
check("S6 no message: refused, nothing moved", rc == 1 and sh("git", "rev-parse", "HEAD", cwd=w) == head, out)
shutil.rmtree(root)

# commit-verified
root, w, msg = repo(0)
open(os.path.join(w, "a.txt"), "a").write("change\n")
open(os.path.join(w, "b.txt"), "w").write("other\n"); sh("git", "add", "b.txt", cwd=w); sh("git", "commit", "-qm", "b", cwd=w)
open(os.path.join(w, "b.txt"), "a").write("unlisted change\n")
rc, out = run(w, "commit-verified", "--msg", msg, "a.txt", env={"PIPELINE": "false"})
check("C2 pipeline fails: exit 1, nothing committed", rc == 1 and "pipeline exited" in out
      and sh("git", "log", "-1", "--format=%s", cwd=w) == "b", out)
check("C2 pipeline fails: nothing staged", sh("git", "diff", "--cached", "--name-only", cwd=w) == "", out)
rc, out = run(w, "commit-verified", "--msg", msg, "a.txt", env={"PIPELINE": "true"})
check("C1 pipeline passes: exit 0", rc == 0, out)
check("C1 commit holds exactly a.txt", sh("git", "show", "--name-only", "--format=", "HEAD", cwd=w) == "a.txt", out)
check("C5 unlisted modified file reported, not committed", "b.txt" in out
      and "b.txt" in sh("git", "status", "--porcelain", cwd=w), out)
sh("git", "add", "b.txt", cwd=w)
open(os.path.join(w, "a.txt"), "a").write("more\n")
rc, out = run(w, "commit-verified", "--msg", msg, "a.txt", env={"PIPELINE": "true"})
check("C3 other file pre-staged: refused", rc == 1 and "already staged" in out, out)
sh("git", "reset", "-q", "b.txt", cwd=w); sh("git", "checkout", "-q", "--", "a.txt", cwd=w)
rc, out = run(w, "commit-verified", "--msg", msg, "a.txt", env={"PIPELINE": "true"})
check("C4 listed file unchanged: refused", rc == 1 and "no changes" in out, out)
shutil.rmtree(root)

# digests
d = tempfile.mkdtemp(); open(os.path.join(d, "x"), "w").write("abc")
rc, out = run(d, "digests", "x", "missing")
check("D1 digests: hash printed, missing file fails", rc == 1 and
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" in out and "MISSING missing" in out, out)
shutil.rmtree(d)

failed = [n for n, ok in results if not ok]
print("=== GITOPS VERIFICATION: %d/%d passed ===" % (len(results) - len(failed), len(results)))
raise SystemExit(1 if failed else 0)
