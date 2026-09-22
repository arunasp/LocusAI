"""What is running, what it left behind, and how to stop it.

  jobs.py ROOT list            what every recorded job is doing now
  jobs.py ROOT stop NAME       stop one job's whole process tree
  jobs.py ROOT clean           report orphans and zombies, drop dead records

`make bg` records each job as ROOT/jobs/NAME.json (pid, command, log,
start time). This reads /proc directly -- no `ps` in the image -- and
answers three questions that were being guessed at:

- is the job still running, and is its log finished (a last line of
  exit=N)?
- did an interrupted turn leave a process tree with no owner, or a
  zombie whose parent never reaped it?
- can the job actually be stopped? PIDS DIFFER BY NAMESPACE: `docker
  top` prints the HOST pids while everything inside the container sees
  its own, so signalling a pid read from `docker top` inside the
  container hits nothing and looks exactly like an unkillable process.
  That mistake was made on 2026-09-22 and read as a GPU driver hang;
  with the in-container pid the same processes died at once. Stopping a
  job through this tool always uses the pid the job recorded, which is
  the in-container one.

  A process that genuinely survives SIGKILL is stuck in the driver, and
  then only restarting the container releases the device -- but check
  the namespace first.
"""

import json
import os
import signal
import sys
import time

HZ = os.sysconf("SC_CLK_TCK")
# How long to wait for a signalled process to actually die. A bound on
# patience, not a tuned value: the loop below polls and leaves as soon
# as the tree is gone.
GRACE = 15.0


def alive(pid):
    """A zombie is finished work waiting to be reaped, not a process
    still running: treating it as alive made a successful kill look like
    a failure, and would have kept the unkillable-GPU message for jobs
    that had in fact stopped."""
    s = stat_of(pid)
    return s is not None and s["state"] != "Z"


def stat_of(pid):
    """(name, state, ppid, cpu_seconds, start_seconds) or None."""
    try:
        with open("/proc/%d/stat" % pid) as fh:
            raw = fh.read()
    except OSError:
        return None
    name = raw[raw.find("(") + 1:raw.rfind(")")]
    rest = raw[raw.rfind(")") + 2:].split()
    cpu = (int(rest[11]) + int(rest[12])) / HZ
    return {"pid": pid, "name": name, "state": rest[0],
            "ppid": int(rest[1]), "cpu": cpu,
            "start": int(rest[19]) / HZ}


def cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def all_procs():
    out = []
    for e in os.listdir("/proc"):
        if e.isdigit():
            s = stat_of(int(e))
            if s:
                s["cmd"] = cmdline(int(e))
                out.append(s)
    return out


def tree(pid, procs):
    """pid and every descendant, parents before children."""
    kids = {}
    for p in procs:
        kids.setdefault(p["ppid"], []).append(p["pid"])
    seen, stack = [], [pid]
    while stack:
        q = stack.pop(0)
        seen.append(q)
        stack += kids.get(q, [])
    return seen


def records(root):
    d = os.path.join(root, "jobs")
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                try:
                    out.append(json.load(fh))
                except ValueError:
                    pass
    return out


def log_state(path):
    """'running', 'exit=N', or 'no log'."""
    if not path or not os.path.exists(path):
        return "no log"
    with open(path, errors="replace") as fh:
        last = ""
        for line in fh:
            if line.strip():
                last = line.strip()
    return last if last.startswith("exit=") else "running"


def cmd_list(root, procs):
    recs = records(root)
    if not recs:
        print("no jobs recorded under %s/jobs" % root)
        return 0
    by_pid = {p["pid"]: p for p in procs}
    print("%-16s %8s %6s %9s %10s  %s"
          % ("job", "pid", "state", "cpu", "log", "alive in tree"))
    for r in recs:
        pid = r.get("pid", 0)
        p = by_pid.get(pid)
        live = ([q for q in tree(pid, procs)
                 if q in by_pid and by_pid[q]["state"] != "Z"]
                if p else [])
        print("%-16s %8d %6s %8.1fs %10s  %d"
              % (r.get("job", "?"), pid, p["state"] if p else "-",
                 p["cpu"] if p else 0.0, log_state(r.get("log")),
                 len(live)))
    return 0


def cmd_stop(root, procs, name):
    rec = [r for r in records(root) if r.get("job") == name]
    if not rec:
        print("no record for job %s" % name)
        return 2
    pid = rec[0].get("pid", 0)
    live = {p["pid"]: p for p in procs}
    targets = [q for q in tree(pid, procs) if q in live
               and live[q]["state"] != "Z"]
    if not targets:
        print("%s: nothing running (pid %d)" % (name, pid))
        return 0
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for q in reversed(targets):
            try:
                os.kill(q, sig)
            except OSError:
                pass
        # POLL rather than sleep a fixed time: a signalled process dies
        # when it is next scheduled, and on a machine already saturated
        # by a GPU job that took longer than the 2 s this used to wait,
        # reporting a successful kill as a survivor.
        deadline = time.time() + GRACE
        while time.time() < deadline:
            targets = [q for q in targets if alive(q)]
            if not targets:
                print("%s: stopped" % name)
                return 0
            time.sleep(0.1)
    print("%s: %d process(es) SURVIVED SIGKILL: %s"
          % (name, len(targets), ", ".join(str(q) for q in targets)))
    for q in targets:
        s = stat_of(q)
        if s:
            print("   pid %d state %s cpu %.1fs  %s"
                  % (q, s["state"], s["cpu"], cmdline(q)[:70]))
    print("These pids are the ones the job recorded, so a namespace")
    print("mix-up is ruled out (`docker top` prints HOST pids, which")
    print("signal nothing from inside). Surviving SIGKILL therefore")
    print("means wedged in the GPU driver; release the device with:")
    print("   docker restart server-local-bash-1")
    return 1


def cmd_clean(root, procs):
    recs = records(root)
    known = set()
    for r in recs:
        known |= set(tree(r.get("pid", 0), procs))
    zombies = [p for p in procs if p["state"] == "Z"]
    ours = [p for p in procs
            if ("make bg" in p["cmd"] or "learn_device" in p["cmd"]
                or "perfmon.py" in p["cmd"] or "tools/know.py" in p["cmd"])
            and p["pid"] not in known]
    print("recorded jobs: %d" % len(recs))
    print("zombies: %d" % len(zombies))
    for p in zombies:
        print("   pid %d %s (parent %d)" % (p["pid"], p["name"], p["ppid"]))
    print("unmanaged, matching our workloads: %d" % len(ours))
    for p in ours:
        print("   pid %d state %s cpu %.1fs  %s"
              % (p["pid"], p["state"], p["cpu"], p["cmd"][:70]))
    dropped = 0
    for r in recs:
        if not alive(r.get("pid", 0)) and \
                log_state(r.get("log")).startswith("exit="):
            os.remove(os.path.join(root, "jobs", r["job"] + ".json"))
            dropped += 1
    print("dropped %d finished record(s)" % dropped)
    print("zombies belong to their parent: they clear when it exits or "
          "the container restarts.")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    root, what, rest = argv[0], argv[1], argv[2:]
    procs = all_procs()
    if what == "list":
        return cmd_list(root, procs)
    if what == "stop" and rest:
        return cmd_stop(root, procs, rest[0])
    if what == "clean":
        return cmd_clean(root, procs)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
