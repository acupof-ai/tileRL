#!/usr/bin/env python3
"""Shared board for the tileRL sessions: topics anyone posts to, everyone reads.

    board.py who                      roster: name, ListAgents name, role, topics
    board.py topics                   one line per topic, newest first
    board.py show <topic>             that topic's rows, oldest first
    board.py post <topic> <kind> <text> [--artifact P]
    board.py open <topic> --owner W --question Q
    board.py feed [-n N]              everything, newest first
    board.py brief                    only what others posted since you last ran a command

kind: find | rule | block | done | ask | note.  find/rule/done need --artifact.
The board lives in the main checkout's runs/ (gitignored), so every worktree
sees one file. Every command first prints what is new to this session.
Set TILERL_SESSION=<roster name>; the branch name is the fallback identity.

Scope: live session state only -- who works on what, who holds a card, what
blocks whom. Conclusions and numbers go to CHANGELOG.md and docs/experience/;
a board row that carries one is a second source of truth and is wrong here.
Rows are append-only and timestamped; the newest row on a topic is its state,
and it is a claim -- nvidia-smi and git ls-tree stay the measurement.
"""

import argparse
import json
import os
import subprocess
import sys
import time

_common = subprocess.run(
    ["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True
).stdout.strip()
ROOT = os.path.dirname(os.path.abspath(_common))
RUNS = os.path.join(ROOT, "runs")
BOARD = os.path.join(RUNS, "board.jsonl")
ROSTER = os.path.join(RUNS, "roster.json")
SEEN = os.path.join(RUNS, ".board_seen.json")
KINDS = ("find", "rule", "block", "done", "ask", "note", "open")
NEEDS_ARTIFACT = {"find", "rule", "done"}


def rows():
    if not os.path.exists(BOARD):
        return []
    with open(BOARD, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def whoami():
    if os.environ.get("TILERL_SESSION"):
        return os.environ["TILERL_SESSION"]
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def append(row):
    os.makedirs(RUNS, exist_ok=True)
    with open(BOARD, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def now():
    return time.strftime("%Y-%m-%d %H:%MZ", time.gmtime())


def fmt(r, topic=True):
    art = f"  [{r['artifact']}]" if r.get("artifact") else ""
    t = f"{r['topic']:<16}" if topic else ""
    return f"{r['ts']}  {t}{r['who']:<10}{r['kind']:<6}{r['text']}{art}"


def brief(me):
    rs = rows()
    seen = load(SEEN) or {}
    fresh = [r for r in rs[seen.get(me, 0) :] if r["who"] != me]
    if fresh:
        print(f"--- {len(fresh)} new since {me} last looked ---")
        for r in fresh[-12:]:
            print(fmt(r))
        print("---")
    seen[me] = len(rs)
    os.makedirs(RUNS, exist_ok=True)
    with open(SEEN, "w", encoding="utf-8") as fh:
        json.dump(seen, fh)


def cmd_post(a):
    if a.kind not in KINDS:
        sys.exit(f"kind must be one of {', '.join(KINDS)}")
    if a.kind in NEEDS_ARTIFACT and not a.artifact:
        sys.exit(f"a '{a.kind}' needs --artifact: a claim nobody can check is chatter")
    known = {r["topic"] for r in rows()}
    if a.topic not in known:
        sys.exit(
            f"no topic '{a.topic}'. open it first; known: {', '.join(sorted(known)) or '(none)'}"
        )
    append(
        {
            "ts": now(),
            "who": whoami(),
            "topic": a.topic,
            "kind": a.kind,
            "text": a.text,
            "artifact": a.artifact,
        }
    )
    print(f"{a.topic} <- {a.kind} by {whoami()}")


def cmd_open(a):
    append(
        {
            "ts": now(),
            "who": whoami(),
            "topic": a.topic,
            "kind": "open",
            "text": a.question,
            "artifact": "",
            "owner": a.owner,
        }
    )
    print(f"opened {a.topic} (owner {a.owner})")


def cmd_who(a):
    r = load(ROSTER) or sys.exit(f"no {ROSTER}")
    print(f"{'name':<10}{'ListAgents':<20}{'role':<14}{'pair':<8}topics")
    for m in r["members"]:
        print(
            f"{m['name']:<10}{m['session']:<20}{m['role']:<14}{m.get('pair', '-'):<8}"
            f"{','.join(m['topics']) or '-'}"
        )
    for m in r.get("not_on_this_team", []):
        print(f"  not on team: {m['name']:<20}{m['why']}")


def cmd_topics(a):
    by = {}
    for r in rows():
        by.setdefault(r["topic"], []).append(r)
    for topic, rs in sorted(by.items(), key=lambda kv: kv[1][-1]["ts"], reverse=True):
        owner = next((r.get("owner") for r in rs if r.get("owner")), "?")
        blocked = sum(r["kind"] == "block" for r in rs) > sum(r["kind"] == "done" for r in rs)
        print(
            f"{topic:<20}{owner:<10}{len(rs):>3} rows  {'BLOCKED ' if blocked else ''}"
            f"{rs[-1]['ts']}  {rs[-1]['kind']}: {rs[-1]['text'][:70]}"
        )


def cmd_show(a):
    rs = [r for r in rows() if r["topic"] == a.topic]
    if not rs:
        sys.exit(f"no topic '{a.topic}'")
    for r in rs:
        print(fmt(r, topic=False))


def cmd_feed(a):
    for r in rows()[::-1][: a.n]:
        print(fmt(r))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("post")
    q.add_argument("topic")
    q.add_argument("kind")
    q.add_argument("text")
    q.add_argument("--artifact", default="")
    q.set_defaults(fn=cmd_post)
    q = sub.add_parser("open")
    q.add_argument("topic")
    q.add_argument("--owner", required=True)
    q.add_argument("--question", required=True)
    q.set_defaults(fn=cmd_open)
    sub.add_parser("who").set_defaults(fn=cmd_who)
    sub.add_parser("topics").set_defaults(fn=cmd_topics)
    q = sub.add_parser("show")
    q.add_argument("topic")
    q.set_defaults(fn=cmd_show)
    q = sub.add_parser("feed")
    q.add_argument("-n", type=int, default=40)
    q.set_defaults(fn=cmd_feed)
    sub.add_parser("brief").set_defaults(fn=lambda a: None)
    a = p.parse_args()
    brief(whoami())
    a.fn(a)


if __name__ == "__main__":
    main()
