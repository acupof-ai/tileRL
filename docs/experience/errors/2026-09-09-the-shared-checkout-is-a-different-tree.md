# The shared checkout is a different tree — 2026-09-09

## Context

Two sessions in one day answered questions about "what is in the tree" by
reading files in `/Users/bytedance/code/tileRL` and got wrong answers. One
grepped `allow_short_rollouts` in the shared checkout, got zero hits, and
nearly reported the flag did not exist; the checkout was parked at `a702c9a`,
dozens of commits behind `origin/main`, where the flag had not landed yet.
The other (mine) inventoried `scripts/bench_*.py` there, counted 17, and
reported the count as fact; `origin/main` has 32, and the 15 missing include
every script the current sm70 and SSD work runs.

## Root cause

The shared checkout is a working tree like any other: it stays at whatever
commit its last operator left it at, and nothing updates it. Reading a file
there answers "what was in the tree at `a702c9a`", not "what is in the
tree". The path is the same path everyone uses for the real tree, so the
answer *feels* current — the staleness is invisible until you compare.

## Fix

Before reading a file in the shared checkout to answer "what does the tree
have", run `git -C /Users/bytedance/code/tileRL log --oneline -1` and compare
against `origin/main`. Work that lands on a branch happens in a scratchpad
worktree off `origin/main`; the shared checkout is for running, not for
answering questions about the tree.

## Rule

在共享检出上读文件回答"树里有什么"之前,先 `git log --oneline -1` 对一下它在哪。
