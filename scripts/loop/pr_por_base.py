"""Keep only the PRs whose base branch is the one this run delivers into (stdin: gh pr list JSON; argv[1]: base).

The wrapper used to take the newest `loop` PR of the whole repo as its own delivery, so two runs on different
lots (VM and notebook) adopted each other's PR: wrong labels, wrong size check, wrong body in last-result.txt.
"""
import json
import sys

if __name__ == "__main__":
    base = sys.argv[1]
    print(json.dumps([p for p in json.load(sys.stdin) if p.get("baseRefName") == base]))
