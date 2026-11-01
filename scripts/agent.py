#!/usr/bin/env python3
"""Natural-language task runner: one sentence in, verified artifact out.

Usage:
  agent "写一个贪吃蛇游戏"                 # single task
  agent --batch wishes.txt                 # one wish per line, auto fleet
  echo "make a landing page for coffee" | agent   # stdin

Each wish gets a fresh workspace (clean inventory), runs through the
controller with plain-answer + native write prefill, and is verified
client-side (structure gate) with one automatic retry.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("RWKV_AGENT_URL", "http://127.0.0.1:8122")
ROOT = os.environ.get("RWKV_AGENT_WORKSPACES", os.path.expanduser("~/rwkv-4090d/workspaces"))
FLEET_ADMIT = int(os.environ.get("RWKV_AGENT_FLEET", "16"))

FILE_RE = re.compile(r"(?:write|创建|写|生成|make|create)\s+[`\"']?([\w./-]+\.(?:html|py|js|css|md|txt|json|ts|sh|yaml))", re.I)


def derive_objective(wish: str):
    """Extract a target filename if the wish names one; keep the wish as objective."""
    hit = FILE_RE.search(wish)
    target = hit.group(1) if hit else None
    objective = wish.strip().rstrip("。.")
    if not target:
        slug = hashlib.sha1(wish.encode()).hexdigest()[:6]
        target = f"out-{slug}.html" if re.search(r"网页|页面|游戏|page|game|site|html", wish, re.I) else f"out-{slug}.md"
        objective = f"Write {target}: {wish.strip()}"
    return objective, target


def artifact_ok(path: str) -> bool:
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    if len(src) < 200:
        return False
    if path.endswith((".html", ".htm")):
        if not src.rstrip().endswith("</html>"):
            return False
        if len(re.findall(r"<script\b", src)) > len(re.findall(r"</script>", src)):
            return False
    return True


async def submit(session: str, objective: str, wd: str, sem, results: dict, key: str):
    body = json.dumps({"session_id": session, "task_spec": {
        "objective": objective,
        "acceptance_criteria": ["the requested artifact exists and is complete"],
        "working_directory": wd, "requires_mutation": True}})
    loop = asyncio.get_event_loop()

    async def once():
        req = urllib.request.Request(BASE + "/v1/tasks", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(400):
            try:
                resp = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=1800))
                return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 399:
                    await asyncio.sleep(6)
                    continue
                raise
    async with sem:
        t0 = time.time()
        try:
            await once()
        except Exception as e:
            results[key] = (False, f"submit: {e}", round(time.time() - t0))
            return
        results[key] = (None, "ok", round(time.time() - t0))


async def run_wishes(wishes):
    jobs = []
    for idx, wish in enumerate(wishes):
        objective, target = derive_objective(wish)
        wd = f"nl-{hashlib.sha1((wish + str(idx)).encode()).hexdigest()[:10]}"
        os.makedirs(f"{ROOT}/{wd}", exist_ok=True)
        jobs.append((idx, wish, objective, target, wd, f"nl-{int(time.time())}-{idx}"))

    results = {}
    sem = asyncio.Semaphore(FLEET_ADMIT)
    await asyncio.gather(*(submit(j[5], j[2], j[3] if False else j[4], sem, results, j[0]) for j in jobs))

    # verify + variant retries: same wish, different phrasing beats
    # deterministic first-turn derails; compact hint beats token-ceiling truncation.
    def variant(objective: str, attempt: int) -> str:
        if attempt == 1:
            return objective + " Keep the file compact and make sure it finishes completely."
        return objective + " Output a complete, minimal, fully working version."
    for attempt in (1, 2):
        retry_jobs = []
        for idx, wish, objective, target, wd, session in jobs:
            ok = artifact_ok(f"{ROOT}/{wd}/{target}")
            if not ok:
                retry_jobs.append((idx, wish, variant(objective, attempt), target, wd, f"{session}-v{attempt}"))
        if not retry_jobs:
            break
        await asyncio.gather(*(submit(j[5], j[2], j[4], sem, results, j[0]) for j in retry_jobs))

    outcomes = []
    for idx, wish, objective, target, wd, session in jobs:
        path = f"{ROOT}/{wd}/{target}"
        good = artifact_ok(path)
        outcomes.append({"wish": wish[:60], "target": target, "ok": good,
                         "path": path if good else None,
                         "secs": results.get(idx, (None, None, 0))[2]})
    return outcomes


def main():
    ap = argparse.ArgumentParser(description="natural-language task runner")
    ap.add_argument("wish", nargs="*", help="natural language task")
    ap.add_argument("--batch", help="file with one wish per line")
    args = ap.parse_args()
    if args.batch:
        wishes = [l.strip() for l in open(args.batch, encoding="utf-8") if l.strip() and not l.startswith("#")]
    elif args.wish:
        wishes = [" ".join(args.wish)]
    elif not sys.stdin.isatty():
        wishes = [l.strip() for l in sys.stdin if l.strip()]
    else:
        ap.error("give a wish, --batch file, or stdin")

    t0 = time.time()
    outcomes = asyncio.run(run_wishes(wishes))
    ok = sum(1 for o in outcomes if o["ok"])
    for o in outcomes:
        mark = "✅" if o["ok"] else "❌"
        print(f"{mark} {o['wish']:<50} {o['secs']:>4}s  {o['path'] or o['target']}")
    print(f"\n{ok}/{len(outcomes)} done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
