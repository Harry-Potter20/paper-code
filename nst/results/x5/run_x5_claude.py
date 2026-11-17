"""NST R3-X5: Claude repeats, per-case and unbatched (NST_R3_ROBUSTNESS_PREREG_2026-11-10.md). One fresh `claude -p`
process per prompt, run from an empty directory with a replacement system prompt, no tools, no settings, no MCP, no
session persistence. T1 prompt (the paper's), the paper's 2 settings, 26 validated syndromes, 3 repeats.
Writes raw outputs incrementally to x5_claude_raw.jsonl (resumable)."""
import json, os, re, subprocess, sys, time
B = os.path.dirname(os.path.abspath(__file__)); EMPTY = sys.argv[1]
SYN = json.load(open(f"{B}/../syndromes_26.json"))["syndromes"]
T1 = ('You are assisting a clinician. Give your ranked differential diagnosis, MOST likely first.\n'
      'Respond with ONLY JSON: {{"ranked_differential": ["<most likely>", "...", "<least likely>"]}}\n'
      'Setting: {region}\nCase: {presentation}.\nJSON:')
OUT = f"{B}/x5_claude_raw.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); done.add((d["rep"], d["side"], d["syndrome"]))
CMD = ["claude", "-p", "--model", "sonnet", "--system-prompt", "You are a helpful assistant.", "--tools", "",
       "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
for rep in range(3):
    for s in SYN:
        for side, region in (("low", s["endemic_region"]), ("temp", s["temperate_region"])):
            if (rep, side, s["name"]) in done:
                continue
            p = T1.format(region=region, presentation=s["presentation"])
            for attempt in range(4):
                try:
                    r = subprocess.run(CMD + [p], cwd=EMPTY, capture_output=True, text=True, timeout=240)
                    d = json.loads(r.stdout); break
                except Exception as e:
                    d = {"is_error": True, "result": f"{type(e).__name__}"}; time.sleep(30 * (attempt + 1))
            rec = {"rep": rep, "side": side, "syndrome": s["name"], "text": d.get("result"), "is_error": d.get("is_error"),
                   "models": list((d.get("modelUsage") or {}).keys())}
            with open(OUT, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(rep, side, s["name"][:40], "err" if rec["is_error"] else "ok", flush=True)
print("X5_DONE", flush=True)
