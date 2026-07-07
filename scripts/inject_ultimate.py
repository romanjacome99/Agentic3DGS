"""Read the 4 ultimate-eval budget_sweep CSVs and inject them into
budget_conditioned.html as const DATA."""
import csv, json, re
from pathlib import Path

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
BASE = ROOT / "outputs/agentic_rl_budget/ultimate"
HTML = ROOT / "agentic_gs_phase1/docs/budget_conditioned.html"
COMBOS = {("fastergs", "agent"): "fastergs_agent", ("fastergs", "baseline"): "fastergs_baseline",
          ("3dgs", "agent"): "3dgs_agent", ("3dgs", "baseline"): "3dgs_baseline"}

data = {"fastergs": {}, "3dgs": {}}
actions = {"fastergs": {}, "3dgs": {}}
for (backend, method), d in COMBOS.items():
    csvp = BASE / d / "budget_sweep.csv"
    rows = []
    for r in csv.DictReader(open(csvp)):
        rows.append({
            "budget_s": float(r["budget_s"]),
            "final_iter": int(float(r["final_iter"])),
            "train_seconds": round(float(r["train_seconds"]), 1),
            "gaussians": int(float(r["gaussians"])),
            "test_psnr": round(float(r["test_psnr"]), 3),
            "test_ssim": round(float(r["test_ssim"]), 4),
            "densify_on": int(float(r["densify_on_blocks"])) if "densify_on_blocks" in r else int(float(r.get("densify_on", 0))),
            "self_stopped": r.get("self_stopped", ""),
            "blocks": int(float(r.get("blocks", 0))),
        })
    rows.sort(key=lambda x: x["budget_s"])
    data[backend][method] = rows
    if method == "agent":
        prof = BASE / d / "action_profiles.json"
        if prof.exists():
            actions[backend] = json.loads(prof.read_text())

import re as _re
blob = json.dumps(data, separators=(",", ":"))
ablob = json.dumps(actions, separators=(",", ":"))
html = HTML.read_text(encoding="utf-8")
# Idempotent: replace the placeholder OR a previously-injected blob.
html = _re.sub(r"const DATA = (?:__ULTIMATE_JSON__|\{.*?\});", "const DATA = " + blob + ";", html, count=1, flags=_re.S)
html = _re.sub(r"const ACTIONS = (?:__ACTIONS_JSON__|\{.*?\});", "const ACTIONS = " + ablob + ";", html, count=1, flags=_re.S)
HTML.write_text(html, encoding="utf-8")
print("injected DATA + ACTIONS; backends:", list(data),
      "| action budgets:", sorted(actions.get("fastergs", {}).keys(), key=lambda x: int(x)))
