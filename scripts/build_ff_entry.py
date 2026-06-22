"""Build the webpage FF[scene] entry (force-full curve + time-to-target targets)
from a *_time_to_target_v8_forcefull report dir. Prints compact JSON.

Usage: python scripts/build_ff_entry.py <scene> <report_dir>
"""
import csv, json, sys
from pathlib import Path


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(scene, report_dir):
    rd = Path(report_dir)
    # --- curve from sampled_test_curve.csv ---
    curve = {"baseline": [], "agentic": []}
    for r in csv.DictReader(open(rd / "sampled_test_curve.csv")):
        m = r["method"]
        if m not in curve:
            continue
        curve[m].append({
            "it": int(float(r["iteration"])),
            "t": round(float(r["train_wall_seconds"]), 1),
            "psnr": round(float(r["test_psnr"]), 3),
            "N": int(float(r["final_gaussians"])),
        })
    for m in curve:
        curve[m].sort(key=lambda p: p["it"])

    # --- targets from time_to_target_comparison.csv ---
    targets = []
    for r in csv.DictReader(open(rd / "time_to_target_comparison.csv")):
        targets.append({
            "target": round(float(r["target_psnr"]), 1),
            "base": num(r["baseline_time_seconds"]),
            "agent": num(r["agentic_time_seconds"]),
            "speedup": num(r["speedup_x"]),
            "ready": r["claim_ready"].strip(),
        })
    targets.sort(key=lambda d: d["target"])

    print(json.dumps({scene: {"curve": curve, "targets": targets}}, separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
