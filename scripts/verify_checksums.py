"""Checks the integrity of the released files.

1. Every pre-specified analysis plan still has the SHA256 that the
   corresponding test-set run recorded when it finished.
2. The frozen split file matches the hash in its freeze report.
3. Optional (--images): your local FGADR images are byte-identical to the
   images used in the paper.

Usage (from the repository root):
    python scripts/verify_checksums.py
    python scripts/verify_checksums.py --images      # needs config.yaml
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PLANS = REPO / "analysis_plans"

# (plan file, run-info file written after the test-set run, key holding the hash)
PLAN_RECORDS = [
    ("Exp13_Analysis_Plan.json", "Exp13A_run_info.json", "analysis_plan_sha256"),
    ("Exp13B_Probe_Plan.json", "Exp13B_run_info.json", "plan_sha256"),
    ("Exp13C_Attribution_Plan.json", "Exp13C_run_info.json", "plan_sha256"),
    ("Exp14_External_Plan.json", "Exp14_run_info.json", "plan_sha256"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="also hash the local FGADR images")
    args = ap.parse_args()
    failures = 0

    print("Pre-specified analysis plans")
    for plan, info, key in PLAN_RECORDS:
        recorded = json.loads((PLANS / info).read_text())[key]
        actual = sha256(PLANS / plan)
        ok = actual == recorded
        failures += not ok
        print(f"  {'OK ' if ok else 'BAD'} {plan:32s} {actual[:16]}  (recorded in {info})")

    print("\nOther protocol files (SHA256SUMS.txt)")
    for line in (PLANS / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        ok = sha256(PLANS / name) == expected
        failures += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name}")

    print("\nFrozen split")
    report = json.loads((REPO / "data_splits/FGADR_Split_V2_Freeze_Report.json").read_text())
    actual = sha256(REPO / "data_splits/FGADR_Split_V2_Frozen.csv")
    ok = actual == report["frozen_split_sha256"]
    failures += not ok
    print(f"  {'OK ' if ok else 'BAD'} FGADR_Split_V2_Frozen.csv {actual[:16]}")

    if args.images:
        sys.path.insert(0, str(REPO))
        from common.paths import load_index
        index = load_index()
        bad = [r.image_id for r in index.itertuples() if not Path(r.source_path).is_file() or sha256(r.source_path) != r.sha256]
        failures += len(bad)
        print(f"\nFGADR images: {len(index) - len(bad)}/{len(index)} match")
        if bad:
            print("  missing or different:", ", ".join(bad[:20]), "..." if len(bad) > 20 else "")

    print("\nALL CHECKS PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
