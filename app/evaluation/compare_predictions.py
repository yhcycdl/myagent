from __future__ import annotations

import csv
import sys
from pathlib import Path


def load(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row["ret"] for row in csv.DictReader(handle)}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m app.evaluation.compare_predictions old.csv new.csv")
        return 2
    old = load(Path(sys.argv[1]))
    new = load(Path(sys.argv[2]))
    changed = [qid for qid, answer in new.items() if old.get(qid) != answer]
    print(f"changed={len(changed)}")
    for qid in changed[:80]:
        print(qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
