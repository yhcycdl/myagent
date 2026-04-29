from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.knowledge_base import parse_manual_files, write_jsonl


def main() -> None:
    settings = Settings()
    settings.ensure_data_dirs()

    manuals = []
    skipped = []
    for manual_path in sorted(settings.raw_manual_dir.glob("*.txt")):
        parsed = parse_manual_files(manual_path)
        if not parsed:
            skipped.append(manual_path.name)
            continue
        manuals.extend(parsed)

    write_jsonl(settings.parsed_manual_path, (asdict(item) for item in manuals))
    print(f"parsed manuals: {len(manuals)}")
    if skipped:
        print(f"skipped manuals: {', '.join(skipped)}")
    print(f"output: {settings.parsed_manual_path}")


if __name__ == "__main__":
    main()
