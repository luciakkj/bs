from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a held-out external behavior test set from loitering and running sources."
    )
    parser.add_argument("--loitering-dataset", required=True, help="Path to RS-WACV24 test tracks.jsonl")
    parser.add_argument("--running-dataset", required=True, help="Path to external running val tracks.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = _parse_args()
    loitering_path = Path(args.loitering_dataset).resolve()
    running_path = Path(args.running_dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loitering_rows = _load_jsonl(loitering_path)
    running_rows = _load_jsonl(running_path)
    merged_rows = list(loitering_rows) + list(running_rows)

    tracks_path = output_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "loitering_dataset": str(loitering_path),
        "running_dataset": str(running_path),
        "output_dir": str(output_dir),
        "sample_count": len(merged_rows),
        "label_counts": dict(Counter(str(row.get("primary_label", "")) for row in merged_rows)),
        "source_counts": {
            "loitering_rows": len(loitering_rows),
            "running_rows": len(running_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
