from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the current Avenue behavior dataset with external supplements.")
    parser.add_argument(
        "--base-dataset",
        default="",
    )
    parser.add_argument("--rswacv-dataset", required=True, help="Path to RS-WACV24 tracks.jsonl")
    parser.add_argument("--virat-dataset", required=True, help="Path to VIRAT running tracks.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-normal-cap", type=int, default=300)
    parser.add_argument("--external-loitering-cap", type=int, default=300)
    parser.add_argument("--external-running-cap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sample_rows(rows: list[dict[str, object]], cap: int, rng: random.Random) -> list[dict[str, object]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:cap]


def main() -> None:
    args = _parse_args()
    rng = random.Random(int(args.seed))

    base_rows = _load_jsonl(Path(args.base_dataset).resolve()) if str(args.base_dataset).strip() else []
    rswacv_rows = _load_jsonl(Path(args.rswacv_dataset).resolve())
    virat_rows = _load_jsonl(Path(args.virat_dataset).resolve())

    rswacv_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rswacv_rows:
        rswacv_by_label[str(row["primary_label"])].append(row)

    virat_by_label: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in virat_rows:
        virat_by_label[str(row["primary_label"])].append(row)

    supplemental_rows = []
    supplemental_rows.extend(_sample_rows(rswacv_by_label.get("normal", []), int(args.external_normal_cap), rng))
    supplemental_rows.extend(_sample_rows(rswacv_by_label.get("loitering", []), int(args.external_loitering_cap), rng))
    supplemental_rows.extend(_sample_rows(virat_by_label.get("running", []), int(args.external_running_cap), rng))

    merged_rows = list(base_rows) + supplemental_rows

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.jsonl"
    with tracks_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "base_dataset": str(Path(args.base_dataset).resolve()) if str(args.base_dataset).strip() else "",
        "rswacv_dataset": str(Path(args.rswacv_dataset).resolve()),
        "virat_dataset": str(Path(args.virat_dataset).resolve()),
        "output_dir": str(output_dir),
        "base_count": len(base_rows),
        "supplemental_count": len(supplemental_rows),
        "merged_count": len(merged_rows),
        "merged_label_counts": dict(Counter(str(row["primary_label"]) for row in merged_rows)),
        "base_label_counts": dict(Counter(str(row["primary_label"]) for row in base_rows)),
        "supplemental_label_counts": dict(Counter(str(row["primary_label"]) for row in supplemental_rows)),
        "caps": {
            "external_normal_cap": int(args.external_normal_cap),
            "external_loitering_cap": int(args.external_loitering_cap),
            "external_running_cap": int(args.external_running_cap),
        },
        "seed": int(args.seed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
