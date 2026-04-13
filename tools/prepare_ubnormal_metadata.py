from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SPLIT_FILES = {
    ("train", "abnormal"): "abnormal_training_video_names.txt",
    ("train", "normal"): "normal_training_video_names.txt",
    ("val", "abnormal"): "abnormal_validation_video_names.txt",
    ("val", "normal"): "normal_validation_video_names.txt",
    ("test", "abnormal"): "abnormal_test_video_names.txt",
    ("test", "normal"): "normal_test_video_names.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build UBnormal manifests from the official split files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the extracted UBnormal dataset root.",
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        required=True,
        help="Path to the downloaded official UBnormal scripts folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where manifests and summary metadata will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they already exist.",
    )
    return parser.parse_args()


def load_split_names(scripts_dir: Path) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for key, filename in SPLIT_FILES.items():
        path = scripts_dir / filename
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result[key] = names
    return result


def index_videos(dataset_root: Path) -> dict[str, Path]:
    return {path.stem: path for path in dataset_root.rglob("*.mp4")}


def count_gt_masks(annotation_dir: Path) -> int:
    if not annotation_dir.exists():
        return 0
    return sum(1 for _ in annotation_dir.glob("*_gt.png"))


def build_record(video_name: str, split: str, label: str, video_path: Path) -> dict[str, object]:
    annotation_dir = video_path.with_name(f"{video_name}_annotations")
    track_path = annotation_dir / f"{video_name}_tracks.txt"
    scene = video_path.parent.name
    return {
        "video_name": video_name,
        "split": split,
        "label": label,
        "is_abnormal": label == "abnormal",
        "scene": scene,
        "video_path": str(video_path),
        "annotation_dir": str(annotation_dir) if annotation_dir.exists() else None,
        "track_path": str(track_path) if track_path.exists() else None,
        "gt_mask_count": count_gt_masks(annotation_dir),
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains files. Pass --overwrite to replace them."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = args.output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    split_names = load_split_names(args.scripts_dir)
    video_index = index_videos(args.dataset_root)

    all_records: list[dict[str, object]] = []
    split_counter: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    split_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for (split, label), names in split_names.items():
        for name in names:
            if name not in video_index:
                raise FileNotFoundError(f"Video listed in official split not found locally: {name}")
            record = build_record(name, split, label, video_index[name])
            all_records.append(record)
            split_rows[split].append(record)
            split_counter[split][label] += 1

    split_summaries = {
        split: {
            "videos": len(rows),
            "normal_videos": split_counter[split]["normal"],
            "abnormal_videos": split_counter[split]["abnormal"],
            "gt_masks": sum(int(row["gt_mask_count"]) for row in rows),
            "videos_with_tracks": sum(1 for row in rows if row["track_path"]),
        }
        for split, rows in split_rows.items()
    }

    metadata = {
        "dataset_name": "UBnormal",
        "dataset_root": str(args.dataset_root),
        "official_scripts_dir": str(args.scripts_dir),
        "videos_total": len(all_records),
        "split_summaries": split_summaries,
    }

    for split, rows in split_rows.items():
        write_jsonl(manifests_dir / f"{split}.jsonl", rows)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
