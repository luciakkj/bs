from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DanceTrack as a YOLO person-detection dataset.")
    parser.add_argument("--train-dirs", nargs="+", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    return parser.parse_args()


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_gt(gt_path: Path, *, min_visibility: float) -> dict[int, list[tuple[float, float, float, float]]]:
    by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    with gt_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue
            frame_id = int(float(row[0]))
            left = float(row[2])
            top = float(row[3])
            width = float(row[4])
            height = float(row[5])
            mark = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])
            if mark != 1 or class_id != 1 or visibility < min_visibility:
                continue
            if width <= 1 or height <= 1:
                continue
            by_frame.setdefault(frame_id, []).append((left, top, width, height))
    return by_frame


def _write_label_file(label_path: Path, boxes: list[tuple[float, float, float, float]], *, width: int, height: int) -> int:
    rows: list[str] = []
    for left, top, box_w, box_h in boxes:
        center_x = left + box_w / 2.0
        center_y = top + box_h / 2.0
        yolo_x = min(max(center_x / width, 0.0), 1.0)
        yolo_y = min(max(center_y / height, 0.0), 1.0)
        yolo_w = min(max(box_w / width, 1e-6), 1.0)
        yolo_h = min(max(box_h / height, 1e-6), 1.0)
        rows.append(f"0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(rows), encoding="utf-8")
    return len(rows)


def _prepare_split(sequence_dirs: list[Path], split_name: str, image_root: Path, label_root: Path, *, min_visibility: float) -> tuple[list[str], dict[str, int]]:
    image_paths: list[str] = []
    stats = {"sequences": 0, "images": 0, "labels": 0}
    for seq_dir in sequence_dirs:
        seqinfo = {}
        with (seq_dir / "seqinfo.ini").open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if "=" in line and not line.startswith("["):
                    key, value = line.split("=", 1)
                    seqinfo[key.strip()] = value.strip()
        width = int(seqinfo.get("imWidth", "1920"))
        height = int(seqinfo.get("imHeight", "1080"))
        image_ext = seqinfo.get("imExt", ".jpg")
        gt_by_frame = _load_gt(seq_dir / "gt" / "gt.txt", min_visibility=min_visibility)
        stats["sequences"] += 1

        for image_path in sorted((seq_dir / "img1").glob(f"*{image_ext}")):
            frame_id = int(image_path.stem)
            dst_image = image_root / split_name / f"{seq_dir.name}_{image_path.name}"
            dst_label = label_root / split_name / f"{seq_dir.name}_{image_path.stem}.txt"
            _link_or_copy(image_path, dst_image)
            label_count = _write_label_file(dst_label, gt_by_frame.get(frame_id, []), width=width, height=height)
            image_paths.append(str(dst_image.resolve()))
            stats["images"] += 1
            stats["labels"] += label_count
    return image_paths, stats


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    image_root = output_dir / "images"
    label_root = output_dir / "labels"
    split_root = output_dir / "splits"
    for split in ("train", "val"):
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (label_root / split).mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    train_sequence_dirs: list[Path] = []
    for train_dir in args.train_dirs:
        train_sequence_dirs.extend(sorted(path for path in Path(train_dir).resolve().iterdir() if path.is_dir()))
    val_sequence_dirs = sorted(path for path in Path(args.val_dir).resolve().iterdir() if path.is_dir())

    train_images, train_stats = _prepare_split(
        train_sequence_dirs, "train", image_root, label_root, min_visibility=float(args.min_visibility)
    )
    val_images, val_stats = _prepare_split(
        val_sequence_dirs, "val", image_root, label_root, min_visibility=float(args.min_visibility)
    )

    train_txt = split_root / "train.txt"
    val_txt = split_root / "val.txt"
    train_txt.write_text("\n".join(train_images), encoding="utf-8")
    val_txt.write_text("\n".join(val_images), encoding="utf-8")

    yaml_path = output_dir / "dancetrack_person.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_dir),
                "train": str(train_txt.resolve()),
                "val": str(val_txt.resolve()),
                "names": {0: "person"},
                "nc": 1,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "output_dir": str(output_dir),
        "train_dirs": [str(Path(item).resolve()) for item in args.train_dirs],
        "val_dir": str(Path(args.val_dir).resolve()),
        "min_visibility": float(args.min_visibility),
        "train": train_stats,
        "val": val_stats,
        "yaml_path": str(yaml_path.resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
