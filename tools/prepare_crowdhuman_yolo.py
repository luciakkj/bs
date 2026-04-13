from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CrowdHuman to a YOLO person dataset.")
    parser.add_argument("--crowdhuman-root", default="data/external/crowdhuman")
    parser.add_argument("--output-dir", default="data/processed/crowdhuman_person_yolo")
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_odgt(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "hardlink":
        dst.hardlink_to(src)
    else:
        dst.write_bytes(src.read_bytes())


def _write_label_file(label_path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _convert_record(record: dict, image_path: Path, label_path: Path) -> dict[str, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image_h, image_w = image.shape[:2]
    yolo_boxes: list[tuple[float, float, float, float]] = []
    ignored = 0
    for item in record.get("gtboxes", []):
        if item.get("tag") != "person":
            ignored += 1
            continue
        extra = item.get("extra", {}) or {}
        head_attr = item.get("head_attr", {}) or {}
        if int(extra.get("ignore", 0) or 0) == 1 or int(head_attr.get("ignore", 0) or 0) == 1:
            ignored += 1
            continue

        x, y, w, h = (float(v) for v in item.get("fbox", [0, 0, 0, 0]))
        if w <= 1 or h <= 1:
            ignored += 1
            continue

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(image_w), x + w)
        y2 = min(float(image_h), y + h)
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 1 or clipped_h <= 1:
            ignored += 1
            continue

        cx = ((x1 + x2) / 2.0) / image_w
        cy = ((y1 + y2) / 2.0) / image_h
        nw = clipped_w / image_w
        nh = clipped_h / image_h
        yolo_boxes.append((cx, cy, nw, nh))

    _write_label_file(label_path, yolo_boxes)
    return {
        "kept_boxes": len(yolo_boxes),
        "ignored_boxes": ignored,
        "image_width": image_w,
        "image_height": image_h,
    }


def _prepare_split(
    records: list[dict],
    images_root: Path,
    output_dir: Path,
    split_name: str,
    link_mode: str,
) -> dict[str, int]:
    image_out = output_dir / "images" / split_name
    label_out = output_dir / "labels" / split_name
    split_txt = output_dir / "splits" / f"{split_name}.txt"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    split_txt.parent.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    image_count = 0
    kept_boxes = 0
    ignored_boxes = 0

    for record in records:
        image_name = f"{record['ID']}.jpg"
        src_image = images_root / image_name
        if not src_image.exists():
            raise FileNotFoundError(f"Image missing for record {record['ID']}: {src_image}")

        dst_image = image_out / image_name
        dst_label = label_out / f"{record['ID']}.txt"
        _link_or_copy(src_image, dst_image, link_mode)
        stats = _convert_record(record, src_image, dst_label)
        image_paths.append(str(dst_image.resolve()))
        image_count += 1
        kept_boxes += stats["kept_boxes"]
        ignored_boxes += stats["ignored_boxes"]

    split_txt.write_text("\n".join(image_paths), encoding="utf-8")
    return {
        "images": image_count,
        "kept_boxes": kept_boxes,
        "ignored_boxes": ignored_boxes,
    }


def main() -> None:
    args = _parse_args()
    crowdhuman_root = Path(args.crowdhuman_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_root = crowdhuman_root / "images" / "Images"
    train_ann = crowdhuman_root / "annotation_train.odgt"
    val_ann = crowdhuman_root / "annotation_val.odgt"

    if not images_root.exists():
        raise FileNotFoundError(f"CrowdHuman images directory not found: {images_root}")
    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError("CrowdHuman annotation files are missing.")

    _ensure_clean_dir(output_dir, args.overwrite)
    train_stats = _prepare_split(_load_odgt(train_ann), images_root, output_dir, "train", args.link_mode)
    val_stats = _prepare_split(_load_odgt(val_ann), images_root, output_dir, "val", args.link_mode)

    data_yaml = output_dir / "crowdhuman_person.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                f"train: {output_dir / 'splits' / 'train.txt'}",
                f"val: {output_dir / 'splits' / 'val.txt'}",
                "names:",
                "  0: person",
                "nc: 1",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "crowdhuman_root": str(crowdhuman_root),
        "output_dir": str(output_dir),
        "link_mode": args.link_mode,
        "train": train_stats,
        "val": val_stats,
        "data_yaml": str(data_yaml),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
