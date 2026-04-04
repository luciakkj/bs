from __future__ import annotations

import configparser
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from training.config import PrepareConfig


@dataclass(slots=True)
class MOTSequence:
    name: str
    base_name: str
    detector_name: str
    root: Path
    image_dir: Path
    gt_path: Path
    seqinfo_path: Path
    frame_rate: int
    seq_length: int
    image_width: int
    image_height: int
    image_ext: str


def read_seqinfo(seqinfo_path: Path) -> MOTSequence:
    parser = configparser.ConfigParser()
    parser.read(seqinfo_path, encoding="utf-8")
    seq = parser["Sequence"]
    name = seq.get("name", seqinfo_path.parent.name)
    base_name, detector_name = split_sequence_name(name)
    root = seqinfo_path.parent
    return MOTSequence(
        name=name,
        base_name=base_name,
        detector_name=detector_name,
        root=root,
        image_dir=root / seq.get("imDir", "img1"),
        gt_path=root / "gt" / "gt.txt",
        seqinfo_path=seqinfo_path,
        frame_rate=seq.getint("frameRate", fallback=30),
        seq_length=seq.getint("seqLength", fallback=0),
        image_width=seq.getint("imWidth", fallback=1920),
        image_height=seq.getint("imHeight", fallback=1080),
        image_ext=seq.get("imExt", ".jpg"),
    )


def split_sequence_name(sequence_name: str) -> tuple[str, str]:
    parts = sequence_name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:2]), parts[2]
    return sequence_name, ""


def discover_sequences(mot_root: Path, split_name: str = "train", detector_filter: str | None = "FRCNN") -> list[MOTSequence]:
    split_root = mot_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"MOT split directory not found: {split_root}")

    sequences: list[MOTSequence] = []
    for seqinfo_path in sorted(split_root.glob("*/seqinfo.ini")):
        sequence = read_seqinfo(seqinfo_path)
        if detector_filter and sequence.detector_name.upper() != detector_filter.upper():
            continue
        if split_name.lower() == "train" and not sequence.gt_path.exists():
            continue
        sequences.append(sequence)

    if not sequences:
        raise FileNotFoundError(
            f"No sequences found under {split_root} with detector filter {detector_filter!r}."
        )
    return sequences


def load_gt_annotations(
    sequence: MOTSequence,
    include_classes: tuple[int, ...] = (1,),
    min_visibility: float = 0.25,
) -> dict[int, list[dict[str, float | int]]]:
    frames: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    with sequence.gt_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 9:
                continue

            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            left = float(row[2])
            top = float(row[3])
            width = float(row[4])
            height = float(row[5])
            mark = int(float(row[6]))
            class_id = int(float(row[7]))
            visibility = float(row[8])

            if mark != 1:
                continue
            if class_id not in include_classes:
                continue
            if visibility < min_visibility:
                continue
            if width <= 1 or height <= 1:
                continue

            frames[frame_id].append(
                {
                    "track_id": track_id,
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "visibility": visibility,
                    "class_id": class_id,
                }
            )
    return frames


class MOT17ToYOLOConverter:
    def __init__(self, config: PrepareConfig):
        self.config = config.resolved()

    def prepare(self) -> dict[str, object]:
        sequences = discover_sequences(
            self.config.mot_root,
            split_name=self.config.split_name,
            detector_filter=self.config.detector_filter,
        )
        split_map = self._build_split_map(sequences)

        if self.config.output_dir.exists() and self.config.overwrite:
            shutil.rmtree(self.config.output_dir)

        image_root = self.config.output_dir / "images"
        label_root = self.config.output_dir / "labels"
        split_root = self.config.output_dir / "splits"
        for split in ("train", "val"):
            (image_root / split).mkdir(parents=True, exist_ok=True)
            (label_root / split).mkdir(parents=True, exist_ok=True)
        split_root.mkdir(parents=True, exist_ok=True)

        image_lists = {"train": [], "val": []}
        stats = {
            "total_sequences": len(sequences),
            "train_sequences": 0,
            "val_sequences": 0,
            "total_images": 0,
            "total_labels": 0,
            "images_per_split": {"train": 0, "val": 0},
            "labels_per_split": {"train": 0, "val": 0},
        }

        for sequence in sequences:
            split = split_map[sequence.base_name]
            stats[f"{split}_sequences"] += 1
            frame_annotations = load_gt_annotations(
                sequence,
                include_classes=self.config.include_classes,
                min_visibility=self.config.min_visibility,
            )

            for image_path in sorted(sequence.image_dir.glob(f"*{sequence.image_ext}")):
                frame_id = int(image_path.stem)
                linked_image = image_root / split / f"{sequence.name}_{frame_id:06d}{image_path.suffix.lower()}"
                label_path = label_root / split / f"{sequence.name}_{frame_id:06d}.txt"

                self._link_or_copy(image_path, linked_image)
                labels = self._build_yolo_labels(sequence, frame_annotations.get(frame_id, []))
                label_path.write_text("\n".join(labels), encoding="utf-8")

                image_lists[split].append(str(linked_image.resolve()))
                stats["total_images"] += 1
                stats["images_per_split"][split] += 1
                if labels:
                    stats["total_labels"] += len(labels)
                    stats["labels_per_split"][split] += len(labels)

        train_txt = split_root / "train.txt"
        val_txt = split_root / "val.txt"
        train_txt.write_text("\n".join(image_lists["train"]), encoding="utf-8")
        val_txt.write_text("\n".join(image_lists["val"]), encoding="utf-8")

        data_yaml_path = self.config.output_dir / "mot17_person.yaml"
        data_yaml = {
            "path": str(self.config.output_dir.resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "names": {0: "person"},
            "nc": 1,
        }
        data_yaml_path.write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        metadata = {
            "config": {
                "mot_root": str(self.config.mot_root),
                "output_dir": str(self.config.output_dir),
                "split_name": self.config.split_name,
                "detector_filter": self.config.detector_filter,
                "include_classes": list(self.config.include_classes),
                "min_visibility": self.config.min_visibility,
                "val_ratio": self.config.val_ratio,
                "val_sequences": list(self.config.val_sequences),
            },
            "sequences": {
                sequence.name: {
                    "base_name": sequence.base_name,
                    "detector_name": sequence.detector_name,
                    "split": split_map[sequence.base_name],
                    "frame_rate": sequence.frame_rate,
                    "seq_length": sequence.seq_length,
                    "image_width": sequence.image_width,
                    "image_height": sequence.image_height,
                }
                for sequence in sequences
            },
            "stats": stats,
        }
        (self.config.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata

    def _build_split_map(self, sequences: list[MOTSequence]) -> dict[str, str]:
        base_names = sorted({sequence.base_name for sequence in sequences})
        if self.config.val_sequences:
            val_set = set(self.config.val_sequences)
        else:
            val_count = max(1, math.ceil(len(base_names) * self.config.val_ratio))
            val_set = set(base_names[-val_count:])

        split_map = {}
        for base_name in base_names:
            split_map[base_name] = "val" if base_name in val_set else "train"
        return split_map

    def _build_yolo_labels(
        self,
        sequence: MOTSequence,
        annotations: list[dict[str, float | int]],
    ) -> list[str]:
        labels: list[str] = []
        width = float(sequence.image_width)
        height = float(sequence.image_height)

        for ann in annotations:
            box_w = float(ann["width"])
            box_h = float(ann["height"])
            center_x = float(ann["left"]) + box_w / 2.0
            center_y = float(ann["top"]) + box_h / 2.0

            yolo_x = min(max(center_x / width, 0.0), 1.0)
            yolo_y = min(max(center_y / height, 0.0), 1.0)
            yolo_w = min(max(box_w / width, 1e-6), 1.0)
            yolo_h = min(max(box_h / height, 1e-6), 1.0)
            labels.append(f"0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")

        return labels

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

