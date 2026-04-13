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
from statistics import median

import yaml
from PIL import Image

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
        dense_small_candidates: list[tuple[float, str, int]] = []
        dense_small_crop_candidates: list[dict[str, object]] = []
        stats = {
            "total_sequences": len(sequences),
            "train_sequences": 0,
            "val_sequences": 0,
            "total_images": 0,
            "total_labels": 0,
            "images_per_split": {"train": 0, "val": 0},
            "labels_per_split": {"train": 0, "val": 0},
            "dense_small_frames": 0,
            "dense_small_extra_repeats": 0,
            "dense_small_crop_frames": 0,
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
                candidate = self._dense_small_repeat_candidate(
                    sequence,
                    frame_annotations.get(frame_id, []),
                    split,
                    str(linked_image.resolve()),
                )
                if candidate is not None:
                    dense_small_candidates.append(candidate)
                crop_candidate = self._dense_small_crop_candidate(
                    sequence,
                    image_path,
                    frame_annotations.get(frame_id, []),
                    split,
                )
                if crop_candidate is not None:
                    dense_small_crop_candidates.append(crop_candidate)
                stats["total_images"] += 1
                stats["images_per_split"][split] += 1
                if labels:
                    stats["total_labels"] += len(labels)
                    stats["labels_per_split"][split] += len(labels)

        self._apply_dense_small_repeats(image_lists["train"], dense_small_candidates, stats)
        self._apply_dense_small_crops(
            image_lists["train"],
            image_root / "train",
            label_root / "train",
            dense_small_crop_candidates,
            stats,
        )

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
                "dense_small_repeat_factor": self.config.dense_small_repeat_factor,
                "dense_small_max_repeat_frames": self.config.dense_small_max_repeat_frames,
                "dense_small_min_gt_count": self.config.dense_small_min_gt_count,
                "dense_small_min_small_ratio": self.config.dense_small_min_small_ratio,
                "dense_small_max_median_area_ratio": self.config.dense_small_max_median_area_ratio,
                "small_box_area_ratio_thresh": self.config.small_box_area_ratio_thresh,
                "dense_small_crop_enable": self.config.dense_small_crop_enable,
                "dense_small_crop_max_frames": self.config.dense_small_crop_max_frames,
                "dense_small_crop_width_ratio": self.config.dense_small_crop_width_ratio,
                "dense_small_crop_height_ratio": self.config.dense_small_crop_height_ratio,
                "dense_small_crop_min_boxes": self.config.dense_small_crop_min_boxes,
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

    def _dense_small_repeat_candidate(
        self,
        sequence: MOTSequence,
        annotations: list[dict[str, float | int]],
        split: str,
        image_path: str,
    ) -> tuple[float, str, int] | None:
        if split != "train":
            return None
        if self.config.dense_small_repeat_factor <= 1:
            return None
        if len(annotations) < self.config.dense_small_min_gt_count:
            return None

        frame_area = float(sequence.image_width * sequence.image_height)
        area_ratios = [
            float(ann["width"]) * float(ann["height"]) / frame_area
            for ann in annotations
        ]
        if not area_ratios:
            return None

        small_count = sum(
            1 for area_ratio in area_ratios if area_ratio <= self.config.small_box_area_ratio_thresh
        )
        small_ratio = small_count / len(area_ratios)
        median_area_ratio = float(median(area_ratios))
        if small_ratio < self.config.dense_small_min_small_ratio:
            return None
        if median_area_ratio > self.config.dense_small_max_median_area_ratio:
            return None

        # Favor crowded frames with many very small boxes.
        score = (len(annotations) * small_ratio) / max(median_area_ratio, 1e-9)
        return (float(score), image_path, self.config.dense_small_repeat_factor - 1)

    def _dense_small_crop_candidate(
        self,
        sequence: MOTSequence,
        image_path: Path,
        annotations: list[dict[str, float | int]],
        split: str,
    ) -> dict[str, object] | None:
        if split != "train":
            return None
        if not self.config.dense_small_crop_enable:
            return None
        if len(annotations) < self.config.dense_small_min_gt_count:
            return None

        frame_area = float(sequence.image_width * sequence.image_height)
        area_ratios = [
            float(ann["width"]) * float(ann["height"]) / frame_area
            for ann in annotations
        ]
        if not area_ratios:
            return None

        small_annotations = [
            ann
            for ann, area_ratio in zip(annotations, area_ratios)
            if area_ratio <= self.config.small_box_area_ratio_thresh
        ]
        if not small_annotations:
            return None

        small_ratio = len(small_annotations) / len(annotations)
        median_area_ratio = float(median(area_ratios))
        if small_ratio < self.config.dense_small_min_small_ratio:
            return None
        if median_area_ratio > self.config.dense_small_max_median_area_ratio:
            return None

        score = (len(annotations) * small_ratio) / max(median_area_ratio, 1e-9)
        return {
            "score": float(score),
            "sequence": sequence,
            "image_path": image_path,
            "annotations": annotations,
            "small_annotations": small_annotations,
        }

    def _apply_dense_small_repeats(
        self,
        train_image_list: list[str],
        candidates: list[tuple[float, str, int]],
        stats: dict[str, object],
    ) -> None:
        if not candidates:
            return
        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.config.dense_small_max_repeat_frames is not None:
            candidates = candidates[: self.config.dense_small_max_repeat_frames]

        extra_repeats = 0
        for _, image_path, repeat_count in candidates:
            train_image_list.extend([image_path] * repeat_count)
            extra_repeats += repeat_count

        stats["dense_small_frames"] = len(candidates)
        stats["dense_small_extra_repeats"] = extra_repeats

    def _apply_dense_small_crops(
        self,
        train_image_list: list[str],
        image_dir: Path,
        label_dir: Path,
        candidates: list[dict[str, object]],
        stats: dict[str, object],
    ) -> None:
        if not candidates:
            return

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if self.config.dense_small_crop_max_frames is not None:
            candidates = candidates[: self.config.dense_small_crop_max_frames]

        crop_count = 0
        for candidate in candidates:
            crop_path = self._write_dense_small_crop(image_dir, label_dir, candidate)
            if crop_path is None:
                continue
            train_image_list.append(str(crop_path.resolve()))
            crop_count += 1

        stats["dense_small_crop_frames"] = crop_count

    def _write_dense_small_crop(
        self,
        image_dir: Path,
        label_dir: Path,
        candidate: dict[str, object],
    ) -> Path | None:
        sequence = candidate["sequence"]
        if not isinstance(sequence, MOTSequence):
            return None

        image_path = candidate["image_path"]
        annotations = candidate["annotations"]
        small_annotations = candidate["small_annotations"]
        if not isinstance(image_path, Path) or not isinstance(annotations, list) or not isinstance(small_annotations, list):
            return None

        crop_box = self._select_dense_small_crop_box(sequence, small_annotations)
        if crop_box is None:
            return None

        cropped_labels = self._build_crop_labels(sequence, annotations, crop_box)
        if len(cropped_labels) < self.config.dense_small_crop_min_boxes:
            return None

        x1, y1, x2, y2 = crop_box
        with Image.open(image_path) as image:
            cropped = image.crop((x1, y1, x2, y2))
            stem = f"{sequence.name}_{image_path.stem}_crop_{x1}_{y1}_{x2}_{y2}"
            image_output_path = image_dir / f"{stem}{image_path.suffix.lower()}"
            label_output_path = label_dir / f"{stem}.txt"
            cropped.save(image_output_path)
            label_output_path.write_text("\n".join(cropped_labels), encoding="utf-8")
        return image_output_path

    def _select_dense_small_crop_box(
        self,
        sequence: MOTSequence,
        small_annotations: list[dict[str, float | int]],
    ) -> tuple[int, int, int, int] | None:
        crop_w = max(64, int(round(sequence.image_width * self.config.dense_small_crop_width_ratio)))
        crop_h = max(64, int(round(sequence.image_height * self.config.dense_small_crop_height_ratio)))
        crop_w = min(crop_w, sequence.image_width)
        crop_h = min(crop_h, sequence.image_height)

        best_score = -1.0
        best_box: tuple[int, int, int, int] | None = None
        frame_area = float(sequence.image_width * sequence.image_height)
        for ann in small_annotations:
            center_x = float(ann["left"]) + float(ann["width"]) / 2.0
            center_y = float(ann["top"]) + float(ann["height"]) / 2.0
            x1 = int(round(center_x - crop_w / 2.0))
            y1 = int(round(center_y - crop_h / 2.0))
            x1 = min(max(0, x1), sequence.image_width - crop_w)
            y1 = min(max(0, y1), sequence.image_height - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            score = 0.0
            for candidate_ann in small_annotations:
                candidate_cx = float(candidate_ann["left"]) + float(candidate_ann["width"]) / 2.0
                candidate_cy = float(candidate_ann["top"]) + float(candidate_ann["height"]) / 2.0
                if x1 <= candidate_cx <= x2 and y1 <= candidate_cy <= y2:
                    area_ratio = (
                        float(candidate_ann["width"]) * float(candidate_ann["height"]) / frame_area
                    )
                    score += 1.0 / max(area_ratio, 1e-9)

            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)

        return best_box

    def _build_crop_labels(
        self,
        sequence: MOTSequence,
        annotations: list[dict[str, float | int]],
        crop_box: tuple[int, int, int, int],
    ) -> list[str]:
        x1, y1, x2, y2 = crop_box
        crop_w = float(x2 - x1)
        crop_h = float(y2 - y1)
        labels: list[str] = []

        for ann in annotations:
            box_x1 = float(ann["left"])
            box_y1 = float(ann["top"])
            box_x2 = box_x1 + float(ann["width"])
            box_y2 = box_y1 + float(ann["height"])
            center_x = (box_x1 + box_x2) / 2.0
            center_y = (box_y1 + box_y2) / 2.0
            if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
                continue

            clipped_x1 = max(box_x1, float(x1))
            clipped_y1 = max(box_y1, float(y1))
            clipped_x2 = min(box_x2, float(x2))
            clipped_y2 = min(box_y2, float(y2))
            clipped_w = clipped_x2 - clipped_x1
            clipped_h = clipped_y2 - clipped_y1
            if clipped_w <= 1 or clipped_h <= 1:
                continue

            yolo_x = ((clipped_x1 + clipped_x2) / 2.0 - x1) / crop_w
            yolo_y = ((clipped_y1 + clipped_y2) / 2.0 - y1) / crop_h
            yolo_w = clipped_w / crop_w
            yolo_h = clipped_h / crop_h
            labels.append(f"0 {yolo_x:.6f} {yolo_y:.6f} {yolo_w:.6f} {yolo_h:.6f}")

        return labels

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
