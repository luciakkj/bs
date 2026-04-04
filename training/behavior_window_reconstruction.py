from __future__ import annotations

import json
from pathlib import Path
import random

from training.behavior_dataset_expansion import _extract_window, _load_jsonl, _write_jsonl
from training.config import resolve_path


def _window_sizes_for_label(label: str, frame_count: int) -> list[int]:
    if label == "running":
        candidates = [24, 30, 36, 48, 60, 72]
    elif label == "loitering":
        candidates = [48, 60, 72, 84, 96, 120]
    else:
        candidates = [24, 36, 48, 60, 72, 96]
    return sorted({size for size in candidates if 12 <= size <= frame_count})


def _should_keep_window(parent: dict[str, object], child: dict[str, object]) -> bool:
    label = str(parent["primary_label"])
    features = child["features"]
    frame_count = int(child["frame_count"])
    support_frames = int(features.get("support_frames", 0))
    avg_speed = float(features.get("avg_speed", 0.0))
    p90_speed = float(features.get("p90_speed", 0.0))
    movement_extent = float(features.get("movement_extent", 0.0))
    centroid_radius = float(features.get("centroid_radius", 0.0))

    if label == "running":
        return (
            frame_count >= 24
            and support_frames >= max(4, frame_count // 6)
            and (p90_speed >= 14.0 or avg_speed >= 7.5)
            and movement_extent >= 80.0
        )
    if label == "loitering":
        return (
            frame_count >= 48
            and support_frames >= max(8, frame_count // 4)
            and avg_speed <= 2.4
            and movement_extent <= 60.0
            and centroid_radius <= 32.0
        )
    return support_frames == 0 and frame_count >= 24


def reconstruct_behavior_windows(
    input_path: str | Path = "output/behavior_hard_negatives/subset20.jsonl",
    output_dir: str | Path = "output/behavior_reconstructed",
    *,
    max_normal_windows_per_track: int = 3,
    normal_to_positive_ratio: float = 2.0,
    seed: int = 42,
) -> dict[str, object]:
    input_path = resolve_path(input_path)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _load_jsonl(input_path)
    positive_rows: list[dict[str, object]] = []
    normal_rows: list[dict[str, object]] = []
    normal_windows_per_track: dict[str, int] = {}

    for row in base_rows:
        row = dict(row)
        label = str(row.get("primary_label", "unknown"))
        if label not in {"running", "loitering", "normal"}:
            continue

        row["source_track_id"] = row.get("source_track_id", row["sample_id"])
        if label == "normal":
            normal_rows.append(row)
        else:
            positive_rows.append(row)

        frame_count = int(row["frame_count"])
        window_sizes = _window_sizes_for_label(label, frame_count)
        for window_size in window_sizes:
            if frame_count < window_size:
                continue
            stride = max(4, window_size // (3 if label == "loitering" else 2))
            last_start = frame_count - window_size
            starts = list(range(0, last_start + 1, stride))
            if starts and starts[-1] != last_start:
                starts.append(last_start)
            elif not starts:
                starts = [0]

            for start_idx in starts:
                end_idx = start_idx + window_size
                child = _extract_window(row, start_idx, end_idx)
                child["source_track_id"] = row["source_track_id"]
                child["primary_label"] = label
                child["pseudo_labels"] = [label]
                child["label_reasons"] = [f"reconstructed_from_{label}_window"]
                if label == "normal":
                    hard_negative_score = float(row.get("features", {}).get("hard_negative_score", 0.0) or 0.0)
                    if hard_negative_score <= 0.0 and str(row["sample_id"]).startswith("hardneg_"):
                        hard_negative_score = 1.0
                    if hard_negative_score > 0.0:
                        child["features"]["hard_negative_score"] = hard_negative_score
                if not _should_keep_window(row, child):
                    continue

                if label == "normal":
                    track_key = str(row["source_track_id"])
                    current_count = normal_windows_per_track.get(track_key, 0)
                    if current_count >= max_normal_windows_per_track:
                        continue
                    normal_windows_per_track[track_key] = current_count + 1
                    normal_rows.append(child)
                else:
                    positive_rows.append(child)

    rng = random.Random(int(seed))
    positive_count = len(positive_rows)
    max_normal_count = max(positive_count, int(round(positive_count * float(normal_to_positive_ratio))))
    if len(normal_rows) > max_normal_count:
        hard_negative_rows = [row for row in normal_rows if float(row.get("features", {}).get("hard_negative_score", 0.0) or 0.0) > 0.0]
        other_normal_rows = [row for row in normal_rows if row not in hard_negative_rows]
        rng.shuffle(hard_negative_rows)
        rng.shuffle(other_normal_rows)

        selected_normal_rows = hard_negative_rows[:max_normal_count]
        if len(selected_normal_rows) < max_normal_count:
            remaining = max_normal_count - len(selected_normal_rows)
            selected_normal_rows.extend(other_normal_rows[:remaining])
        normal_rows = selected_normal_rows

    reconstructed_rows = [*positive_rows, *normal_rows]
    deduped_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in reconstructed_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen_ids:
            continue
        seen_ids.add(sample_id)
        deduped_rows.append(row)

    label_counts: dict[str, int] = {}
    for row in deduped_rows:
        label = str(row["primary_label"])
        label_counts[label] = label_counts.get(label, 0) + 1

    output_path = output_dir / "tracks_reconstructed.jsonl"
    _write_jsonl(output_path, deduped_rows)

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "summary": {
            "base_rows": len(base_rows),
            "reconstructed_rows": len(deduped_rows),
            "label_counts": label_counts,
            "positive_rows": positive_count,
            "normal_rows_after_balance": len(normal_rows),
            "max_normal_windows_per_track": int(max_normal_windows_per_track),
            "normal_to_positive_ratio": float(normal_to_positive_ratio),
        },
        "artifacts": {
            "tracks_reconstructed": str(output_path),
        },
        "notes": [
            "Windows are reconstructed with label-specific sizes to better align training with runtime sliding-window inference.",
            "Normal hard-negative rows preserve hard_negative_score so training can keep emphasizing false-positive corrections.",
            "Balanced reconstruction limits normal-window explosion with both per-track caps and a global normal-to-positive ratio.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
