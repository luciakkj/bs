from __future__ import annotations

import json
import random
from pathlib import Path

from training.behavior_dataset_expansion import _extract_window, _load_jsonl, _write_jsonl
from training.config import AvenueBehaviorWindowConfig


def _window_reason(parts: list[str]) -> str:
    return " | ".join(parts)


def _label_window(sample: dict[str, object], cfg: AvenueBehaviorWindowConfig) -> tuple[str, list[str]]:
    features = sample["features"]
    frame_count = int(sample["frame_count"])
    support_frames = int(features.get("support_frames", 0))
    support_ratio = float(features.get("support_ratio", 0.0))
    avg_speed = float(features.get("avg_speed", 0.0))
    p90_speed = float(features.get("p90_speed", 0.0))
    high_speed_ratio = float(features.get("high_speed_ratio", 0.0))
    movement_extent = float(features.get("movement_extent", 0.0))
    centroid_radius = float(features.get("centroid_radius", 0.0))
    stationary_ratio = float(features.get("stationary_ratio", 0.0))
    revisit_ratio = float(features.get("revisit_ratio", 0.0))
    straightness = float(features.get("straightness", 1.0))

    if (
        frame_count >= cfg.running_window
        and support_frames >= cfg.min_running_support_frames
        and support_ratio >= cfg.min_running_support_ratio
        and high_speed_ratio >= cfg.min_running_high_speed_ratio
        and avg_speed >= cfg.min_running_avg_speed
        and p90_speed >= cfg.min_running_p90_speed
        and movement_extent >= cfg.min_running_movement_extent
    ):
        return "running", [
            f"support_ratio={support_ratio:.3f}",
            f"high_speed_ratio={high_speed_ratio:.3f}",
            f"avg_speed={avg_speed:.3f}",
            f"p90_speed={p90_speed:.3f}",
            f"movement_extent={movement_extent:.3f}",
        ]

    if (
        frame_count >= cfg.loiter_window
        and support_frames >= cfg.min_loiter_support_frames
        and support_ratio >= cfg.min_loiter_support_ratio
        and avg_speed <= 2.2
        and movement_extent <= 55.0
        and centroid_radius <= 28.0
        and stationary_ratio >= cfg.min_loiter_stationary_ratio
        and revisit_ratio >= cfg.min_loiter_revisit_ratio
        and straightness <= cfg.max_loiter_straightness
    ):
        return "loitering", [
            f"support_ratio={support_ratio:.3f}",
            f"avg_speed={avg_speed:.3f}",
            f"movement_extent={movement_extent:.3f}",
            f"straightness={straightness:.3f}",
            f"revisit_ratio={revisit_ratio:.3f}",
        ]

    if (
        frame_count >= cfg.normal_window
        and support_frames <= cfg.max_normal_support_frames
        and support_ratio <= cfg.max_normal_support_ratio
    ):
        return "normal", [
            f"support_frames={support_frames}",
            f"support_ratio={support_ratio:.3f}",
            f"avg_speed={avg_speed:.3f}",
            f"movement_extent={movement_extent:.3f}",
        ]

    return "unknown", []


def _normal_hardness(sample: dict[str, object]) -> float:
    features = sample["features"]
    avg_speed = float(features.get("avg_speed", 0.0))
    movement_extent = float(features.get("movement_extent", 0.0))
    centroid_radius = float(features.get("centroid_radius", 0.0))
    straightness = float(features.get("straightness", 1.0))
    return (
        max(0.0, 2.2 - avg_speed) * 2.0
        + max(0.0, 55.0 - movement_extent) * 0.05
        + max(0.0, 28.0 - centroid_radius) * 0.05
        + max(0.0, 0.8 - straightness)
    )


def _dedupe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    deduped = []
    for row in rows:
        key = row["sample_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def build_avenue_behavior_windows(config: AvenueBehaviorWindowConfig) -> dict[str, object]:
    cfg = config.resolved()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _load_jsonl(cfg.input_path)
    random.seed(cfg.seed)

    positive_rows: list[dict[str, object]] = []
    normal_candidates_by_track: dict[str, list[dict[str, object]]] = {}
    skipped_tracks = 0

    for row in base_rows:
        frame_count = int(row.get("frame_count", 0))
        if frame_count < cfg.min_track_frames:
            skipped_tracks += 1
            continue

        source_track_id = str(row.get("source_track_id", row["sample_id"]))
        starts_by_mode = {
            "running": (cfg.running_window, cfg.running_stride),
            "loitering": (cfg.loiter_window, cfg.loiter_stride),
            "normal": (cfg.normal_window, cfg.normal_stride),
        }

        for mode_name, (window_size, stride) in starts_by_mode.items():
            if frame_count < window_size:
                continue
            last_start = max(0, frame_count - window_size)
            starts = list(range(0, last_start + 1, max(1, stride)))
            if starts and starts[-1] != last_start:
                starts.append(last_start)
            elif not starts:
                starts = [0]

            for start_idx in starts:
                end_idx = start_idx + window_size
                if end_idx > frame_count:
                    continue

                window_row = _extract_window(row, start_idx, end_idx)
                window_row["source_track_id"] = source_track_id
                label, reasons = _label_window(window_row, cfg)
                if label == "unknown":
                    continue

                window_row["primary_label"] = label
                window_row["pseudo_labels"] = [label]
                window_row["label_reasons"] = [_window_reason(reasons)]

                if label == "normal":
                    normal_candidates_by_track.setdefault(source_track_id, []).append(window_row)
                else:
                    positive_rows.append(window_row)

    positive_rows = _dedupe_rows(positive_rows)
    positive_count = len(positive_rows)
    max_normal_count = int(round(positive_count * max(0.0, cfg.normal_to_positive_ratio)))

    selected_normal_rows: list[dict[str, object]] = []
    for source_track_id, rows in normal_candidates_by_track.items():
        rows = _dedupe_rows(rows)
        rows.sort(key=_normal_hardness, reverse=True)
        selected_normal_rows.extend(rows[: cfg.max_normal_windows_per_track])

    selected_normal_rows = _dedupe_rows(selected_normal_rows)
    selected_normal_rows.sort(key=_normal_hardness, reverse=True)
    if max_normal_count > 0:
        selected_normal_rows = selected_normal_rows[:max_normal_count]
    else:
        selected_normal_rows = []

    final_rows = _dedupe_rows(positive_rows + selected_normal_rows)
    random.shuffle(final_rows)

    summary = {
        "input_path": str(cfg.input_path),
        "output_dir": str(cfg.output_dir),
        "total_input_tracks": len(base_rows),
        "skipped_tracks": skipped_tracks,
        "total_output_samples": len(final_rows),
        "label_counts": {
            "running": sum(1 for row in final_rows if row["primary_label"] == "running"),
            "loitering": sum(1 for row in final_rows if row["primary_label"] == "loitering"),
            "normal": sum(1 for row in final_rows if row["primary_label"] == "normal"),
        },
        "config": {
            "loiter_window": cfg.loiter_window,
            "running_window": cfg.running_window,
            "normal_window": cfg.normal_window,
            "loiter_stride": cfg.loiter_stride,
            "running_stride": cfg.running_stride,
            "normal_stride": cfg.normal_stride,
            "min_loiter_support_frames": cfg.min_loiter_support_frames,
            "min_loiter_support_ratio": cfg.min_loiter_support_ratio,
            "min_loiter_stationary_ratio": cfg.min_loiter_stationary_ratio,
            "min_loiter_revisit_ratio": cfg.min_loiter_revisit_ratio,
            "max_loiter_straightness": cfg.max_loiter_straightness,
            "min_running_support_frames": cfg.min_running_support_frames,
            "min_running_support_ratio": cfg.min_running_support_ratio,
            "min_running_high_speed_ratio": cfg.min_running_high_speed_ratio,
            "min_running_avg_speed": cfg.min_running_avg_speed,
            "min_running_p90_speed": cfg.min_running_p90_speed,
            "min_running_movement_extent": cfg.min_running_movement_extent,
            "max_normal_support_frames": cfg.max_normal_support_frames,
            "max_normal_support_ratio": cfg.max_normal_support_ratio,
            "max_normal_windows_per_track": cfg.max_normal_windows_per_track,
            "normal_to_positive_ratio": cfg.normal_to_positive_ratio,
            "seed": cfg.seed,
        },
    }

    tracks_path = cfg.output_dir / "tracks_reconstructed.jsonl"
    _write_jsonl(tracks_path, final_rows)
    for label in ("running", "loitering", "normal"):
        _write_jsonl(
            cfg.output_dir / f"{label}_tracks.jsonl",
            [row for row in final_rows if row["primary_label"] == label],
        )
    (cfg.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
