import cv2


def draw(frame, tracks, alarms, fps, roi, cross_line, track_infos):
    image = frame.copy()
    trajectory_length = 12
    trajectory_stride = 2

    if roi:
        x1, y1, x2, y2 = [int(v) for v in roi]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            image,
            "ROI",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if cross_line:
        p1 = tuple(int(v) for v in cross_line[0])
        p2 = tuple(int(v) for v in cross_line[1])
        cv2.line(image, p1, p2, (0, 165, 255), 2)
        cv2.putText(
            image,
            "LINE",
            (p1[0], max(p1[1] - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    alarm_map = {}
    for item in alarms:
        if len(item) >= 2:
            track_id, alarm_type = item[0], item[1]
            alarm_map[int(track_id)] = str(alarm_type)

    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        alarm_type = alarm_map.get(track_id)
        color = (0, 0, 255) if alarm_type else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        info = track_infos.get(track_id, {})
        speed = info.get("speed", 0.0)
        avg_speed = info.get("avg_speed", 0.0)
        inside_roi = info.get("inside_roi", False)

        label = f"ID {track_id}"
        if alarm_type:
            label += f" | {alarm_type}"
        if inside_roi:
            label += " | ROI"

        text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
        cv2.putText(
            image,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        trajectory = info.get("trajectory", [])
        if trajectory:
            pts = [(int(px), int(py)) for px, py in trajectory[-trajectory_length:]]
            for i in range(trajectory_stride, len(pts), trajectory_stride):
                cv2.line(image, pts[i - trajectory_stride], pts[i], (255, 0, 0), 1)

    lines = [
        f"FPS: {fps:.2f}",
        f"Targets: {len(tracks)}",
        f"Alarms: {len(alarms)}",
    ]

    x, y = 15, 30
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return image
