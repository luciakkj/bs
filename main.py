from pathlib import Path

from app.pipeline import VideoAnalyticsPipeline
from config import get_config


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def _resolve_image_sequence_pattern(path: Path) -> Path | None:
    """Resolve a user-provided directory to an OpenCV image sequence pattern."""
    candidate_dirs = [path]

    img1_dir = path / "img1"
    if img1_dir.is_dir():
        candidate_dirs.insert(0, img1_dir)

    for directory in candidate_dirs:
        jpg_files = sorted(directory.glob("*.jpg"))
        if jpg_files:
            return directory / "%06d.jpg"

    return None


def ask_input_source():
    print("请输入本次要测试的输入路径。")
    print("直接回车：使用 config.yaml 里的默认输入源")
    print("输入视频文件路径：本次运行临时使用该视频")
    print("输入图片序列文件夹路径（如 MOT17 的 img1 或序列根目录）：本次运行临时使用该图片序列")
    user_input = input("输入路径: ").strip()

    if not user_input:
        return None, None

    path = Path(user_input)
    if not path.exists():
        print(f"路径不存在，将继续使用默认输入源: {path}")
        return None, None

    if path.is_dir():
        seq_path = _resolve_image_sequence_pattern(path)
        if seq_path is None:
            print(f"目录中未找到可用的 jpg 图片序列，将继续使用默认输入源: {path}")
            return None, None
        return "image_sequence", str(seq_path.resolve())

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video", str(path.resolve())
        print(f"该文件不是常见视频格式，将继续使用默认输入源: {path}")
        return None, None

    return None, None


def main():
    cfg = get_config("config.yaml")

    source_type, source_path = ask_input_source()

    if source_type == "video":
        cfg.source.use_camera = False
        cfg.source.video_path = source_path
        cfg.source.mot17_seq = ""
    elif source_type == "image_sequence":
        cfg.source.use_camera = False
        cfg.source.video_path = ""
        cfg.source.mot17_seq = source_path

    app = VideoAnalyticsPipeline(cfg)
    app.run()


if __name__ == "__main__":
    main()
