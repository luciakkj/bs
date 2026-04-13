import argparse
from pathlib import Path

from app.pipeline import VideoAnalyticsPipeline
from config import get_config


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}


def _resolve_config_path() -> str:
    parser = argparse.ArgumentParser(description="Run the video analytics pipeline.")
    parser.add_argument(
        "--mode",
        choices=("anomaly", "tracking", "compromise"),
        default="anomaly",
        help="Select the default runtime mode.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional config file path. Overrides --mode when provided.",
    )
    args = parser.parse_args()
    if args.config_path:
        config_path = Path(args.config_path)
    elif args.mode == "tracking":
        config_path = Path("config_tracking.yaml")
    elif args.mode == "compromise":
        config_path = Path("config_compromise.yaml")
    else:
        config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return str(config_path)


def _ask_input_source() -> tuple[str | None, str | None]:
    print("Enter the input path for this run.")
    print("Press Enter to use the default source from the selected config.")
    print("Enter a video file path to run on a video.")
    print("Enter an image sequence folder or MOT sequence root to run on jpg frames.")
    user_input = input("Input path: ").strip()

    if not user_input:
        return None, None

    path = Path(user_input)
    if not path.exists():
        print(f"Path does not exist. Continue with config source: {path}")
        return None, None

    if path.is_dir():
        candidate_dirs = [path]
        img1_dir = path / "img1"
        if img1_dir.is_dir():
            candidate_dirs.insert(0, img1_dir)
        seq_path = next((directory / "%06d.jpg" for directory in candidate_dirs if next(directory.glob("*.jpg"), None)), None)
        if seq_path is None:
            print(f"No jpg image sequence found. Continue with config source: {path}")
            return None, None
        return "image_sequence", str(seq_path.resolve())

    if path.is_file():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return "video", str(path.resolve())
        print(f"File is not a supported video. Continue with config source: {path}")
        return None, None

    return None, None


def main() -> None:
    config_path = _resolve_config_path()
    print(f"Using config: {Path(config_path).resolve()}")
    cfg = get_config(config_path)

    source_type, source_path = _ask_input_source()
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
