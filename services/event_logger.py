import json
import os


class EventLogger:
    def __init__(self, output_dir="output", run_id="default", file_prefix="events"):
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{file_prefix}_{run_id}.jsonl"
        self.output_path = os.path.join(output_dir, filename)

    def log(self, event: dict):
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")