import json
import os


class RunMetaService:
    def __init__(self, output_dir="output", run_id="default"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, f"run_meta_{run_id}.json")

    def save(self, meta: dict):
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)