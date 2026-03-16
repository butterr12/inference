import json
from datetime import datetime

import config


class OutputHandler:
    def __init__(self, output_file: str = None):
        self.output_file = output_file or config.OUTPUT_FILE

    def append_run(self, results: list, total_duration: float, avg_inference_time: float):
        run_entry = {
            "run_timestamp": datetime.now().isoformat(),
            "total_duration_seconds": round(total_duration, 2),
            "inferences_count": len(results),
            "avg_inference_time_seconds": round(avg_inference_time, 2),
            "results": results
        }

        existing_data = self._load_existing()
        existing_data.append(run_entry)
        self._save(existing_data)

    def _load_existing(self) -> list:
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self, data: list):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_summary(self, total_valid: int, inferences_count: int, total_duration: float, avg_time: float):
        print(f"\n=== Inference Summary ===")
        print(f"Total valid entries in dataset: {total_valid}")
        print(f"Inferences made: {inferences_count}")
        print(f"Total run duration: {total_duration:.2f} seconds")
        print(f"Average time per inference: {avg_time:.2f} seconds")
