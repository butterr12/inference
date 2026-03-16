import requests
import time
from typing import Optional

import config


class HuggingFaceClient:
    def __init__(self):
        self.api_url = config.ZS_API
        self.headers = config.HEADERS
        self.timeout = 30

    def classify(self, text: str, candidate_labels: list) -> Optional[dict]:
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": candidate_labels, "multi_label": True}
        }

        start_time = time.time()
        try:
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
            inference_time = time.time() - start_time
            return {
                "data": resp.json(),
                "inference_time": inference_time,
                "error": None
            }
        except Exception as e:
            inference_time = time.time() - start_time
            return {
                "data": None,
                "inference_time": inference_time,
                "error": str(e)
            }
