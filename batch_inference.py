import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import requests

import config
from models import InferenceResult


CACHE_FILE = "inference_cache.json"
PROCESSED_FILE = "processed_ids.json"


class Cache:
    def __init__(self):
        self._cache = {}
        self._processed_ids = set()
        self._load()

    def _load(self):
        cache_path = Path(CACHE_FILE)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._cache = {k: InferenceResult(**v) for k, v in data.items()}
            except (json.JSONDecodeError, ValueError):
                self._cache = {}

        processed_path = Path(PROCESSED_FILE)
        if processed_path.exists():
            try:
                with open(processed_path, "r", encoding="utf-8") as f:
                    self._processed_ids = set(json.load(f))
            except (json.JSONDecodeError, ValueError):
                self._processed_ids = set()

        # Sync: any cache key that's not in processed_ids should be added
        for review_id in self._cache.keys():
            if review_id not in self._processed_ids:
                self._processed_ids.add(review_id)

        # Save synced state
        if self._processed_ids:
            self.save_processed()

    def save_cache(self):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self._cache.items()}, f, indent=2)

    def save_processed(self):
        with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
            json.dump(list(self._processed_ids), f, indent=2)

    def is_cached(self, review_id: str) -> bool:
        return review_id in self._cache

    def is_processed(self, review_id: str) -> bool:
        return review_id in self._processed_ids

    def add_result(self, review_id: str, result: InferenceResult):
        self._cache[review_id] = result
        self._processed_ids.add(review_id)

    def get_cache_size(self):
        return len(self._cache)

    def get_processed_count(self):
        return len(self._processed_ids)


class ProgressTracker:
    def __init__(self, total: int):
        self._total = total
        self._processed = 0
        self._cached = 0
        self._errors = 0
        self._start_time = time.time()

    def increment_processed(self):
        self._processed += 1

    def increment_cached(self):
        self._cached += 1

    def increment_errors(self):
        self._errors += 1

    def get_stats(self):
        elapsed = time.time() - self._start_time
        rate = self._processed / elapsed if elapsed > 0 else 0
        remaining = self._total - self._processed
        eta = remaining / rate if rate > 0 else 0
        return {
            "processed": self._processed,
            "cached": self._cached,
            "errors": self._errors,
            "total": self._total,
            "elapsed": elapsed,
            "rate": rate,
            "eta": eta
        }

    def print_progress(self):
        stats = self.get_stats()
        print(f"[{stats['processed']}/{stats['total']}] "
              f"Processed: {stats['processed']}, "
              f"Cached: {stats['cached']}, "
              f"Errors: {stats['errors']}, "
              f"Rate: {stats['rate']:.2f}/s, "
              f"ETA: {stats['eta']:.0f}s")


def classify_review(text: str) -> dict:
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": config.CANDIDATE_LABELS, "multi_label": True}
    }

    start_time = time.time()
    try:
        resp = requests.post(config.ZS_API, headers=config.HEADERS, json=payload, timeout=30)
        inference_time = time.time() - start_time
        return {"data": resp.json(), "inference_time": inference_time, "error": None}
    except Exception as e:
        inference_time = time.time() - start_time
        return {"data": None, "inference_time": inference_time, "error": str(e)}


def parse_response(data: dict) -> tuple:
    if data is None:
        return [], []

    labels = [p.get("label", "") for p in data]
    scores = [p.get("score", 0.0) for p in data]
    return labels, scores


def analyze_response(data: dict) -> dict:
    sentiment_labels = config.SENTIMENT_LABELS
    vector_labels = config.VECTOR_LABELS

    labels, scores = parse_response(data)
    if not labels:
        return None

    best_score = 0.0
    best_label = "neutral"
    for label, score in zip(labels, scores):
        if label in sentiment_labels and score > best_score:
            best_label = label
            best_score = score

    vectors_selected = [
        label for label, score in zip(labels, scores)
        if label in vector_labels and score >= config.VECTOR_THRESHOLD
    ]
    vectors_str = ", ".join(vectors_selected)

    return {
        "sentimentLabel": best_label,
        "score": float(best_score),
        "vectors": vectors_str
    }


def process_reviews(review_ids: list, cache: Cache, progress: ProgressTracker, review_texts: dict):
    for review_id in review_ids:
        if cache.is_processed(review_id):
            progress.increment_processed()
            progress.increment_cached()
            continue

        if cache.is_cached(review_id):
            progress.increment_processed()
            progress.increment_cached()
            cache.save_processed()
            continue

        review_text = review_texts.get(review_id, "")
        if not review_text:
            progress.increment_errors()
            continue

        api_result = classify_review(review_text)

        if api_result.get("error"):
            progress.increment_errors()
            continue

        analysis = analyze_response(api_result.get("data"))
        if not analysis:
            progress.increment_errors()
            continue

        result = InferenceResult(
            internalReviewId=review_id,
            sentimentLabel=analysis["sentimentLabel"],
            score=analysis["score"],
            vectors=analysis["vectors"]
        )

        cache.add_result(review_id, result)
        cache.save_cache()
        cache.save_processed()
        progress.increment_processed()

        if progress.get_stats()["processed"] % 10 == 0:
            progress.print_progress()


def load_review_texts() -> dict:
    df = pd.read_excel(config.DATA_FILE, dtype={"internalReviewId": str})
    df_valid = df[
        df["reviewText"].notna() &
        (df["reviewText"].astype(str).str.strip() != "")
    ]
    return dict(zip(
        df_valid["internalReviewId"].astype(str),
        df_valid["reviewText"].astype(str)
    ))


def get_valid_review_ids() -> list:
    df = pd.read_excel(config.DATA_FILE, dtype={"internalReviewId": str})
    df_valid = df[
        df["reviewText"].notna() &
        (df["reviewText"].astype(str).str.strip() != "")
    ]
    return df_valid["internalReviewId"].astype(str).tolist()


def show_status():
    cache = Cache()
    processed = cache.get_processed_count()
    total = 56135
    print(f"=== Status ===")
    print(f"Total valid entries: {total}")
    print(f"Processed/Cached: {processed}")
    print(f"Remaining: {total - processed}")
    print(f"Cache size: {cache.get_cache_size()}")


def main():
    parser = argparse.ArgumentParser(description="Batch inference on all valid reviews")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from where it left off")
    parser.add_argument("--status", action="store_true",
                        help="Show current status and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of inferences (for testing)")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    review_texts = load_review_texts()
    all_ids = get_valid_review_ids()

    cache = Cache()

    if args.resume:
        processed = cache.get_processed_count()
        print(f"Resuming: {processed}/{len(all_ids)} already processed")

    ids_to_process = [rid for rid in all_ids if not cache.is_processed(rid)]
    if args.limit:
        ids_to_process = ids_to_process[:args.limit]
    print(f"IDs to process: {len(ids_to_process)}")

    progress = ProgressTracker(total=len(all_ids))
    progress._processed = cache.get_processed_count()
    progress._cached = progress._processed

    print(f"Starting inference on {len(ids_to_process)} reviews...")

    start_time = time.time()

    process_reviews(ids_to_process, cache, progress, review_texts)

    elapsed = time.time() - start_time
    stats = progress.get_stats()

    print(f"\n=== Complete ===")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Processed: {stats['processed']}")
    print(f"Cached: {stats['cached']}")
    print(f"Errors: {stats['errors']}")
    print(f"Cache size: {cache.get_cache_size()}")


if __name__ == "__main__":
    main()
