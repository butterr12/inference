import os
import json
import uuid
import threading
import time
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path

from models import InferenceResult, JobStatus
from api_client import HuggingFaceClient
from analyzer import SentimentAnalyzer
import config


CACHE_FILE = "inference_cache.json"
JOBS_FILE = "jobs.json"


def _load_jobs() -> Dict:
    jobs_path = Path(JOBS_FILE)
    if jobs_path.exists():
        try:
            with open(jobs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_jobs(jobs: Dict) -> None:
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


def _update_job(job_id: str, updates: Dict) -> None:
    jobs = _load_jobs()
    if job_id in jobs:
        jobs[job_id].update(updates)
        _save_jobs(jobs)


def _load_cache() -> Dict[str, InferenceResult]:
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {k: InferenceResult(**v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_cache(cache: Dict[str, InferenceResult]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({k: v.model_dump() for k, v in cache.items()}, f, indent=2)


def _load_reviews_df() -> pd.DataFrame:
    df = pd.read_excel(config.DATA_FILE, dtype={"internalReviewId": str})
    df_valid = df[
        df["reviewText"].notna() &
        (df["reviewText"].astype(str).str.strip() != "")
    ].copy()
    df_valid["internalReviewId"] = df_valid["internalReviewId"].astype(str)
    return df_valid


def run_inference(review_text: str, review_id: Optional[str] = None) -> InferenceResult:
    cache = _load_cache()

    if review_id and review_id in cache:
        return cache[review_id]

    text_hash = str(hash(review_text))
    if text_hash in cache:
        result = cache[text_hash]
        if review_id:
            cache[review_id] = result
            _save_cache(cache)
        return result

    client = HuggingFaceClient()
    analyzer = SentimentAnalyzer()

    api_result = client.classify(review_text, config.CANDIDATE_LABELS)
    
    if api_result.get("error"):
        raise Exception(f"HuggingFace API error: {api_result['error']}")

    analysis = analyzer.analyze(api_result)
    if not analysis:
        raise Exception("Failed to analyze API response")

    result = InferenceResult(
        internalReviewId=review_id,
        sentimentLabel=analysis["sentimentLabel"],
        score=analysis["score"],
        vectors=analysis["vectors"]
    )

    if review_id:
        cache[review_id] = result
    cache[text_hash] = result
    _save_cache(cache)

    return result


def get_inference_by_id(review_id: str) -> Optional[InferenceResult]:
    cache = _load_cache()
    return cache.get(review_id)


def get_all_inferences() -> Dict[str, InferenceResult]:
    return _load_cache()


def get_cache_size() -> int:
    cache = _load_cache()
    return len(cache)


def get_review_by_id(review_id: str) -> Optional[Dict]:
    df = _load_reviews_df()
    row = df[df["internalReviewId"] == review_id]
    if row.empty:
        return None
    return {
        "internalReviewId": str(row.iloc[0]["internalReviewId"]),
        "reviewText": str(row.iloc[0]["reviewText"])
    }


def get_all_review_ids() -> List[str]:
    df = _load_reviews_df()
    return df["internalReviewId"].tolist()


def get_all_valid_review_ids() -> List[str]:
    df = _load_reviews_df()
    return df["internalReviewId"].tolist()


def get_job_status(job_id: str) -> Optional[JobStatus]:
    jobs = _load_jobs()
    if job_id not in jobs:
        return None
    job = jobs[job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        total=job["total"],
        processed=job["processed"],
        cached=job["cached"],
        errors=job["errors"]
    )


def get_latest_job_status() -> Optional[JobStatus]:
    jobs = _load_jobs()
    if not jobs:
        return None
    latest_job_id = max(jobs.keys(), key=lambda k: jobs[k].get("created_at", ""))
    job = jobs[latest_job_id]
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        total=job["total"],
        processed=job["processed"],
        cached=job["cached"],
        errors=job["errors"]
    )


def _process_job_worker(job_id: str, review_ids: List[str]) -> None:
    cache = _load_cache()
    processed = 0
    cached = 0
    errors = 0

    client = HuggingFaceClient()
    analyzer = SentimentAnalyzer()

    df = _load_reviews_df()
    review_texts = dict(zip(df["internalReviewId"].astype(str), df["reviewText"].astype(str)))

    for i, review_id in enumerate(review_ids):
        try:
            if review_id in cache:
                cached += 1
            else:
                review_text = review_texts.get(review_id)
                if not review_text:
                    errors += 1
                    continue

                api_result = client.classify(review_text, config.CANDIDATE_LABELS)
                if api_result.get("error"):
                    errors += 1
                    continue

                analysis = analyzer.analyze(api_result)
                if not analysis:
                    errors += 1
                    continue

                result = InferenceResult(
                    internalReviewId=review_id,
                    sentimentLabel=analysis["sentimentLabel"],
                    score=analysis["score"],
                    vectors=analysis["vectors"]
                )
                cache[review_id] = result

            processed += 1

            if processed % 100 == 0:
                _update_job(job_id, {"processed": processed, "cached": cached, "errors": errors})
                _save_cache(cache)

        except Exception:
            errors += 1
            continue

    _save_cache(cache)
    _update_job(job_id, {
        "status": "completed",
        "processed": processed,
        "cached": cached,
        "errors": errors
    })


def process_all_async() -> str:
    job_id = str(uuid.uuid4())
    review_ids = get_all_valid_review_ids()
    total = len(review_ids)

    jobs = _load_jobs()
    jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "total": total,
        "processed": 0,
        "cached": 0,
        "errors": 0,
        "created_at": time.time()
    }
    _save_jobs(jobs)

    thread = threading.Thread(target=_process_job_worker, args=(job_id, review_ids))
    thread.start()

    return job_id
