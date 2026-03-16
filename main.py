import time
import argparse
import json

import config
from api_client import HuggingFaceClient
from data_loader import DataLoader
from analyzer import SentimentAnalyzer
from output_handler import OutputHandler

time_start = time.time()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", type=int, default=15,
                        help="Number of random valid entries to process")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Run inference on all valid entries")
    parser.add_argument(
        "ids", nargs="*", help="Specific review IDs to process")
    return parser.parse_args()


def main():
    args = parse_args()
    target_ids = [str(int(x)) for x in args.ids] if args.ids else None
    count = args.count
    run_all = args.all

    loader = DataLoader()
    analyzer = SentimentAnalyzer()
    client = HuggingFaceClient()
    output_handler = OutputHandler()

    df = loader.load_data()
    df_valid = loader.get_valid_entries(df)

    if target_ids is not None:
        df_valid = loader.filter_by_ids(df_valid, target_ids)
    elif run_all:
        pass  # Use all valid entries
    else:
        df_valid = loader.sample_random(df_valid, count)

    results = []
    inference_times = []

    for _, row in df_valid.iterrows():
        review_id = str(row.get("internalReviewId", ""))
        review_text = str(row.get("reviewText", ""))

        if target_ids and review_id not in target_ids:
            continue

        api_result = client.classify(review_text, config.CANDIDATE_LABELS)
        inference_times.append(api_result["inference_time"])

        if api_result.get("error"):
            results.append({
                "internalReviewId": review_id,
                "raw_response": api_result["error"]
            })
            continue

        analysis = analyzer.analyze(api_result)
        if analysis:
            results.append({
                "internalReviewId": review_id,
                **analysis
            })
        else:
            results.append({
                "internalReviewId": review_id,
                "raw_response": "Failed to analyze"
            })

    total_duration = time.time() - time_start
    avg_time = sum(inference_times) / \
        len(inference_times) if inference_times else 0
    total_valid = loader.get_total_valid_count(df)

    output_handler.append_run(results, total_duration, avg_time)
    output_handler.print_summary(
        total_valid, len(results), total_duration, avg_time)

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    time_start = time.time()
    main()
