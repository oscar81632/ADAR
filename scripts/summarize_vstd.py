#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


METRIC_KEYS = [
    "path_length",
    "start_end_manhattan",
    "num_turns",
    "path_bbox_area",
    "path_coverage",
    "long_dependency_score",
    "distractor_ratio",
    "decoy_arrow_count",
]


def load_rows(dataset_dir: Path):
    with (dataset_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def describe(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {}
    n = len(values)
    return {
        "min": values[0],
        "mean": sum(values) / n,
        "p50": values[n // 2],
        "p90": values[int(0.9 * (n - 1))],
        "max": values[-1],
    }


def print_stats(name, stats):
    print(f"{name:24s} min={stats['min']:.4f} mean={stats['mean']:.4f} "
          f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Summarize VSTD metadata distributions.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.dataset_dir)
    print(f"Dataset: {args.dataset_dir}")
    print(f"Samples: {len(rows)}")

    split_counts = Counter(row["split"] for row in rows)
    print("\nSplits:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    label_counts = Counter(row["class_name"] for row in rows)
    print("\nClasses:")
    for cls, count in sorted(label_counts.items()):
        print(f"  {cls}: {count}")

    print("\nMetric Summary:")
    for key in METRIC_KEYS:
        print_stats(key, describe(row.get(key, 0) for row in rows))

    print("\nPath Length x Distractor Counts:")
    bucket = Counter((row["path_length"], row["distractor_ratio"]) for row in rows)
    for (length, distractor), count in sorted(bucket.items()):
        print(f"  L={length:>2} D={distractor:<4}: {count}")

    print("\nBy Split Mean Long Dependency Score:")
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row["long_dependency_score"])
    for split, values in sorted(by_split.items()):
        stats = describe(values)
        print(f"  {split}: {stats['mean']:.4f}")


if __name__ == "__main__":
    main()
