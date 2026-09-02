#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vstd.generator import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate a Visual State Tracking Dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("datasets"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = generate_dataset(args.config, args.output_root, overwrite=args.overwrite)
    print(f"Generated dataset: {out_dir}")


if __name__ == "__main__":
    main()
