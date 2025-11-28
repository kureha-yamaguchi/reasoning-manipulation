#!/usr/bin/env python3
"""Extract best_layer from experiment statistics JSON."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--try_layers", required=True)
    args = parser.parse_args()

    layers_underscored = args.try_layers.replace(",", "_")
    stats_file = Path(f"results/{args.model_name}/attack_results/layer_statistics_{args.type}_layers_{layers_underscored}.json")

    if not stats_file.exists():
        print(f"ERROR: {stats_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(stats_file) as f:
        data = json.load(f)

    best_layer = data.get("best_layer")
    if best_layer is None:
        print(f"ERROR: No 'best_layer' in {stats_file}", file=sys.stderr)
        sys.exit(1)

    print(best_layer)  # Only output the value


if __name__ == "__main__":
    main()