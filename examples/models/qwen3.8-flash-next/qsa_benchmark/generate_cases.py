# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generate a small JSON dataset of synthetic packed layouts without CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark import ROOT, cases, validate_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / "data/synthetic_cases.json").read_text())
    config["seed"] = args.seed
    config["cases"] = cases(args.seed)
    validate_cases(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    sys.stdout.write(f"Generated {len(config['cases'])} synthetic layouts with seed {args.seed}\n")


if __name__ == "__main__":
    main()
