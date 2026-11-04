# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Record an explicitly selected NVIDIA QSA source tree for optimization
experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsa-source", type=Path, required=True)
    parser.add_argument("--commit", required=True, help="Commit identifying the experimental NVIDIA checkout")
    parser.add_argument("--output", type=Path, default=Path("nv-source-manifest.optimized.json"))
    args = parser.parse_args()
    if args.qsa_source.name != "QSA" or not (args.qsa_source / "api.py").is_file():
        parser.error("--qsa-source must point to checkout/python/cudnn/QSA")
    files = []
    for path in sorted(args.qsa_source.rglob("*.py")):
        data = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(args.qsa_source)),
                "git_blob": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(),
            }
        )
    manifest = {
        "repository": "https://github.com/NVIDIA/cudnn-frontend",
        "commit": args.commit,
        "scope": "Explicit experimental source tree; not the original pinned baseline",
        "files": files,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    sys.stdout.write(f"Recorded {len(files)} QSA source files in {args.output}\n")


if __name__ == "__main__":
    main()
