# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU checks for the public synthetic QSA inputs and measured data
provenance."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "examples/models/qwen3.8-flash-next/qsa_benchmark"


def load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location("synthetic_qsa_test", BENCH / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestQsaH20Benchmark(unittest.TestCase):
    def test_stored_layouts_match_generator_and_reference_runs(self) -> None:
        module = load_benchmark()
        config = json.loads((BENCH / "data/synthetic_cases.json").read_text())
        self.assertEqual(module.validate_cases(config), module.cases(config["seed"]))
        expected = {case["name"]: case for case in config["cases"]}
        module.verify_sources(ROOT)
        reference = BENCH / "data/reference/h20_20261104"
        provenance = json.loads((BENCH / "provenance/reference.json").read_text())
        self.assertEqual(
            hashlib.sha256((BENCH / provenance["measurement_harness_file"]).read_bytes()).hexdigest(),
            provenance["measurement_harness_sha256"],
        )
        for run in provenance["reference_runs"]:
            raw = gzip.decompress((BENCH / run["file"]).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), run["uncompressed_sha256"])
            report = json.loads(raw)
            self.assertEqual(report["benchmark_sha256"], provenance["measurement_harness_sha256"])
            for case in report["cases"]:
                self.assertEqual(case["lengths"], expected[case["name"]]["lengths"])
            self.assertTrue((reference / (run["name"] + "-medians.csv")).is_file())

    def test_rejects_invalid_boundaries_before_cuda(self) -> None:
        module = load_benchmark()
        original = json.loads((BENCH / "data/synthetic_cases.json").read_text())
        for field, value in (
            ("lengths", [1, -1]),
            ("total_tokens", 1),
            ("padding_tokens", -1),
            ("real_documents", 0),
            ("name", "../escape"),
        ):
            with self.subTest(field=field):
                invalid = copy.deepcopy(original)
                invalid["cases"][0][field] = value
                with self.assertRaises(ValueError):
                    module.validate_cases(invalid)

    def test_describe_works_outside_checkout_without_ml_imports(self) -> None:
        code = (
            "import runpy,sys; "
            "sys.modules.update({name:None for name in ('torch','triton','numpy','megatron')}); "
            f"sys.argv=[{str(BENCH / 'benchmark.py')!r},'--describe','--cases','packed-short64']; "
            f"runpy.run_path({str(BENCH / 'benchmark.py')!r},run_name='__main__')"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=directory, capture_output=True, text=True, check=True
            )
        description = json.loads(result.stdout)
        self.assertEqual(description["cases"][0]["total_tokens"], 65536)
        self.assertEqual(description["source_commit"], "d16f2e3ec45be059db6377295090b1d49257892c")

    def test_sample_csv_values_reproduce_reference_medians(self) -> None:
        reference = BENCH / "data/reference/h20_20261104"
        total_rows = 0
        for run in ("h20-6q", "h20-12q"):
            values = {}
            for path in (reference / "samples").glob(run + "-*.csv"):
                with path.open() as stream:
                    for row in csv.DictReader(stream):
                        key = (row["case"], row["backend"], row["metric"])
                        values.setdefault(key, []).append(float(row["cuda_ms"]))
                        total_rows += 1
            with (reference / (run + "-medians.csv")).open() as stream:
                for row in csv.DictReader(stream):
                    samples = values[(row["case"], row["backend"], row["metric"])]
                    self.assertEqual(len(samples), 27)
                    self.assertEqual(statistics.median(samples), float(row["median_cuda_ms"]))
        self.assertEqual(total_rows, 7830)


if __name__ == "__main__":
    unittest.main()
