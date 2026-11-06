# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preflight and runtime evidence for the Qishu frozen CPU PLE table recipe."""

import argparse
import errno
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any


_PUBLISHED_AUDITS: set[tuple[Path, int]] = set()
_TRANSIENT_AUDIT_ERRORS = {errno.EAGAIN, errno.EINTR, errno.EIO, errno.ESTALE, errno.ETIMEDOUT}


def _write_initial_audit(path: Path, evidence: dict[str, Any]) -> None:
    contents = json.dumps(evidence, indent=2) + "\n"
    temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(contents, encoding="utf-8")
            temporary.replace(path)
            return
        except OSError as error:
            if error.errno not in _TRANSIENT_AUDIT_ERRORS or attempt == 2:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.2 * (attempt + 1))


def before_train_step(
    args: argparse.Namespace,
    rollout_id: int,
    step_id: int,
    model: list[Any],
    optimizer: Any,
    opt_param_scheduler: Any,
) -> None:
    """Check table ownership and all remaining PLE parameters before
    training."""
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    from relax.utils.logging_utils import get_logger

    if step_id != 0:
        return
    if os.environ.get("QWEN38_PLE_CPU_OFFLOAD") != "1":
        raise ValueError("QWEN38_PLE_CPU_OFFLOAD=1 must reach every training actor")
    parameters = []
    tables = []
    for chunk in model:
        for name, buffer in chunk.named_buffers():
            if name.endswith("per_layer_embedding.ple_embedding.ngram_embedding.weight"):
                record = {
                    "name": name,
                    "device": str(buffer.device),
                    "shape": list(buffer.shape),
                    "requires_grad": buffer.requires_grad,
                    "pinned": buffer.is_pinned(),
                }
                tables.append(record)
                if buffer.device.type != "cpu" or buffer.requires_grad or not buffer.is_pinned():
                    raise ValueError(f"PLE table must be a frozen pinned CPU buffer: {record}")
        for name, parameter in chunk.named_parameters():
            if ".per_layer_embedding." not in name:
                continue
            record = {"name": name, "device": str(parameter.device), "requires_grad": parameter.requires_grad}
            parameters.append(record)
            if name.endswith("ngram_embedding.weight"):
                raise ValueError("PLE table is still an optimizer Parameter")
            if not parameter.requires_grad or parameter.device.type != "cuda":
                raise ValueError(f"PLE projections, norms and convolution must stay trainable: {record}")
        for module in chunk.modules():
            if type(module).__name__ == "PinnedVocabParallelEmbedding" and not module._table_loaded:
                raise ValueError("PLE host table was not loaded from HF or checkpoint")
    if parameters and not tables:
        raise ValueError("A PLE-owning rank has no CPU table")
    rank = dist.get_rank(group=dist.group.WORLD)
    output = Path(os.environ["QISHU_PLE_AUDIT_DIR"])
    audit_key = (output, rank)
    if audit_key not in _PUBLISHED_AUDITS:
        # The coordinator needs the first record for GPU handover. Keep the
        # checks above on every rollout, with no subsequent cloud writes.
        evidence = {
            "rank": rank,
            "tp": mpu.get_tensor_model_parallel_rank(),
            "pp": mpu.get_pipeline_model_parallel_rank(),
            "rollout_id": rollout_id,
            "tables": tables,
            "trainable_ple_parameters": parameters,
            "freeze_patterns": args.freeze_params_name_list,
            "tensor_values_read": False,
        }
        _write_initial_audit(output / f"rank-{rank}-rollout-{rollout_id}.json", evidence)
        _PUBLISHED_AUDITS.add(audit_key)
    if tables:
        get_logger(__name__).info(
            "PLE CPU audit passed: rank=%s, tables=%s, trainable=%s", rank, len(tables), len(parameters)
        )


def _mount_root(path: str) -> str:
    parts = Path(path).parts
    return str(Path(*parts[:5])) if path.startswith("/mnt/") else str(Path(path).parent)


def wait_ready(gpus: int) -> None:
    """Wait for a newly created QS Ray cluster to register its GPU workers."""
    import ray

    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
    deadline = time.monotonic() + 1800
    while ray.cluster_resources().get("GPU", 0) < gpus:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Expected {gpus} GPUs; cluster is still starting. Resources are preserved.")
        time.sleep(5)
    available = ray.available_resources().get("GPU", 0)
    if available < gpus:
        raise ValueError(f"Need {gpus} free GPUs; only {available} are free. Existing jobs are preserved.")


def preflight(data: str, model: str, gpus: int) -> dict[str, Any]:
    """Inspect original JSONL and refuse missing image mounts or occupied
    GPUs."""
    import ray
    from megatron.core.transformer.per_layer_embedding import PinnedVocabParallelEmbedding  # noqa: F401

    config = json.loads((Path(model) / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != 48 or text.get("ple_layer_ids") != [2]:
        raise ValueError("This recipe requires the full 48-layer Flash-Next checkpoint with PLE layer [2]")
    roots: set[str] = {model, str(Path(model) / "config.json"), data}
    image_examples: dict[str, list[str]] = {}
    rows = image_rows = 0
    with open(data) as stream:
        for rows, line in enumerate(stream, 1):
            sample = json.loads(line)
            if not isinstance(sample.get("messages"), list) or not sample["messages"]:
                raise ValueError(f"Row {rows} has no messages")
            images = sample.get("images") or []
            if not isinstance(images, list) or any(not isinstance(image, str) for image in images):
                raise ValueError(f"Row {rows}: expected images to be a list of paths")
            image_rows += bool(images)
            for image in images:
                root = _mount_root(image)
                roots.add(root)
                examples = image_examples.setdefault(root, [])
                if len(examples) < 2 and image not in examples:
                    examples.append(image)
    roots.update(image for examples in image_examples.values() for image in examples)
    if not rows:
        raise ValueError("Empty JSONL")
    from urllib.request import urlopen

    endpoint = os.environ.get("RAY_JOBS_ADDRESS") or os.environ.get("RAY_DASHBOARD_ADDRESS", "http://127.0.0.1:8265")
    with urlopen(endpoint.rstrip("/") + "/api/serve/applications/", timeout=15) as response:
        applications = json.load(response).get("applications", {})
    if applications:
        raise ValueError(f"Existing Ray Serve applications must be preserved: {list(applications)}")
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=False)
    resources = ray.available_resources()
    if resources.get("GPU", 0) < gpus:
        raise ValueError(f"Need {gpus} free GPUs; available: {resources.get('GPU', 0)}. Existing jobs are preserved.")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=0, num_gpus=8)
    def check_node(required: list[str]) -> dict[str, Any]:
        import socket
        import subprocess

        import torch

        hardware = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                text=True,
            )
            .strip()
            .splitlines()
        )
        # Check the CUDA capability used by training. Driver-reported capability
        # differs from CUDA on the H800Z nodes (8.9 versus 9.0).
        capabilities = [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())]
        if len(hardware) != 8 or len(capabilities) != 8 or any(capability != [9, 0] for capability in capabilities):
            raise ValueError(
                f"Expected 8 Hopper GPUs on {socket.gethostname()}: CUDA={capabilities}, driver={hardware}"
            )
        return {
            "node": socket.gethostname(),
            "missing": [path for path in required if not os.access(path, os.R_OK)],
            "hardware": hardware,
            "cuda_capabilities": capabilities,
        }

    checks = ray.get(
        [
            check_node.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(
                sorted(roots)
            )
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) > 0
        ]
    )
    missing = [check for check in checks if check["missing"]]
    if missing:
        raise ValueError(f"Missing paths on GPU nodes: {missing}. Mount them on the workers or add QS mount paths.")
    return {"data": data, "rows": rows, "image_rows": image_rows, "gpus": gpus, "nodes": checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "wait-ready"])
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--gpus", required=True, type=int)
    options = parser.parse_args()
    if options.action == "wait-ready":
        wait_ready(options.gpus)
    else:
        if not options.data or not options.model:
            parser.error("preflight requires --data and --model")
        print(json.dumps(preflight(options.data, options.model, options.gpus), indent=2))
