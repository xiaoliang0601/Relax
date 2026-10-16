# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from relax.utils.reloadable_process_group import ReloadableProcessGroup


@pytest.mark.skipif(
    not hasattr(dist.ProcessGroup, "all_gather_single"), reason="Requires PyTorch's single-tensor collective API"
)
def test_cached_tensor_collectives_follow_replaced_process_group(tmp_path: Path) -> None:
    if dist.is_initialized():
        pytest.skip("Requires an isolated default process group")

    # MCore caches these before Relax patches torch.distributed. Calling the
    # cached functions must still dispatch to the wrapper's current inner PG.
    gather = dist.all_gather_into_tensor
    scatter = dist.reduce_scatter_tensor
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    wrapper = None
    try:
        wrapper = ReloadableProcessGroup(dist.new_group([0], backend="gloo"), [0])
        for cycle in range(2):
            source = torch.tensor([float(cycle + 1)])
            gathered = torch.empty_like(source)
            reduced = torch.empty_like(source)
            gather(gathered, source, group=wrapper)
            scatter(reduced, source, group=wrapper)
            torch.testing.assert_close(gathered, source)
            torch.testing.assert_close(reduced, source)
            if cycle == 0:
                dist.destroy_process_group(wrapper.group)
                wrapper.group = dist.new_group([0], backend="gloo")
    finally:
        if wrapper is not None:
            dist.destroy_process_group(wrapper.group)
            ReloadableProcessGroup.GROUPS[os.getpid()].remove(wrapper)
        dist.destroy_process_group()
