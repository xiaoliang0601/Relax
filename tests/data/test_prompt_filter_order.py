# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import Future
from dataclasses import replace

import pytest

from relax.utils.data import data_utils
from relax.utils.types import Sample


@pytest.mark.parametrize("completion_order", [(3, 2, 1, 0), (1, 3, 0, 2)])
def test_parallel_prompt_filter_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch, completion_order: tuple
) -> None:
    samples = [Sample(prompt=str(index)) for index in range(4)]

    class CompletedExecutor:
        def __init__(self, **kwargs):
            self.pending = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, sample, max_length):
            future = Future()
            self.pending.append((future, sample))
            if len(self.pending) == len(samples):
                for index in completion_order:
                    pending, value = self.pending[index]
                    if value.prompt == "2":
                        pending.set_exception(ValueError("unreadable image"))
                    else:
                        checked = replace(value, metadata={"checked": True})
                        pending.set_result((checked, value.prompt != "1"))
            return future

    monkeypatch.setattr(data_utils, "ProcessPoolExecutor", CompletedExecutor)
    filtered = data_utils.filter_long_prompts(samples, None, object(), max_length=100, num_workers=4)
    assert [sample.prompt for sample in filtered] == ["0", "2", "3"]
    assert filtered[0].metadata == {"checked": True}
    assert filtered[1] is samples[2]  # Preserve the existing exception fallback.
