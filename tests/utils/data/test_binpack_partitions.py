# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the ms-swift-compatible worst-fit-decreasing packer."""

import random

from relax.utils.data.data import get_binpack_partitions, get_minimum_num_micro_batch_size


def _reference_to_constant_volume(values, volume):
    """Transcription of ``binpacking.to_constant_volume`` (v2.0.1), the packer
    ms-swift calls for ``packing_strategy='binpack'``."""
    bins, weight_sum = [], []
    for i in sorted(range(len(values)), key=lambda i: -values[i]):
        weight = values[i]
        candidates = [b for b in range(len(weight_sum)) if weight_sum[b] + weight <= volume]
        if candidates:
            target = min(candidates, key=lambda b: weight_sum[b])
        else:
            bins.append([])
            weight_sum.append(0)
            target = len(bins) - 1
        bins[target].append(i)
        weight_sum[target] += weight
    return [sorted(current_bin) for current_bin in bins]


def test_binpack_partitions_matches_ms_swift_reference():
    random.seed(0)
    for _ in range(500):
        values = [random.randint(1, 90000) for _ in range(random.randint(1, 40))]
        assert get_binpack_partitions(values, 131072) == _reference_to_constant_volume(values, 131072)


def test_binpack_partitions_assigns_every_sample_exactly_once():
    random.seed(1)
    for _ in range(500):
        n = random.randint(1, 40)
        values = [random.randint(1, 90000) for _ in range(n)]
        partitions = get_binpack_partitions(values, 131072)
        assert sorted(i for part in partitions for i in part) == list(range(n))


def test_binpack_partitions_respects_max_tokens_cap():
    """The property Karmarkar-Karp balancing does not give us: no bin may
    exceed the budget unless a single sample already does."""
    random.seed(2)
    for _ in range(500):
        values = [random.randint(1, 90000) for _ in range(random.randint(1, 40))]
        for part in get_binpack_partitions(values, 131072):
            assert sum(values[i] for i in part) <= 131072


def test_binpack_partitions_isolates_oversize_sample():
    partitions = get_binpack_partitions([200000, 10, 20], 131072)
    assert partitions == [[0], [1, 2]]


def test_binpack_partitions_handles_empty_input():
    assert get_binpack_partitions([], 131072) == []


def test_binpack_partitions_may_exceed_first_fit_bin_count():
    """Worst-fit-decreasing is not always as tight as plain first-fit, which is
    why callers must derive num_microbatches from this function rather than
    from ``get_minimum_num_micro_batch_size``."""
    random.seed(0)
    seen_more = False
    for _ in range(3000):
        values = [random.randint(1, 90000) for _ in range(random.randint(1, 40))]
        if len(get_binpack_partitions(values, 131072)) > get_minimum_num_micro_batch_size(values, 131072):
            seen_more = True
            break
    assert seen_more
