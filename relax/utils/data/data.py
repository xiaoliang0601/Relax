# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import random

from relax.utils.data.data_utils import (
    BaseDataset,
    filter_long_prompts,
    read_file,
)
from relax.utils.types import Sample


__all__ = ["Dataset", "BaseDataset"]

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class Dataset(BaseDataset):
    """Eager-loading dataset that loads all data into memory at initialization.

    This is suitable for smaller datasets or when random access performance is
    critical. For large datasets, consider using StreamingDataset.
    """

    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        system_prompt=None,
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        use_audio_in_video=False,
        multimodal_config=None,
        custom_prompt_func=None,
        teacher_prompt_key=None,
        teacher_multimodal_keys=None,
    ):
        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            prompt_key=prompt_key,
            multimodal_keys=multimodal_keys,
            label_key=label_key,
            tool_key=tool_key,
            metadata_key=metadata_key,
            system_prompt=system_prompt,
            seed=seed,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            use_audio_in_video=use_audio_in_video,
            multimodal_config=multimodal_config,
            custom_prompt_func=custom_prompt_func,
            teacher_prompt_key=teacher_prompt_key,
            teacher_multimodal_keys=teacher_multimodal_keys,
        )

        # Load all samples into memory
        origin_samples = []
        for data in read_file(path):
            sample = self._process_data(data)
            origin_samples.append(sample)

        # Apply length filtering
        if max_length is not None:
            self.origin_samples = filter_long_prompts(
                origin_samples,
                tokenizer,
                processor,
                max_length,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
            )
        else:
            logger.warning("max_length is not set. Skipping filter_long_prompts.")
            self.origin_samples = origin_samples

        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch.

        Args:
            new_epoch_id: Epoch identifier for reproducible shuffling
        """
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index."""
        return self.samples[idx]

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)


def get_first_fit_partitions(total_lengths: list[int], max_tokens_per_gpu: int) -> list[list[int]]:
    """Pack indices in input order; oversized samples get their own bin."""
    batches: list[int] = []
    partitions: list[list[int]] = []
    for index, length in enumerate(total_lengths):
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                partitions[i].append(index)
                break
        else:
            batches.append(length)
            partitions.append([index])

    return partitions


def get_minimum_num_micro_batch_size(total_lengths: list[int], max_tokens_per_gpu: int) -> int:
    return len(get_first_fit_partitions(total_lengths, max_tokens_per_gpu))


def get_binpack_partitions(total_lengths, max_tokens_per_gpu):
    """Worst-fit-decreasing bin packing, matching ms-swift's
    ``packing_strategy="binpack"``.

    ms-swift delegates packing to ``binpacking.to_constant_volume``, which sorts
    items by descending length and drops each into the *least loaded* bin it still
    fits in, opening a new bin only when none has room.

    Differs from the two existing helpers in ways that matter for cross-framework
    comparison: unlike :func:`relax.utils.data.seqlen_balancing.get_seqlen_balanced_partitions`
    (Karmarkar-Karp) every bin respects ``max_tokens_per_gpu`` except an oversized singleton,
    and unlike :func:`get_minimum_num_micro_batch_size` (plain first-fit, input
    order) it returns the assignment rather than just the count.

    Packing is local to each Relax update. The cross-update rolling window
    used by ms-swift is not replicated here.

    Returns a list of bins, each a sorted list of indices into ``total_lengths``.
    """
    bin_sums: list[int] = []
    bins: list[list[int]] = []
    # Stable descending sort, matching binpacking's revargsort tie-breaking.
    for index in sorted(range(len(total_lengths)), key=lambda i: -total_lengths[i]):
        length = total_lengths[index]
        candidates = [b for b in range(len(bin_sums)) if bin_sums[b] + length <= max_tokens_per_gpu]
        if candidates:
            # Least-loaded fit. min() keeps the lowest index on ties, like argmin.
            target = min(candidates, key=lambda b: bin_sums[b])
        else:
            bin_sums.append(0)
            bins.append([])
            target = len(bins) - 1
        bin_sums[target] += length
        bins[target].append(index)
    return [sorted(current_bin) for current_bin in bins]
