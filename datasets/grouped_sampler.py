from collections import defaultdict
from pathlib import Path
import random
from torch.utils.data import Sampler


class FolderGroupedBatchSampler(Sampler):
    """
    Groups samples by parent folder under dataset.image_dir.

    Args:
        dataset: dataset with `samples` and `image_dir`
        batch_size: number of samples per batch
        drop_last: if True, remove incomplete final batch
        shuffle: if True, shuffle batch order across folders/chunks
        shuffle_within_folder: if True, shuffle samples inside each folder before batching
        random_drop_within_folder: if True and drop_last=True, randomly remove the
            remainder samples from each folder so only full batches remain
        seed: base random seed
    """

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        shuffle=False,
        shuffle_within_folder=False,
        random_drop_within_folder=False,
        seed=0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.shuffle_within_folder = shuffle_within_folder
        self.random_drop_within_folder = random_drop_within_folder
        self.seed = seed
        self.epoch = 0

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if self.random_drop_within_folder and not self.drop_last:
            raise ValueError(
                "random_drop_within_folder=True requires drop_last=True, "
                "because random dropping is only used to remove incomplete remainders."
            )

        self.folder_to_indices = self._build_folder_indices()

    def _build_folder_indices(self):
        folder_to_indices = defaultdict(list)

        for idx, sample in enumerate(self.dataset.samples):
            img_path = Path(sample["image_path"])
            rel_path = img_path.relative_to(self.dataset.image_dir)
            folder_key = str(rel_path.parent)  # e.g. "cats", "dogs"
            folder_to_indices[folder_key].append(idx)

        return dict(sorted(folder_to_indices.items(), key=lambda x: x[0]))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _make_batches(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for folder_key, indices in self.folder_to_indices.items():
            folder_indices = list(indices)

            # Optional shuffle within folder before any dropping/chunking
            if self.shuffle_within_folder:
                rng.shuffle(folder_indices)

            if self.drop_last:
                remainder = len(folder_indices) % self.batch_size

                if remainder != 0:
                    if self.random_drop_within_folder:
                        # Randomly drop exactly `remainder` samples from this folder
                        drop_positions = set(rng.sample(range(len(folder_indices)), remainder))
                        folder_indices = [
                            idx for pos, idx in enumerate(folder_indices)
                            if pos not in drop_positions
                        ]
                    else:
                        # Deterministic tail drop
                        usable = (len(folder_indices) // self.batch_size) * self.batch_size
                        folder_indices = folder_indices[:usable]

            for i in range(0, len(folder_indices), self.batch_size):
                batch = folder_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        return all_batches

    def __iter__(self):
        batches = self._make_batches()
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.folder_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total