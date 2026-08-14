"""Paired dataset, patch sampling, dihedral augmentation, CutBlur, real/synthetic mixing.

SPEC 6.1-6.3. Paired crop rule: LR origin (i,j) => GT origin (2i,2j) (V26).
Validation split is read from configs/split_val.txt and NEVER regenerated at runtime (V29).

Data is .npy float32: np.load(path, allow_pickle=False). No image library.

Hazard: train/ and test_NoisyLR/ filenames collide (both start at 000000.npy, different
images). Key everything by split or full path.

Owner: data-pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch.utils.data import Dataset


class PairedRestorationDataset(Dataset):
    """LR/GT pairs with paired crops and identical augmentation applied to both."""

    def __init__(self, data_root: str | Path, cfg: Mapping[str, Any], split: str = "train") -> None:
        raise NotImplementedError("PairedRestorationDataset.__init__: not implemented yet")

    def __len__(self) -> int:
        raise NotImplementedError("PairedRestorationDataset.__len__: not implemented yet")

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError("PairedRestorationDataset.__getitem__: not implemented yet")
