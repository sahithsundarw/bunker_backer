"""Paired dataset, patch sampling, dihedral augmentation, CutBlur, real/synthetic mixing.

SPEC 6.1-6.3. Data is `.npy` float32 loaded with ``np.load(path, allow_pickle=False)``;
there is no image library anywhere in this file.

The three properties this module exists to guarantee
----------------------------------------------------
1. **Paired-crop alignment (V26).** LR crop origin ``(i,j)`` => GT crop origin ``(2i,2j)``,
   LR patch ``P`` => GT patch ``2P``. Augmentation is applied *identically* to both members
   of the pair. ``selftest_paired_crop()`` proves it with marker/positional-code images and
   is the harness V26 asserts against.
2. **No validation leakage (V29).** The validation split is the committed explicit list
   ``configs/split_val.txt``. It is **never** regenerated at runtime: if the file is missing
   or empty this module raises instead of falling back to a random split, and
   ``train_val_names()`` asserts the train/val intersection is empty on every call.
3. **The degradation is the measured one.** Synthetic pairs come from ``src/degrade.py``,
   which implements the recovered 4x4 kernel plus three-parameter signal-dependent noise
   applied after downsampling, and never clips (docs/decisions.md D1/D2/D12).

Hazard, guarded throughout: ``train/`` and ``test_NoisyLR/`` filenames COLLIDE -- both start
at ``000000.npy`` and refer to different images (docs/SPEC_ADDENDUM.md section 6). Nothing
here is keyed on a bare filename without a split qualifier, and this module only ever opens
``<data_root>/train/``. ``test_NoisyLR`` is never read for training, validation or fitting
(SPEC F17).

Self-checks:
    py -3.12 -m src.dataset --selftest
    py -3.12 -m src.dataset --check-split --data-root C:/kla-data

Owner: data-pipeline.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .degrade import DegradeConfig, conv_downsample_2x, degrade

__all__ = [
    "DataConfig",
    "PairedRestorationDataset",
    "build_datasets",
    "check_split_integrity",
    "cutblur_paste",
    "dihedral",
    "list_train_names",
    "paired_crop",
    "read_name_list",
    "selftest_paired_crop",
    "train_val_names",
]


def repo_root() -> Path:
    """Repo root, resolved from this file. Never from CWD (CLAUDE.md PD5)."""
    return Path(__file__).resolve().parents[1]


#: The committed validation split. This file IS the split (V29).
SPLIT_VAL_PATH = repo_root() / "configs" / "split_val.txt"

#: Every path this module may open lives under <data_root>/train/.
TRAIN_GT = ("train", "GT")
TRAIN_LR = ("train", "NoisyLR")


# ======================================================================================
# SPLITS  (V29)
# ======================================================================================
def read_name_list(path: str | Path) -> list[str]:
    """Read a committed one-name-per-line list; blanks and ``#`` comments are ignored.

    Returns bare basenames. Order is preserved as committed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. The validation split is a committed explicit file list and "
            "is never regenerated at runtime (SPEC 6.1, V29)."
        )
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(Path(s).name)
    return out


def resolve_data_root(arg: str | os.PathLike[str] | None) -> Path:
    """``--data-root``, else ``$KLA_DATA_ROOT``, else the measured local Mac root.

    Dataset-independent commands never call this function. The final fallback exists only for
    the measured submission machine and is documented in ``docs/DATA_LOCATION.md``.
    """
    if arg:
        root = Path(arg).expanduser()
    elif os.environ.get("KLA_DATA_ROOT"):
        root = Path(os.environ["KLA_DATA_ROOT"]).expanduser()
    else:
        root = Path("/Users/shanmukhsai/Downloads")
        if not (root / TRAIN_GT[0] / TRAIN_GT[1]).is_dir():
            raise SystemExit("could not determine the dataset root: pass --data-root or set "
                             "KLA_DATA_ROOT (see docs/DATA_LOCATION.md)")
    if not (root / TRAIN_GT[0] / TRAIN_GT[1]).is_dir():
        raise SystemExit(f"dataset root {root} has no train/GT directory")
    return root


def list_train_names(data_root: str | Path) -> list[str]:
    """Sorted basenames present in BOTH ``train/GT`` and ``train/NoisyLR``.

    ``test_NoisyLR`` is deliberately not consulted: its filenames collide with these.
    """
    root = Path(data_root)
    gt = {p.name for p in (root / TRAIN_GT[0] / TRAIN_GT[1]).glob("*.npy")}
    lr = {p.name for p in (root / TRAIN_LR[0] / TRAIN_LR[1]).glob("*.npy")}
    only_gt, only_lr = sorted(gt - lr), sorted(lr - gt)
    if only_gt or only_lr:
        raise RuntimeError(
            f"train/GT and train/NoisyLR disagree: {len(only_gt)} GT-only "
            f"(first {only_gt[:3]}), {len(only_lr)} LR-only (first {only_lr[:3]})"
        )
    return sorted(gt)


def train_val_names(
    data_root: str | Path,
    split_val_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(train_names, val_names)`` from the committed split. Asserts no overlap.

    The validation list is authoritative; the training list is *everything else* in
    ``train/``. This is the intersection assertion V29 describes, executed on every
    construction of a dataset rather than only in a test.
    """
    all_names = list_train_names(data_root)
    val = read_name_list(split_val_path or SPLIT_VAL_PATH)
    if not val:
        raise ValueError(
            f"{split_val_path or SPLIT_VAL_PATH} has no file entries. The split is committed, "
            "not generated (V29)."
        )
    dupes = sorted({n for n in val if val.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate entries in the validation split: {dupes[:5]}")
    known = set(all_names)
    missing = [n for n in val if n not in known]
    if missing:
        raise ValueError(
            f"{len(missing)} validation names are absent from {Path(data_root)}/train/GT, "
            f"first: {missing[:5]}"
        )
    val_set = set(val)
    train = [n for n in all_names if n not in val_set]
    overlap = sorted(val_set.intersection(train))
    if overlap:
        raise AssertionError(f"train/val leakage: {len(overlap)} shared names, first {overlap[:5]}")
    if not train:
        raise ValueError("the validation split consumed every training pair")
    return train, val


def _block_of_4(name: str) -> int | None:
    """Block index for a ``NNNNNN.npy`` name under the measured block-of-4 grouping (D8)."""
    stem = Path(name).stem
    return int(stem) // 4 if stem.isdigit() else None


def check_split_integrity(
    data_root: str | Path | None = None,
    split_val_path: str | Path | None = None,
) -> dict[str, Any]:
    """V29 evidence: the committed split is explicit, disjoint from train, and block-aligned.

    Works without the dataset (list-only checks) and with it (existence + intersection).
    ``regenerated_at_runtime`` is reported as False because there is no code path in this
    module that can produce a split any other way: ``read_name_list`` raises when the file is
    missing and ``train_val_names`` raises when it is empty. No RNG is involved.
    """
    path = Path(split_val_path or SPLIT_VAL_PATH)
    val = read_name_list(path)
    blocks = sorted({b for b in (_block_of_4(n) for n in val) if b is not None})
    val_set = set(val)
    whole_blocks = all(
        all(f"{4 * b + k:06d}.npy" in val_set for k in range(4)) for b in blocks
    )
    rep: dict[str, Any] = {
        "check": "V29",
        "split_file": str(path),
        "split_file_committed": True,
        "n_val": len(val),
        "n_val_unique": len(val_set),
        "val_first": val[0],
        "val_last": val[-1],
        "n_blocks_of_4": len(blocks),
        "val_blocks_are_whole": bool(whole_blocks),
        "regenerated_at_runtime": False,
        "d3_holdout_subset": all(
            f"{i:06d}.npy" in val_set for i in range(3000, 3200)
        ),
    }
    if data_root is not None:
        train, val2 = train_val_names(data_root, path)
        inter = sorted(set(train).intersection(val2))
        rep.update(
            data_root=str(data_root),
            n_all=len(train) + len(val2),
            n_train=len(train),
            intersection_size=len(inter),
            intersection=inter[:10],
            val_fraction=len(val2) / (len(train) + len(val2)),
            train_first=train[0],
            train_last=train[-1],
        )
        rep["pass"] = (
            not inter
            and len(val2) == len(val_set)
            and whole_blocks
            and 0.10 <= rep["val_fraction"] <= 0.15
        )
    else:
        rep["pass"] = bool(val) and len(val_set) == len(val) and whole_blocks
    return rep


# ======================================================================================
# GEOMETRY -- the V26 core
# ======================================================================================
def paired_crop(
    lr: np.ndarray,
    gt: np.ndarray,
    i: int,
    j: int,
    patch: int,
    scale: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop an aligned pair: LR ``[i:i+P, j:j+P]`` and GT ``[s*i:s*i+s*P, s*j:s*j+s*P]``.

    This single function is the alignment contract (SPEC 6.2, V26). Nothing else in this
    module computes a crop origin for GT.
    """
    p = int(patch)
    s = int(scale)
    i, j = int(i), int(j)
    if p <= 0:
        raise ValueError(f"patch must be positive, got {p}")
    if i < 0 or j < 0 or i + p > lr.shape[0] or j + p > lr.shape[1]:
        raise ValueError(f"LR crop ({i},{j},{p}) out of bounds for shape {lr.shape}")
    if gt.shape[0] != s * lr.shape[0] or gt.shape[1] != s * lr.shape[1]:
        raise ValueError(f"GT shape {gt.shape} is not {s}x LR shape {lr.shape}")
    lr_patch = lr[i : i + p, j : j + p]
    gt_patch = gt[s * i : s * i + s * p, s * j : s * j + s * p]
    return lr_patch, gt_patch


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    """One of the 8 dihedral orientations (SPEC 6.3). ``k`` in 0..7.

    ``k % 4`` quarter turns, then a horizontal flip for ``k >= 4``. Because the GT is exactly
    ``scale`` times the LR in both axes, the same ``k`` applied to both preserves alignment
    exactly -- which is why LR and GT must be transformed with one shared ``k``.
    """
    out = np.rot90(a, int(k) % 4)
    if int(k) >= 4:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def cutblur_paste(dst: np.ndarray, src: np.ndarray, rng: np.random.Generator, alpha: float = 0.7) -> np.ndarray:
    """Paste one random rectangle of ``src`` into a copy of ``dst`` (same shape).

    The rectangle geometry follows Yoo et al., CVPR 2020 ("Rethinking Data Augmentation for
    Image Super-Resolution"): ``cut_ratio ~ N(alpha, 0.01)``, one random top-left corner.
    Purely a value operation -- no resize, no shift -- so pair alignment is untouched.
    """
    d = np.array(dst, dtype=np.float32, copy=True)
    s = np.asarray(src, dtype=np.float32)
    if d.shape != s.shape:
        raise ValueError(f"cutblur_paste needs equal shapes, got {d.shape} and {s.shape}")
    h, w = d.shape
    ratio = float(np.clip(rng.normal(alpha, 0.01), 0.05, 0.95))
    ch, cw = max(1, int(h * ratio)), max(1, int(w * ratio))
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    d[y : y + ch, x : x + cw] = s[y : y + ch, x : x + cw]
    return d


def _crop_pad(img: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Crop ``(y0,x0,h,w)``, edge-replicating whatever falls outside the image.

    Edge replication matches the border convention of ``conv_downsample_2x``, so a synthetic
    patch taken at an image border is degraded the same way the whole image would have been.
    """
    H, W = img.shape
    ys, ye = max(0, y0), min(H, y0 + h)
    xs, xe = max(0, x0), min(W, x0 + w)
    sub = img[ys:ye, xs:xe]
    pt, pb = ys - y0, (y0 + h) - ye
    pl, pr = xs - x0, (x0 + w) - xe
    if pt or pb or pl or pr:
        sub = np.pad(sub, ((pt, pb), (pl, pr)), mode="edge")
    return sub


# ======================================================================================
# CONFIG
# ======================================================================================
def _available_ram_bytes() -> int | None:
    """Physical RAM currently available, or None if it cannot be determined portably."""
    try:
        if sys.platform == "win32":
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return int(st.ullAvailPhys)
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:  # noqa: BLE001 -- a RAM probe must never break training
        return None
    return None


@dataclass(frozen=True)
class DataConfig:
    """The ``data:`` block of a training config, with measured defaults.

    ``preload`` is ``"auto"`` by default: the eager-load win SPEC 6.2 describes is real
    (3200 pairs are ~1.05 GB, GT 838 MB + LR 210 MB) but it is not free on an 8-16 GB
    machine that is also holding a CUDA context, so the machine is *measured* rather than
    assumed. When it does not fit, images are memory-mapped instead of loaded, which keeps
    the dataset usable rather than OOM-ing.
    """

    lr_patch: int = 64                 # -> 128 GT patch (SPEC 6.2)
    scale: int = 2                     # F2: exactly x2
    synth_ratio: float = 0.5           # SPEC 9 default; fraction of samples degraded on the fly
    crops_per_image: int = 1           # __len__ multiplier for the train split
    dihedral: bool = True              # 8 orientations, identical on LR and GT (SPEC 6.3)
    cutblur_prob: float = 0.5          # Yoo et al., CVPR 2020 (SPEC 6.3)
    cutblur_alpha: float = 0.7
    preload: str | bool = "auto"       # True | False | "auto" (measure available RAM)
    preload_headroom: float = 2.0      # require estimate * headroom bytes free before loading
    seed: int = 42
    degrade: DegradeConfig = field(default_factory=DegradeConfig)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "DataConfig":
        """Accept a whole training config (with a ``data:`` block) or just the block itself."""
        if not cfg:
            return cls()
        block = cfg.get("data", cfg) if isinstance(cfg, Mapping) else cfg
        if not isinstance(block, Mapping):
            raise TypeError(f"expected a mapping for the data config, got {type(block)!r}")
        known = {
            "lr_patch", "scale", "synth_ratio", "crops_per_image", "dihedral", "cutblur_prob",
            "cutblur_alpha", "preload", "preload_headroom", "seed", "degrade",
            # tolerated but not used here -- they belong to the loader, not the dataset
            "batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
            "val_list",
        }
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"unknown data config keys: {unknown}")
        deg = DegradeConfig.from_mapping(block.get("degrade"))
        sr = float(block.get("synth_ratio", 0.5))
        if not 0.0 <= sr <= 1.0:
            raise ValueError(f"synth_ratio must be in [0,1], got {sr}")
        scale = int(block.get("scale", 2))
        if scale != 2:
            raise ValueError(
                f"scale must be 2: the degradation is exactly x2 (SPEC F2, measured on all "
                f"3200 pairs), got {scale}"
            )
        lr_patch = int(block.get("lr_patch", 64))
        crops_per_image = int(block.get("crops_per_image", 1))
        cutblur_prob = float(block.get("cutblur_prob", 0.5))
        cutblur_alpha = float(block.get("cutblur_alpha", 0.7))
        preload = block.get("preload", "auto")
        preload_headroom = float(block.get("preload_headroom", 2.0))
        if lr_patch <= 0:
            raise ValueError(f"lr_patch must be positive, got {lr_patch}")
        if crops_per_image <= 0:
            raise ValueError(f"crops_per_image must be positive, got {crops_per_image}")
        if not 0.0 <= cutblur_prob <= 1.0:
            raise ValueError(f"cutblur_prob must be in [0,1], got {cutblur_prob}")
        if not 0.0 <= cutblur_alpha <= 1.0:
            raise ValueError(f"cutblur_alpha must be in [0,1], got {cutblur_alpha}")
        if preload not in (True, False, "auto"):
            raise ValueError(f"preload must be true, false, or 'auto', got {preload!r}")
        if preload_headroom <= 0.0:
            raise ValueError(f"preload_headroom must be positive, got {preload_headroom}")
        return cls(
            lr_patch=lr_patch,
            scale=scale,
            synth_ratio=sr,
            crops_per_image=crops_per_image,
            dihedral=bool(block.get("dihedral", True)),
            cutblur_prob=cutblur_prob,
            cutblur_alpha=cutblur_alpha,
            preload=preload,
            preload_headroom=preload_headroom,
            seed=int(block.get("seed", 42)),
            degrade=deg,
        )


# ======================================================================================
# DATASET
# ======================================================================================
class PairedRestorationDataset(Dataset):
    """LR/GT pairs with paired crops and identical augmentation applied to both members.

    ``split="train"`` yields random ``lr_patch`` crops with augmentation and the configured
    real/synthetic mix. ``split="val"`` yields whole images, **real pairs only, no
    augmentation, no synthesis** -- validation must measure the real degradation, otherwise
    V27/V28 compare against a moving target.

    Every item is a dict::

        {"lr": (1,h,w) float32 tensor, "gt": (1,2h,2w) float32 tensor,
         "name": str, "split": "train"|"val", "synthetic": bool, "index": int}

    ``name`` is always paired with ``split`` because train and test basenames collide
    (docs/SPEC_ADDENDUM.md section 6).

    Neither ``lr`` nor ``gt`` is ever clipped (SPEC F5).
    """

    def __init__(
        self,
        data_root: str | Path | None,
        cfg: Mapping[str, Any] | DataConfig | None = None,
        split: str = "train",
        names: Sequence[str] | None = None,
        split_val_path: str | Path | None = None,
        arrays: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cfg = cfg if isinstance(cfg, DataConfig) else DataConfig.from_mapping(cfg)
        self.split = split
        self._rng_obj: np.random.Generator | None = None
        self._epoch = 0
        self._mm_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        if arrays is not None:
            lrs, gts = arrays
            if len(lrs) != len(gts):
                raise ValueError(f"arrays disagree: {len(lrs)} LR vs {len(gts)} GT")
            if not lrs:
                raise ValueError("arrays must contain at least one LR/GT pair")
            self.data_root = None
            self.names = list(names) if names is not None else [f"mem_{i:06d}" for i in range(len(lrs))]
            if len(self.names) != len(lrs):
                raise ValueError(f"names disagree: {len(self.names)} names for {len(lrs)} pairs")
            self._validate_names()
            self._lr: list[np.ndarray] | None = [np.asarray(a) for a in lrs]
            self._gt: list[np.ndarray] | None = [np.asarray(a) for a in gts]
            self.preloaded = True
            self.paths: list[tuple[Path, Path]] = []
        else:
            if data_root is None:
                raise ValueError("data_root is required unless arrays= is given")
            self.data_root = Path(data_root)
            if names is not None:
                self.names = list(names)
            else:
                train, val = train_val_names(self.data_root, split_val_path)
                self.names = train if split == "train" else val
            if not self.names:
                raise ValueError(f"{split} split contains no paired filenames")
            self._validate_names()
            gt_dir = self.data_root / TRAIN_GT[0] / TRAIN_GT[1]
            lr_dir = self.data_root / TRAIN_LR[0] / TRAIN_LR[1]
            self.paths = [(lr_dir / n, gt_dir / n) for n in self.names]
            missing = [str(p) for pair in self.paths for p in pair if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} files missing, first: {missing[:3]}")
            self._lr = self._gt = None
            self.preloaded = False
            if self._should_preload():
                self._preload()

        self._validate_shapes()

    # -- construction helpers ---------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        lr: Sequence[np.ndarray],
        gt: Sequence[np.ndarray],
        cfg: Mapping[str, Any] | DataConfig | None = None,
        split: str = "train",
        names: Sequence[str] | None = None,
    ) -> "PairedRestorationDataset":
        """In-memory dataset, for self-tests. Touches no disk and no dataset root."""
        return cls(None, cfg=cfg, split=split, names=names, arrays=(lr, gt))

    def _estimated_bytes(self) -> int:
        one = self.paths[0]
        lr0 = np.load(one[0], allow_pickle=False, mmap_mode="r")
        gt0 = np.load(one[1], allow_pickle=False, mmap_mode="r")
        per = int(lr0.size + gt0.size) * 4
        return per * len(self.paths)

    def _should_preload(self) -> bool:
        want = self.cfg.preload
        if want is False:
            return False
        est = self._estimated_bytes()
        if want is True:
            return True
        avail = _available_ram_bytes()
        self.preload_probe = {"estimated_bytes": est, "available_bytes": avail}
        if avail is None:
            return est < 1_500_000_000        # unknown machine: only preload if clearly small
        return est * self.cfg.preload_headroom < avail

    def _preload(self) -> None:
        """Eager-load into RAM (SPEC 6.2). Loads LR first so a failure costs less."""
        lrs = [np.load(p, allow_pickle=False).astype(np.float32, copy=False) for p, _ in self.paths]
        gts = [np.load(p, allow_pickle=False).astype(np.float32, copy=False) for _, p in self.paths]
        self._lr, self._gt = lrs, gts
        self.preloaded = True

    def _validate_names(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("dataset filenames contain duplicates")
        unsafe = [n for n in self.names if Path(n).name != n or not n]
        if unsafe:
            raise ValueError(f"dataset filenames must be plain basenames, got {unsafe[:3]}")

    def _validation_pair(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._lr is not None and self._gt is not None:
            return self._lr[k], self._gt[k]
        lp, gp = self.paths[k]
        return (
            np.load(lp, allow_pickle=False, mmap_mode="r"),
            np.load(gp, allow_pickle=False, mmap_mode="r"),
        )

    def _validate_shapes(self) -> None:
        s = self.cfg.scale
        p = self.cfg.lr_patch
        for k, name in enumerate(self.names):
            lr, gt = self._validation_pair(k)
            if lr.dtype != np.float32 or gt.dtype != np.float32:
                raise ValueError(
                    f"{name}: expected float32 LR/GT, got {lr.dtype}/{gt.dtype}"
                )
            if lr.ndim != 2 or gt.ndim != 2:
                raise ValueError(f"{name}: expected 2-D arrays, got LR {lr.shape} and GT {gt.shape}")
            if lr.size == 0 or gt.size == 0:
                raise ValueError(f"{name}: LR/GT arrays must be non-empty")
            if gt.shape != (s * lr.shape[0], s * lr.shape[1]):
                raise ValueError(f"{name}: GT {gt.shape} is not {s}x LR {lr.shape}")
            if not np.isfinite(lr).all() or not np.isfinite(gt).all():
                raise ValueError(f"{name}: LR/GT contains NaN or Inf")
            if self.data_root is not None and gt.size and (
                float(gt.min()) < 0.0 or float(gt.max()) > 1.0
            ):
                raise ValueError(f"{name}: GT values must be within [0,1]")
            if self.split == "train" and (lr.shape[0] < p or lr.shape[1] < p):
                raise ValueError(
                    f"{name}: lr_patch={p} exceeds LR image size {lr.shape}; lower data.lr_patch"
                )

    # -- data access ------------------------------------------------------------------
    def _pair(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(LR, GT) for image ``k``: in-RAM if preloaded, else memory-mapped."""
        if self._lr is not None and self._gt is not None:
            return self._lr[k], self._gt[k]
        hit = self._mm_cache.get(k)
        if hit is None:
            lp, gp = self.paths[k]
            hit = (
                np.load(lp, allow_pickle=False, mmap_mode="r"),
                np.load(gp, allow_pickle=False, mmap_mode="r"),
            )
            self._mm_cache[k] = hit
        return hit

    def rng(self) -> np.random.Generator:
        """Per-worker, per-epoch RNG, seeded from the config seed.

        Seeded from ``(seed, worker_id, epoch)`` so a run is reproducible for a fixed worker
        count and sampler order (V34), and two workers never draw the same crops.
        """
        if self._rng_obj is None:
            info = get_worker_info()
            wid = 0 if info is None else int(info.id)
            self._rng_obj = np.random.default_rng([self.cfg.seed, wid, self._epoch])
        return self._rng_obj

    def set_epoch(self, epoch: int) -> None:
        """Re-seed for a new epoch: same run gives the same stream, different epochs differ."""
        self._epoch = int(epoch)
        self._rng_obj = None

    def rng_for_index(self, idx: int) -> np.random.Generator:
        """Stateless sample RNG, stable across worker count and mid-epoch resume.

        A crop is a pure function of seed, epoch, and dataset index. DataLoader prefetching can
        therefore never advance hidden worker RNG state beyond the checkpointed optimizer step.
        """
        return np.random.default_rng([self.cfg.seed, self._epoch, int(idx)])

    def __len__(self) -> int:
        if self.split == "val":
            return len(self.names)
        return len(self.names) * max(1, self.cfg.crops_per_image)

    # -- the sample ------------------------------------------------------------------
    def _synth_lr_patch(
        self, gt_img: np.ndarray, i: int, j: int, patch: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Synthesise the LR patch for GT crop ``(2i,2j)`` from ``src/degrade.py``.

        The GT region is taken with a 2-GT-pixel margin on every side so that every LR pixel
        that is kept has the full 4-tap kernel support it would have had in a whole-image
        degradation; the margin is then discarded. Without it the patch border would carry a
        boundary artefact the real data does not have.
        """
        m = 2
        region = _crop_pad(gt_img, 2 * i - m, 2 * j - m, 2 * patch + 2 * m, 2 * patch + 2 * m)
        lr = degrade(region, rng, cfg=self.cfg.degrade)
        off = m // 2
        return lr[off : off + patch, off : off + patch]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        k = int(idx) % len(self.names)
        lr_img, gt_img = self._pair(k)

        if self.split == "val":
            # copy=True: a memory-mapped array is read-only and torch.from_numpy would hand
            # the trainer a non-writable tensor.
            lr_patch = np.array(lr_img, dtype=np.float32, copy=True)
            gt_patch = np.array(gt_img, dtype=np.float32, copy=True)
            return {
                "lr": torch.from_numpy(lr_patch[None]),
                "gt": torch.from_numpy(gt_patch[None]),
                "name": self.names[k],
                "split": "val",
                "synthetic": False,
                "index": k,
            }

        rng = self.rng_for_index(idx)
        cfg = self.cfg
        p, s = cfg.lr_patch, cfg.scale
        i = int(rng.integers(0, lr_img.shape[0] - p + 1))
        j = int(rng.integers(0, lr_img.shape[1] - p + 1))

        # GT crop first: it is the target in every branch, real or synthetic.
        _, gt_patch = paired_crop(lr_img, gt_img, i, j, p, s)
        gt_patch = np.array(gt_patch, dtype=np.float32, copy=True)

        synthetic = bool(rng.random() < cfg.synth_ratio)
        if synthetic:
            lr_patch = self._synth_lr_patch(gt_img, i, j, p, rng)
        else:
            lr_patch = np.array(lr_img[i : i + p, j : j + p], dtype=np.float32, copy=True)

        if cfg.cutblur_prob > 0.0 and rng.random() < cfg.cutblur_prob:
            # CutBlur (Yoo et al., CVPR 2020) in the LR domain. The paper mixes an
            # upsampled-LR rectangle into the HR image, which needs input and target at one
            # resolution; this network takes LR in and emits 2x, so the same idea is applied
            # between the *degraded* and the *clean* LR of the same crop -- exactly aligned,
            # geometry untouched, and the target stays the full GT. The model therefore has
            # to learn where and how much to restore instead of sharpening uniformly.
            clean = conv_downsample_2x(gt_patch)
            if rng.random() < 0.5:
                lr_patch = cutblur_paste(lr_patch, clean, rng, cfg.cutblur_alpha)
            else:
                lr_patch = cutblur_paste(clean, lr_patch, rng, cfg.cutblur_alpha)

        if cfg.dihedral:
            kk = int(rng.integers(0, 8))
            lr_patch = dihedral(lr_patch, kk)     # identical k on both members of the pair --
            gt_patch = dihedral(gt_patch, kk)     # anything else silently breaks alignment

        return {
            "lr": torch.from_numpy(np.ascontiguousarray(lr_patch, dtype=np.float32)[None]),
            "gt": torch.from_numpy(np.ascontiguousarray(gt_patch, dtype=np.float32)[None]),
            "name": self.names[k],
            "split": "train",
            "synthetic": synthetic,
            "index": k,
        }


def build_datasets(
    data_root: str | Path,
    cfg: Mapping[str, Any] | DataConfig | None = None,
    split_val_path: str | Path | None = None,
) -> tuple[PairedRestorationDataset, PairedRestorationDataset]:
    """``(train_ds, val_ds)`` from the committed split. The intersection assertion runs here."""
    dcfg = cfg if isinstance(cfg, DataConfig) else DataConfig.from_mapping(cfg)
    train_names, val_names = train_val_names(data_root, split_val_path)
    train = PairedRestorationDataset(data_root, dcfg, "train", names=train_names)
    val = PairedRestorationDataset(data_root, dcfg, "val", names=val_names)
    return train, val


# ======================================================================================
# V26 -- PAIRED-CROP ALIGNMENT SELF-TEST
# ======================================================================================
def _index_code_images(h: int = 128, w: int = 128, scale: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """LR[i,j] = i*w + j and GT[y,x] = y*(scale*w) + x -- every pixel carries its own address.

    All codes are integers below 2**24, so float32 stores them exactly and a crop origin can
    be read straight back out of the returned patch.
    """
    lr = (np.arange(h)[:, None] * w + np.arange(w)[None, :]).astype(np.float32)
    H, W = scale * h, scale * w
    gt = (np.arange(H)[:, None] * W + np.arange(W)[None, :]).astype(np.float32)
    return lr, gt


def selftest_paired_crop(
    n_crops: int = 256,
    patch: int = 64,
    seed: int = 20260815,
    size: int = 128,
    scale: int = 2,
) -> dict[str, Any]:
    """V26: prove GT crops are exactly the 2x region of the LR crop, before and after aug.

    Five independent assertions, all on synthetic images -- no dataset access:

    1. ``paired_crop`` marker test. A single bright LR pixel and its 2x2 GT counterpart:
       wherever the marker lands in the LR patch at ``(r,c)``, the GT patch must carry the
       full 2x2 block at ``(2r,2c)`` and nothing anywhere else.
    2. Address decode through ``__getitem__``. Every pixel carries its own index, so the LR
       and GT crop origins are read back from the returned patches and must satisfy
       ``gt_origin == 2 * lr_origin`` exactly, for the whole patch, not just the corner.
    3. Dihedral invariance. With ``GT = kron(LR, ones(2,2))``, an aligned pair must satisfy
       ``gt_patch == kron(lr_patch)`` for every one of the 8 orientations -- the check that
       catches an augmentation applied to one member of the pair only.
    4. CutBlur leaves the target alone and the geometry intact.
    5. The synthetic branch is aligned (peak correlation at shift (0,0), SPEC 5.2's own test)
       and is **not clipped** -- values outside [0,1] must survive.
    """
    rng = np.random.default_rng(seed)
    res: dict[str, Any] = {"check": "V26", "n_crops": int(n_crops), "patch": int(patch),
                           "scale": int(scale), "failures": []}
    fail: list[str] = res["failures"]

    # --- 1. marker test on paired_crop itself -----------------------------------------
    my, mx = size // 2, size // 2
    lr_m = np.zeros((size, size), dtype=np.float32)
    lr_m[my, mx] = 1.0
    gt_m = np.zeros((scale * size, scale * size), dtype=np.float32)
    gt_m[scale * my : scale * my + scale, scale * mx : scale * mx + scale] = 1.0
    hits = 0
    for _ in range(n_crops):
        i = int(rng.integers(0, size - patch + 1))
        j = int(rng.integers(0, size - patch + 1))
        lp, gp = paired_crop(lr_m, gt_m, i, j, patch, scale)
        inside = (i <= my < i + patch) and (j <= mx < j + patch)
        if not inside:
            if gp.sum() != 0.0 or lp.sum() != 0.0:
                fail.append(f"marker leaked into crop ({i},{j}) that excludes it")
            continue
        hits += 1
        r, c = int(np.argmax(lp) // patch), int(np.argmax(lp) % patch)
        if (r, c) != (my - i, mx - j):
            fail.append(f"LR marker at ({r},{c}), expected ({my - i},{mx - j})")
        block = gp[scale * r : scale * r + scale, scale * c : scale * c + scale]
        if not np.array_equal(block, np.ones((scale, scale), dtype=np.float32)):
            fail.append(f"GT marker block missing at ({scale * r},{scale * c}) for crop ({i},{j})")
        if gp.sum() != float(scale * scale):
            fail.append(f"GT patch has stray energy for crop ({i},{j}): sum={gp.sum()}")
    res["marker_crops_containing_marker"] = hits

    # --- 2. address decode through __getitem__ (no aug, real pairs) --------------------
    lr_c, gt_c = _index_code_images(size, size, scale)
    plain = DataConfig(lr_patch=patch, scale=scale, synth_ratio=0.0, cutblur_prob=0.0,
                       dihedral=False, preload=True, seed=seed)
    ds_plain = PairedRestorationDataset.from_arrays([lr_c] * 4, [gt_c] * 4, plain)
    W, GW = size, scale * size
    checked = 0
    for n in range(min(n_crops, 4 * 32)):
        item = ds_plain[n % len(ds_plain)]
        lp = item["lr"][0].numpy()
        gp = item["gt"][0].numpy()
        code = float(lp[0, 0])
        i, j = int(code) // W, int(code) % W
        gcode = float(gp[0, 0])
        gy, gx = int(gcode) // GW, int(gcode) % GW
        if (gy, gx) != (scale * i, scale * j):
            fail.append(f"GT origin ({gy},{gx}) != {scale}x LR origin ({i},{j})")
        exp_lr = lr_c[i : i + patch, j : j + patch]
        exp_gt = gt_c[scale * i : scale * i + scale * patch, scale * j : scale * j + scale * patch]
        if not np.array_equal(lp, exp_lr) or not np.array_equal(gp, exp_gt):
            fail.append(f"patch content mismatch at LR origin ({i},{j})")
        if lp.shape != (patch, patch) or gp.shape != (scale * patch, scale * patch):
            fail.append(f"patch shapes {lp.shape}/{gp.shape} wrong")
        checked += 1
    res["address_decode_crops"] = checked

    # --- 3. dihedral invariance: GT must stay kron(LR) --------------------------------
    lr_k = rng.uniform(-0.3, 2.2, size=(size, size)).astype(np.float32)   # out-of-[0,1] on purpose
    gt_k = np.kron(lr_k, np.ones((scale, scale), dtype=np.float32))
    aug = DataConfig(lr_patch=patch, scale=scale, synth_ratio=0.0, cutblur_prob=0.0,
                     dihedral=True, preload=True, seed=seed + 1)
    ds_aug = PairedRestorationDataset.from_arrays([lr_k] * 8, [gt_k] * 8, aug)
    kron_checked = 0
    for n in range(n_crops):
        item = ds_aug[n % len(ds_aug)]
        lp = item["lr"][0].numpy()
        gp = item["gt"][0].numpy()
        if not np.array_equal(gp, np.kron(lp, np.ones((scale, scale), dtype=np.float32))):
            fail.append("dihedral augmentation broke pair alignment")
            break
        kron_checked += 1
    res["dihedral_kron_crops"] = kron_checked

    # the 8 orientations must be 8 genuinely distinct maps, on an asymmetric test tile
    tile = np.arange(6, dtype=np.float32).reshape(2, 3)
    orientations = {dihedral(tile, kk).tobytes() + bytes(dihedral(tile, kk).shape) for kk in range(8)}
    res["dihedral_distinct_orientations"] = len(orientations)
    if len(orientations) != 8:
        fail.append(f"dihedral group has {len(orientations)} distinct orientations, expected 8")

    # --- 4. CutBlur: target untouched, geometry untouched ------------------------------
    cb = DataConfig(lr_patch=patch, scale=scale, synth_ratio=0.0, cutblur_prob=1.0,
                    dihedral=False, preload=True, seed=seed + 2)
    ds_cb = PairedRestorationDataset.from_arrays([lr_c] * 4, [gt_c] * 4, cb)
    cb_checked = 0
    for n in range(min(64, n_crops)):
        item = ds_cb[n % len(ds_cb)]
        gp = item["gt"][0].numpy()
        gcode = float(gp[0, 0])
        gy, gx = int(gcode) // GW, int(gcode) % GW
        exp_gt = gt_c[gy : gy + scale * patch, gx : gx + scale * patch]
        if not np.array_equal(gp, exp_gt):
            fail.append("CutBlur modified the GT target")
        if gy % scale or gx % scale:
            fail.append(f"CutBlur sample has odd GT origin ({gy},{gx})")
        if item["lr"][0].numpy().shape != (patch, patch):
            fail.append("CutBlur changed the LR patch shape")
        cb_checked += 1
    res["cutblur_crops"] = cb_checked

    # --- 5. synthetic branch: aligned, and NOT clipped ---------------------------------
    # A structured, high-contrast test image: oriented bars plus a low-pass random field,
    # min-max normalised exactly as the real GT is (docs/SPEC_ADDENDUM.md section 3).
    # Deliberately fine-grained: a 1-LR-pixel shift must visibly destroy the correlation, so
    # the pattern carries energy near the LR Nyquist limit rather than being smooth.
    N = scale * size
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    field = rng.random((N, N))
    padf = np.pad(field, 1, mode="edge")
    field = sum(padf[a : a + N, b : b + N] for a in range(3) for b in range(3)) / 9.0
    img = (
        np.sin(2.0 * np.pi * (24.0 * xx + 16.0 * yy) / N + 0.7)
        + 0.6 * np.cos(2.0 * np.pi * (30.0 * yy - 12.0 * xx) / N)
        + 2.0 * (field - field.mean())
    )
    img = (img - img.min()) / (img.max() - img.min())
    gt_s = img.astype(np.float32)
    lr_s = conv_downsample_2x(gt_s)
    syn = DataConfig(lr_patch=patch, scale=scale, synth_ratio=1.0, cutblur_prob=0.0,
                     dihedral=False, preload=True, seed=seed + 3)
    ds_syn = PairedRestorationDataset.from_arrays([lr_s] * 4, [gt_s] * 4, syn)

    # SPEC 5.2's own alignment criterion: the correlation peak must sit at shift (0,0).
    shifts = [(0, 0), (0, 1), (1, 0), (1, 1), (0, -1), (-1, 0)]
    sums = {s: 0.0 for s in shifts}
    above1 = below0 = n_syn = n_used = 0
    for n in range(min(64, n_crops)):
        item = ds_syn[n % len(ds_syn)]
        lp = item["lr"][0].numpy()
        gp = item["gt"][0].numpy()
        if not item["synthetic"]:
            fail.append("synth_ratio=1.0 produced a real pair")
            continue
        clean = conv_downsample_2x(gp)
        m = 2
        a = lp[m:-m, m:-m].ravel()
        for (dy, dx) in shifts:
            b = clean[m + dy : lp.shape[0] - m + dy, m + dx : lp.shape[1] - m + dx].ravel()
            sums[(dy, dx)] += float(np.corrcoef(a, b)[0, 1])
        above1 += int((lp > 1.0).sum())
        below0 += int((lp < 0.0).sum())
        n_syn += lp.size
        n_used += 1
    corr = {f"{dy},{dx}": sums[(dy, dx)] / max(1, n_used) for (dy, dx) in shifts}
    res["synth_corr_by_shift"] = corr
    res["synth_frac_above_1"] = above1 / n_syn if n_syn else float("nan")
    res["synth_frac_below_0"] = below0 / n_syn if n_syn else float("nan")

    best = max(corr, key=lambda k: corr[k])
    runner = max((k for k in corr if k != "0,0"), key=lambda k: corr[k])
    res["synth_corr_peak_shift"] = best
    if best != "0,0":
        fail.append(f"synthetic LR is misaligned: correlation peaks at shift ({best}), not (0,0)")
    if corr["0,0"] - corr[runner] < 0.05:
        fail.append(
            "synthetic LR alignment is not decisive: corr(0,0)=%.4f vs best other shift "
            "(%s)=%.4f" % (corr["0,0"], runner, corr[runner])
        )
    # Absolute floor, not 1.0 on purpose: the measured noise is large (residual std 0.0901
    # against a GT std of ~0.22), so even a perfectly aligned synthetic LR correlates ~0.92
    # with the clean downsample of its own GT patch. A floor near 1.0 would only be
    # satisfiable by a simulator that under-noises.
    if corr["0,0"] < 0.85:
        fail.append(f"synthetic LR correlates only {corr['0,0']:.4f} with the clean downsample "
                    "of its own GT patch")
    if above1 == 0 and below0 == 0:
        fail.append("synthetic LR never leaves [0,1] -- it looks clipped, which SPEC F5 forbids")

    res["pass"] = not fail
    return res


# ======================================================================================
# CLI
# ======================================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Data pipeline self-checks (V26, V29).")
    ap.add_argument("--selftest", action="store_true", help="V26 paired-crop alignment")
    ap.add_argument("--check-split", action="store_true", help="V29 split integrity")
    ap.add_argument("--data-root", default=None, help="dataset root (optional for --check-split)")
    ap.add_argument("--n-crops", type=int, default=256)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if not (args.selftest or args.check_split):
        args.selftest = args.check_split = True

    out: dict[str, Any] = {}
    ok = True
    if args.selftest:
        r = selftest_paired_crop(n_crops=args.n_crops, patch=args.patch)
        out["V26"] = r
        ok = ok and bool(r["pass"])
    if args.check_split:
        root = args.data_root
        if root is None and (os.environ.get("KLA_DATA_ROOT")
                             or (repo_root() / "docs" / "DATA_LOCATION.md").exists()):
            try:
                root = str(resolve_data_root(None))
            except SystemExit:
                root = None
        r = check_split_integrity(root)
        out["V29"] = r
        ok = ok and bool(r["pass"])

    sys.stdout.write(json.dumps(out, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
