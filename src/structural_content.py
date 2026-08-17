"""Procedural structural content generators, reproducing the proxy-OOD set's own documented
recipe (`docs/dataset_findings.md`, "Generation method" section) so it can be reused at
training scale, not just as a 40-image evaluation-only fixture.

Plan Phase 3 (real-SEM OOD investigation, `docs/decisions.md` D68/D69): the gap was found to
be content-driven, not degradation-coverage. This module lets training mix in procedural
structural content -- gratings, contact-hole grids, checkerboard, circuit traces, sharp-edge
shapes -- alongside the real natural-photo GT, at a configurable ratio
(`DataConfig.structural_content_ratio`, `src/dataset.py`). This is F15-permitted synthetic
data derived entirely from a generator, not a foreign dataset: no licence surface, no
leakage risk.

Every function returns a float64 array in [0, 1] BEFORE the final per-image min-max
normalisation step (`normalise01`), at the exact recipe documented for the proxy-OOD set --
same params, same distributions, same 3x3 box blur. Nothing here fits or reads real content
statistics; it is a fixed, disclosed procedural recipe, not derived from the training set.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalise01",
    "line_space_grating",
    "contact_hole_grid",
    "checkerboard",
    "circuit_traces",
    "sharp_edge_shapes",
    "CATEGORIES",
    "random_structural_image",
]


def _box_blur3(a: np.ndarray) -> np.ndarray:
    """3x3 box blur via cumulative-sum trick, edge-replicated -- matches the proxy-OOD
    recipe's own "3x3 box-blurred" step for every category."""
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out / 9.0


def normalise01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def line_space_grating(size: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    pitch = rng.uniform(4.0, 28.0)
    duty = rng.uniform(0.35, 0.65)
    angle = float(rng.choice([0.0, 30.0, 45.0, 90.0])) + rng.uniform(-3.0, 3.0)
    theta = np.deg2rad(angle)
    phase = ((x * np.cos(theta) + y * np.sin(theta)) % pitch) / pitch
    a = (phase < duty).astype(np.float64)
    return _box_blur3(a)


def contact_hole_grid(size: int, rng: np.random.Generator) -> np.ndarray:
    pitch = rng.uniform(10.0, 32.0)
    radius = rng.uniform(0.18, 0.38) * pitch
    jitter = rng.uniform(0.0, 0.6)
    a = np.zeros((size, size), dtype=np.float64)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    n = int(size / pitch) + 3
    for gy in range(-1, n):
        for gx in range(-1, n):
            cy = gy * pitch + rng.uniform(-jitter, jitter) * pitch
            cx = gx * pitch + rng.uniform(-jitter, jitter) * pitch
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            a = np.maximum(a, (d < radius).astype(np.float64))
    return _box_blur3(a)


def checkerboard(size: int, rng: np.random.Generator) -> np.ndarray:
    t = int(round(rng.uniform(4.0, 24.0)))
    t = max(1, t)
    y, x = np.mgrid[0:size, 0:size]
    a = ((y // t + x // t) % 2).astype(np.float64)
    return _box_blur3(a)


def circuit_traces(size: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((size, size), dtype=np.float64)
    n_traces = int(rng.integers(8, 19))
    for _ in range(n_traces):
        width = int(rng.integers(2, 6))
        horizontal = bool(rng.random() < 0.5)
        pos = int(rng.integers(0, size))
        start = int(rng.integers(0, size))
        length = int(rng.integers(size // 8, size // 2))
        end = min(size, start + length)
        half = width // 2
        if horizontal:
            lo, hi = max(0, pos - half), min(size, pos + half + 1)
            a[lo:hi, start:end] = 1.0
        else:
            lo, hi = max(0, pos - half), min(size, pos + half + 1)
            a[start:end, lo:hi] = 1.0
    n_pads = int(rng.integers(6, 15))
    for _ in range(n_pads):
        s = int(rng.integers(6, 17))
        y0 = int(rng.integers(0, max(1, size - s)))
        x0 = int(rng.integers(0, max(1, size - s)))
        a[y0:y0 + s, x0:x0 + s] = 1.0
    return _box_blur3(a)


def sharp_edge_shapes(size: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((size, size), dtype=np.float64)
    n = int(rng.integers(10, 26))
    for _ in range(n):
        level = rng.uniform(0.2, 1.0)
        h = int(rng.integers(size // 16, size // 3))
        w = int(rng.integers(size // 16, size // 3))
        y0 = int(rng.integers(0, max(1, size - h)))
        x0 = int(rng.integers(0, max(1, size - w)))
        a[y0:y0 + h, x0:x0 + w] = np.maximum(a[y0:y0 + h, x0:x0 + w], level)
    return _box_blur3(a)


CATEGORIES = {
    "line_space_grating": line_space_grating,
    "contact_hole_grid": contact_hole_grid,
    "checkerboard": checkerboard,
    "circuit_traces": circuit_traces,
    "sharp_edge_shapes": sharp_edge_shapes,
}


def random_structural_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Pick one of the 5 documented categories uniformly at random, generate at `size`,
    normalise to exactly [0,1] (matches the real-GT U1 convention)."""
    name = rng.choice(list(CATEGORIES))
    raw = CATEGORIES[str(name)](size, rng)
    return normalise01(raw).astype(np.float32)
