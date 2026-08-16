"""Seeding, EMA, checkpoint IO, the experiment ledger and small training helpers.

Checkpoint dict contract (V35): ``model, ema, config, iter, metrics, git``.
``build_model(ckpt["config"])`` must load both ``ckpt["model"]`` and ``ckpt["ema"]`` with
``strict=True``, and the file must survive ``torch.load(..., weights_only=True)``.  That
second requirement is why everything written here is a **plain Python / tensor type**: no
custom classes, no lambdas, no ``numpy`` scalars, no ``pathlib.Path`` objects.

Determinism (V34/V44).  ``seed_everything`` seeds ``random``, ``numpy``, ``torch`` and
``torch.cuda``.  ``configure_backends`` decides the cuDNN trade-off and is called with
``deterministic=True`` for smoke/overfit runs; see its docstring for the measured cost.

Owner: trainer.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch

__all__ = [
    "EMA",
    "LEDGER_COLUMNS",
    "append_experiment",
    "capture_source_provenance",
    "capture_rng_state",
    "configure_backends",
    "cosine_warmup_lr",
    "deep_update",
    "format_hms",
    "git_sha",
    "load_config",
    "repo_root",
    "resolve_device",
    "restore_rng_state",
    "save_checkpoint",
    "seed_everything",
    "to_plain",
    "update_checkpoint_metrics",
]


def repo_root() -> Path:
    """Repo root resolved from this file, never from CWD (CLAUDE.md PD5)."""
    return Path(__file__).resolve().parents[1]


# ======================================================================================
# SEEDING  (V44)
# ======================================================================================
def seed_everything(seed: int) -> int:
    """Seed ``random``, ``numpy``, ``torch`` and ``torch.cuda`` from one integer (V44).

    ``PYTHONHASHSEED`` is also set for completeness, though it only takes effect for
    child processes -- the current interpreter's hash seed is fixed at start-up.  Dataloader
    workers inherit it, which is why it is set here rather than only documented.

    Returns the seed, so a caller can log exactly what was used.
    """
    s = int(seed)
    if not 0 <= s < 2**31:
        raise ValueError(f"seed must fit in a positive int32, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG in weights-only-safe tensor/plain form."""
    np_state = np.random.get_state()
    return {
        "python": to_plain(random.getstate()),
        "numpy": {
            "bit_generator": str(np_state[0]),
            "state": np.asarray(np_state[1], dtype=np.uint32).tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": ([v.cpu() for v in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
    }


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(v) for v in value)
    return value


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(_nested_tuple(state["python"]))
    ns = state["numpy"]
    np.random.set_state((
        str(ns["bit_generator"]),
        np.asarray(ns["state"], dtype=np.uint32),
        int(ns["pos"]),
        int(ns["has_gauss"]),
        float(ns["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].to("cpu", dtype=torch.uint8))
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                f"checkpoint has {len(cuda_states)} CUDA RNG states, this host has "
                f"{torch.cuda.device_count()} devices"
            )
        torch.cuda.set_rng_state_all([v.to("cpu", dtype=torch.uint8) for v in cuda_states])


def configure_backends(deterministic: bool, allow_tf32: bool = True) -> dict[str, Any]:
    """Set the cuDNN/matmul knobs and report what was chosen.

    The trade-off, stated explicitly because SPEC 9 asks for it to be documented:

    * ``deterministic=True`` -> ``cudnn.deterministic=True``, ``cudnn.benchmark=False``.
      cuDNN may then only pick reproducible convolution algorithms and cannot autotune.
      This is the mode used for the ``--smoke`` reproducibility check (V34) and for the
      ``--overfit`` gate (V25), where a bit-identical result is the whole point.
    * ``deterministic=False`` -> ``cudnn.benchmark=True``.  cuDNN autotunes per input shape.
      Patch shape is constant during training so autotuning pays for itself once, and the
      run stays *seed-reproducible in distribution* but is not guaranteed bit-identical.

    TF32 is left on for matmuls: this network is convolution-bound and bandwidth-bound
    (docs/SPEC_ADDENDUM.md section 9), and TF32 does not affect the bf16 autocast path.
    """
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except AttributeError:  # pragma: no cover - very old torch
        pass
    if deterministic:
        # cuBLAS needs a fixed workspace for reproducible reductions.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32": bool(allow_tf32),
    }


def resolve_device(spec: str | None = None) -> torch.device:
    """``cuda`` when available unless overridden. Never guesses a device index."""
    if spec:
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# PROVENANCE
# ======================================================================================
def git_sha(short: bool = False) -> str:
    """HEAD SHA of the repo this file lives in, or ``"unknown"`` outside a work tree.

    Returned as a plain ``str`` so it survives ``torch.load(weights_only=True)``.  A
    ``-dirty`` suffix is appended when the work tree has uncommitted changes, because a
    checkpoint produced from a dirty tree is not reproducible from the SHA alone.
    """
    root = repo_root()
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if not short:
            args = ["git", "rev-parse", "HEAD"]
        sha = subprocess.run(args, cwd=str(root), capture_output=True, text=True,
                             timeout=15, check=False)
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        if not out:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                               capture_output=True, text=True, timeout=30, check=False)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def capture_source_provenance(paths: Sequence[str | os.PathLike[str]]) -> dict[str, Any]:
    """Record exact training-source identities and preserve dirty relevant file contents.

    A commit plus ``-dirty`` is not reconstructable. For each requested source/config file,
    this records its working-tree SHA and HEAD blob identity. Files that differ from HEAD (or
    are untracked/external) additionally carry their complete UTF-8 content. A unified patch is
    included for human review; the full dirty contents are the authoritative reconstruction.
    """
    root = repo_root().resolve()
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        head = "unknown"

    records: list[dict[str, Any]] = []
    repo_paths: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        content = path.read_bytes()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        record: dict[str, Any] = {
            "path": rel,
            "working_sha256": hashlib.sha256(content).hexdigest(),
        }
        if path.is_relative_to(root) and head != "unknown":
            repo_paths.append(rel)
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel], cwd=root,
                capture_output=True, text=True, timeout=15, check=False,
            ).returncode == 0
            record["tracked"] = tracked
            head_content: bytes | None = None
            if tracked:
                try:
                    head_content = subprocess.check_output(
                        ["git", "show", f"{head}:{rel}"], cwd=root, timeout=15
                    )
                    record["head_blob"] = subprocess.check_output(
                        ["git", "rev-parse", f"{head}:{rel}"], cwd=root,
                        text=True, timeout=15,
                    ).strip()
                    record["head_sha256"] = hashlib.sha256(head_content).hexdigest()
                except subprocess.SubprocessError:
                    head_content = None
            if head_content != content:
                record["dirty_content_utf8"] = content.decode("utf-8")
        else:
            record["tracked"] = False
            record["dirty_content_utf8"] = content.decode("utf-8")
        records.append(record)

    patch = ""
    if repo_paths and head != "unknown":
        proc = subprocess.run(
            ["git", "diff", "--binary", head, "--", *repo_paths], cwd=root,
            capture_output=True, text=True, timeout=30, check=False,
        )
        if proc.returncode == 0:
            patch = proc.stdout
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
        timeout=30, check=False,
    )
    status_text = status.stdout if status.returncode == 0 else "unavailable"
    return {
        "schema_version": 1,
        "git_commit": head,
        "repository_worktree_clean": not bool(status_text.strip()),
        "git_status_porcelain": status_text,
        "relevant_source_patch": patch,
        "source_files": records,
    }


# ======================================================================================
# CONFIG
# ======================================================================================
def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML training config into plain Python types.

    ``yaml`` is imported here and not at module scope of ``inference.py`` -- this module is
    never imported by the shipped inference script (CLAUDE.md STYLE / V23).
    """
    import yaml  # local: keeps the import off any timed path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise TypeError(f"{p} did not parse to a mapping (got {type(cfg).__name__})")
    return cfg


def deep_update(base: MutableMapping[str, Any], other: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``other`` into ``base`` in place and return ``base``."""
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def to_plain(obj: Any) -> Any:
    """Recursively convert to types ``torch.load(weights_only=True)` will accept.

    numpy scalars/arrays become Python scalars/lists, ``Path`` becomes ``str``, tuples
    become lists.  Anything already plain passes through untouched.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ======================================================================================
# EMA  -- the shipped weights (SPEC 9)
# ======================================================================================
class EMA:
    """Exponential moving average of the model's floating-point parameters and buffers.

    The shadow copy is kept in **fp32 on the training device** regardless of the autocast
    dtype: an EMA accumulated in bf16 loses most of its point, because ``1 - decay`` is
    1e-3 and bf16 carries ~3 decimal digits, so the update would round to nothing.

    Non-floating buffers (integer counters and the like) are copied verbatim rather than
    averaged, which keeps ``state_dict()`` key- and dtype-identical to the live model and
    is what makes ``load_state_dict(..., strict=True)`` work in V35.

    The decay ramps in as ``min(decay, (1 + n) / (10 + n))`` so the average is not dragged
    down by the random initialisation for the first few hundred steps.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"ema decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.num_updates = 0
        sd = model.state_dict()
        self.shadow: dict[str, torch.Tensor] = {}
        self.dtypes: dict[str, torch.dtype] = {}
        for k, v in sd.items():
            self.dtypes[k] = v.dtype
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone().float()
            else:
                self.shadow[k] = v.detach().clone()
        # Keys are split once so ``update`` can use the fused _foreach_ kernels. Measured on
        # an RTX 4060: a per-tensor Python loop over the ~200 NAFSR tensors costs ~30 ms per
        # optimiser step, which is ~15% of a 207 ms training step -- pure launch overhead.
        self._fp32_keys = [k for k, v in self.shadow.items()
                           if v.dtype == torch.float32 and self.dtypes[k] == torch.float32]
        _fast = set(self._fp32_keys)
        self._other_keys = [k for k in self.shadow if k not in _fast]

    def current_decay(self) -> float:
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = self.current_decay()
        sd = model.state_dict()

        if self._fp32_keys:
            dst = [self.shadow[k] for k in self._fp32_keys]
            src = [sd[k].detach() for k in self._fp32_keys]
            torch._foreach_mul_(dst, d)
            torch._foreach_add_(dst, src, alpha=1.0 - d)

        for k in self._other_keys:
            v = sd[k].detach()
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.float(), alpha=1.0 - d)
            else:
                s.copy_(v)

        for k, v in sd.items():  # a key that appeared after construction starts tracking now
            if k not in self.shadow:
                self.dtypes[k] = v.dtype
                self.shadow[k] = (v.detach().clone().float() if v.dtype.is_floating_point
                                  else v.detach().clone())
                self._other_keys.append(k)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """CPU, contiguous, original dtypes -- ready for ``save_checkpoint``."""
        return {
            k: v.detach().to("cpu", dtype=self.dtypes[k]).contiguous()
            for k, v in self.shadow.items()
        }

    def load_state_dict(self, sd: Mapping[str, torch.Tensor]) -> None:
        for k, v in sd.items():
            if k in self.shadow:
                tgt = self.shadow[k]
                tgt.copy_(v.to(tgt.device, dtype=tgt.dtype))

    def copy_to(self, model: torch.nn.Module) -> None:
        """Overwrite ``model``'s weights with the EMA ones (used for validation)."""
        model.load_state_dict(self.state_dict(), strict=True)


# ======================================================================================
# CHECKPOINTS  (V35)
# ======================================================================================
def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: torch.nn.Module,
    ema: "EMA | None",
    config: Mapping[str, Any],
    iteration: int,
    metrics: Mapping[str, Any],
    git: str | None = None,
    training_state: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write the V35 checkpoint dict atomically.

    Inference keys are ``model, ema, config, iter, metrics, git``. Resumable training
    checkpoints additionally carry ``training_state``; release checkpoints may carry a plain
    ``provenance`` block. All values remain compatible with ``weights_only=True``.

    The write goes to a temporary file in the destination directory and is then
    ``os.replace``d, so an interrupted save can never leave a truncated ``best.pt`` behind.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": _cpu_state_dict(model),
        "ema": ema.state_dict() if ema is not None else {},
        "config": to_plain(config),
        "iter": int(iteration),
        "metrics": to_plain(metrics),
        "git": str(git if git is not None else git_sha()),
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    if provenance is not None:
        payload["provenance"] = to_plain(provenance)
    return _atomic_torch_save(payload, out)


def _atomic_torch_save(payload: Mapping[str, Any], out: Path) -> Path:
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=str(out.parent))
    os.close(fd)
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return out


def update_checkpoint_metrics(path: str | os.PathLike[str],
                              metrics: Mapping[str, Any]) -> Path:
    """Replace only the ``metrics`` block of an existing checkpoint, in place.

    Used at the end of a run: the shipped weights must stay the **best** ones selected on
    held-out data, but the reported metrics are the full-split numbers measured afterwards.
    Rewriting the whole checkpoint from the live model would silently ship the *last*
    weights instead of the best ones, which is the classic way a training script quietly
    loses 0.2 dB.
    """
    p = Path(path)
    ck = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise TypeError(f"{p} does not hold a checkpoint dict")
    ck["metrics"] = to_plain(metrics)
    return _atomic_torch_save(ck, p)


# ======================================================================================
# SCHEDULE
# ======================================================================================
def cosine_warmup_lr(it: int, base_lr: float, min_lr: float, warmup_iters: int,
                     total_iters: int) -> float:
    """Linear warmup then cosine decay to ``min_lr`` (SPEC 9).

    ``it`` is 0-based.  Warmup is ``(it + 1) / warmup_iters`` so step 0 is never a zero
    learning rate (a zero first step wastes an iteration and makes step-0 loss traces
    misleading).
    """
    it = int(it)
    w = int(warmup_iters)
    t = max(int(total_iters), 1)
    if w > 0 and it < w:
        return float(base_lr) * float(it + 1) / float(w)
    p = (it - w) / max(1, t - w)
    p = min(max(p, 0.0), 1.0)
    return float(min_lr) + 0.5 * (float(base_lr) - float(min_lr)) * (1.0 + math.cos(math.pi * p))


def format_hms(seconds: float) -> str:
    s = int(round(float(seconds)))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ======================================================================================
# EXPERIMENT LEDGER  (V45, SPEC 9)
# ======================================================================================
#: Fixed column order for ``results/experiments.csv``.  ``structural_kind`` is present
#: because ``ssim`` and ``ms_ssim`` losses are NOT comparable across runs, so a ledger that
#: omitted it would be misleading (src/losses.py:structural_loss).
LEDGER_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp_utc",
    "git_sha",
    "config",
    "seed",
    "model",
    "params",
    "iters",
    "batch_size",
    "lr_patch",
    "structural_kind",
    "best_iter",
    "best_psnr",
    "best_ssim",
    "best_lpips",
    "val_n",
    "weights_used",
    "wall_clock_s",
    "wall_clock_hms",
    "checkpoint",
    "device",
    "notes",
)


def append_experiment(path: str | os.PathLike[str], row: Mapping[str, Any]) -> Path:
    """Append one run to ``results/experiments.csv``, writing the header if needed.

    Unknown keys are rejected rather than silently dropped: a metric that quietly failed to
    land in the ledger is exactly the provenance hole SPEC 9 exists to close.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(row) - set(LEDGER_COLUMNS))
    if unknown:
        raise ValueError(f"unknown ledger column(s): {unknown}; extend LEDGER_COLUMNS first")
    need_header = (not p.exists()) or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LEDGER_COLUMNS), lineterminator="\n")
        if need_header:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in LEDGER_COLUMNS})
    return p


# ======================================================================================
# SELF-TEST
# ======================================================================================
def _self_test() -> int:  # pragma: no cover - exercised manually and by train.py --selftest
    import json

    sys.path.insert(0, str(repo_root()))
    from src.model import build_model

    ok = True
    cfg = {"model": {"name": "NAFSR", "width": 8, "num_blocks": 2, "scale": 2}}
    m = build_model(cfg)
    ema = EMA(m, decay=0.99)
    for _ in range(5):
        with torch.no_grad():
            for prm in m.parameters():
                prm.add_(torch.randn_like(prm) * 0.01)
        ema.update(m)

    with tempfile.TemporaryDirectory() as td:
        ck = save_checkpoint(Path(td) / "t.pt", model=m, ema=ema, config=cfg, iteration=5,
                             metrics={"psnr": 1.0}, git="deadbeef")
        d = torch.load(ck, map_location="cpu", weights_only=True)
        for k in ("model", "ema", "config", "iter", "metrics", "git"):
            if k not in d:
                print(f"FAIL missing key {k}")
                ok = False
        build_model(d["config"]).load_state_dict(d["model"], strict=True)
        build_model(d["config"]).load_state_dict(d["ema"], strict=True)

        led = append_experiment(Path(td) / "e.csv", {"run_id": "a", "best_psnr": 1.0})
        append_experiment(led, {"run_id": "b", "best_psnr": 2.0})
        rows = led.read_text(encoding="utf-8").strip().splitlines()
        if len(rows) != 3:
            print(f"FAIL ledger rows {len(rows)}")
            ok = False

    lrs = [cosine_warmup_lr(i, 1e-3, 1e-6, 10, 100) for i in (0, 9, 10, 99)]
    if not (lrs[0] < lrs[1] and abs(lrs[1] - 1e-3) < 1e-12 and lrs[3] < 1e-5):
        print(f"FAIL schedule {lrs}")
        ok = False

    print(json.dumps({"self_test": "src.utils", "pass": ok, "git": git_sha()[:12],
                      "lr_probe": lrs}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
