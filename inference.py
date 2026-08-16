#!/usr/bin/env python3
"""Standalone inference: restore degraded images (2x SR + joint denoise).

THE FILE KLA RUNS AS-IS. A crash here means the submission is unscored (CLAUDE.md PD4).

    python inference.py --input_dir <degraded> --output_dir <restored>

Module-level imports are limited to EXACTLY the allowlist in CLAUDE.md STYLE:
    argparse os sys time pathlib concurrent.futures numpy torch
NO image IO library. The dataset is .npy end to end. V23 (Tier 0) asserts this statically
and caps `python -X importtime` at 3.0 s. The current external full-run benchmark measures
1.38 s of process startup/import overhead, so every import still costs real score. `src.model`
and `src.io_utils` are imported lazily inside main() for the same reason -- and because `src`
is not on V23's module-level allowlist.

Design notes that are deliberate and measured, not accidental:

* Eager load, no DataLoader. The whole test set is 400 files / 25.05 MB; spawning workers
  costs more than reading it (docs/decisions.md D7 point 4). This contradicts SPEC 11.2 on
  purpose -- do not "fix" it back without a measurement.
* Inputs are grouped by shape before batching (SPEC 7.3 / 11.2 / 18 pitfall 10) so 128 and
  256 inputs coexist in one invocation even though the released test set is uniform.
* The input is NEVER clipped (SPEC F5); the output is ALWAYS clipped to [0,1] and nothing
  else -- no renormalisation, which measured -4.66 dB (docs/decisions.md D3).
* Free levers on by default: inference_mode, channels_last, TF32, cuDNN benchmark, bf16
  autocast on CUDA, pinned + non_blocking H2D, threaded writes. `--compile` and `--tta` are
  opt-in because at 400 images neither amortises its fixed cost (V41, V42).

Owner: inference-engineer.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

#: Weights live beside THIS FILE, never relative to the CWD, never an absolute literal (V05).
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "weights" / "best.pt"

#: Permissive glob per SPEC 11.1. Only the .npy branch is decodable -- the uint8/uint16
#: rescaling branches of the SPEC 11.3 skeleton are deleted rather than left unreachable,
#: because code that divides by 255 must not sit next to data that reaches 2.16
#: (docs/SPEC_ADDENDUM.md section 5). A non-.npy file is reported on stderr, not guessed at.
EXTS = {".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

#: Fixed by the task: output is exactly 2x the input in both spatial axes.
SCALE = 2

#: Value written for an input we could not decode at all. Mid-grey is the MMSE-optimal
#: constant under no information; it is chosen over 0.0 so one unreadable file costs a few dB
#: instead of a floor score, and over any dataset mean because we must not tune to the
#: released proxy data. See the V07/V20 reconciliation note below.
PLACEHOLDER_VALUE = 0.5


def _err(msg: str) -> None:
    """Errors and warnings always go to stderr, regardless of --verbose."""
    sys.stderr.write(f"inference.py: {msg}\n")
    sys.stderr.flush()


def _brief(exc: BaseException, limit: int = 300) -> str:
    """First line of an exception message, capped.

    A ``load_state_dict(strict=True)`` mismatch lists every key in the architecture, which
    is hundreds of lines. That is noise in KLA's log and can bury the line that matters, so
    diagnostics are truncated -- the type and the first line identify the fault.
    """
    text = str(exc).strip().splitlines()
    head = text[0] if text else ""
    more = " [...]" if (len(text) > 1 or len(head) > limit) else ""
    return head[:limit] + more


def _say(verbose: bool, msg: str) -> None:
    """Diagnostics, gated behind --verbose (CLAUDE.md STYLE: no print debugging)."""
    if verbose:
        sys.stdout.write(f"{msg}\n")


def build_argparser() -> argparse.ArgumentParser:
    """Exactly two required args (V02). Everything else optional with working defaults."""
    ap = argparse.ArgumentParser(description="Restore degraded images (2x SR + denoise).")
    ap.add_argument("--input_dir", required=True, help="directory of degraded .npy inputs")
    ap.add_argument("--output_dir", required=True, help="directory for restored .npy outputs")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="optional checkpoint override; defaults to weights/best.pt "
                         "beside this script")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="per-shape-group batch size; halved automatically on CUDA OOM")
    ap.add_argument("--device", default=None, help="cuda | cuda:N | cpu (default: auto)")
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto = bf16 on CUDA, fp32 on CPU (CPU bf16 is slower and less "
                         "accurate with no upside)")
    ap.add_argument("--compile", action="store_true",
                    help="opt-in torch.compile; OFF by default because 30-120 s of "
                         "compilation never amortises over 400 images (V41)")
    ap.add_argument("--tta", action="store_true",
                    help="opt-in 8x dihedral self-ensemble; OFF by default (V42)")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="accepted for SPEC 11.1 compatibility; reads are eager because the "
                         "test set is 25 MB and worker spawn costs more (decisions.md D7)")
    ap.add_argument("--write_threads", type=int, default=0,
                    help="threaded .npy writers (0 = auto)")
    ap.add_argument("--allow_bicubic_fallback", action="store_true",
                    help="DEMO ONLY: allow bicubic outputs when the checkpoint or model "
                         "path cannot be used")
    ap.add_argument("--require_weights", action="store_true",
                    help="explicitly require model weights (the default submission behavior; "
                         "overrides --allow_bicubic_fallback)")
    ap.add_argument("--verbose", action="store_true", help="one-line summary on stdout")
    return ap


# ======================================================================================
# Device / precision / backend setup
# ======================================================================================
def cuda_usable() -> bool:
    """True only if a CUDA device is both available AND actually addressable.

    `torch.cuda.is_available()` alone is not enough: with CUDA_VISIBLE_DEVICES="" on Windows
    it still returns True while `device_count()` is 0, so every later device query raises
    "Invalid device id". Measured on this dev box (RTX 4060, torch 2.11.0+cu128) -- it is
    exactly the V19 failure mode, and the same trap would fire on a machine whose driver is
    present but whose GPU is masked off.
    """
    try:
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except Exception:  # noqa: BLE001 - a broken CUDA install must degrade to CPU, not crash
        return False


def resolve_device(requested: str | None) -> torch.device:
    """CUDA when usable, CPU otherwise. Never raises (V19: CPU fallback must work)."""
    if requested:
        want = torch.device(requested)
        if want.type == "cuda" and not cuda_usable():
            _err(f"device '{requested}' requested but no CUDA device is addressable; using CPU")
            return torch.device("cpu")
        return want
    return torch.device("cuda" if cuda_usable() else "cpu")


def tune_backends(dev: torch.device, precision: str = "bf16") -> None:
    """Free throughput levers. Harmless when CUDA is absent (docs/decisions.md D7 point 2).

    TF32 is on for every path except an explicit ``--precision fp32``, where it is turned off
    so that fp32 means fp32. TF32 carries a 10-bit mantissa, so leaving it on would make the
    fp32 arm of the V22 comparison an approximation too -- measured at 6.9e-05 mean on real
    inputs, i.e. comparing one approximation against another rather than against a reference.
    The default path is unaffected, so the free lever is kept where it actually pays.
    """
    torch.backends.cudnn.benchmark = True          # fixed shapes per group -> autotune pays
    tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if dev.type == "cuda":
        try:
            torch.cuda.init()
        except Exception as exc:  # noqa: BLE001
            _err(f"torch.cuda.init() failed ({type(exc).__name__}: {exc}); continuing")


def resolve_precision(requested: str, dev: torch.device) -> tuple[bool, torch.dtype, str]:
    """Return (use_autocast, autocast dtype, label).

    Device-aware default: bf16 on CUDA, fp32 on CPU. CPU bf16 autocast is a trap -- several
    ops fall back to fp32 anyway, the rest lose precision for no speed, and it would make
    V19 (CPU fallback) and V22 (bf16 vs fp32 agreement) fight each other. An explicit
    --precision bf16/fp16 on CPU is therefore downgraded to fp32 with a warning rather than
    silently honoured.
    """
    if dev.type != "cuda":
        if requested in ("bf16", "fp16"):
            _err(f"--precision {requested} is not used on {dev.type}; running fp32")
        return False, torch.float32, "fp32"
    if requested == "auto":
        requested = "bf16"
    if requested == "fp32":
        return False, torch.float32, "fp32"
    dtype = torch.bfloat16 if requested == "bf16" else torch.float16
    if requested == "bf16":
        try:
            supported = bool(torch.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001 - an unqueryable device must not abort the run
            supported = True
        if not supported:
            _err("bf16 unsupported on this GPU; running fp16")
            return True, torch.float16, "fp16"
    return True, dtype, requested


# ======================================================================================
# Model
# ======================================================================================
class BicubicUpsampler(torch.nn.Module):
    """Parameter-free 2x upsample for an explicitly requested demo fallback.

    Bicubic with antialias OFF is the measured inverse of the degradation kernel
    (docs/decisions.md D1) and is the documented floor baseline (D3). It is never selected
    during normal submission inference: callers must pass --allow_bicubic_fallback. It runs
    in fp32 regardless of --precision because there is nothing to gain from bf16 on a single
    interpolate, and fp32 keeps the demo path bit-reproducible.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(
            x.float(), scale_factor=float(SCALE), mode="bicubic",
            align_corners=False, antialias=False,
        )


def load_net(weights: Path, dev: torch.device, verbose: bool,
             allow_bicubic_fallback: bool = False) -> tuple[torch.nn.Module, bool]:
    """Build the architecture from the checkpoint's own config and load its weights.

    Building via build_model(ckpt["config"]) means weights can never be paired with the
    wrong architecture. EMA weights are preferred when present and non-empty.

    Returns (module, loaded_ok). A checkpoint failure is fatal to normal submission
    inference. Bicubic is returned only for the explicit demo fallback.
    """
    if not weights.is_file():
        if allow_bicubic_fallback:
            _err(f"checkpoint not found at {weights}; demo fallback enabled, using "
                 f"bicubic x{SCALE} upsample")
        else:
            _err(f"checkpoint not found at {weights}; refusing to generate non-model outputs")
        return BicubicUpsampler(), False
    try:
        ckpt = torch.load(str(weights), map_location=dev, weights_only=True)
        if not isinstance(ckpt, dict) or "config" not in ckpt:
            raise ValueError("checkpoint has no 'config' key; architecture is unknown")
        from src.model import build_model

        net = build_model(ckpt["config"])
        state = ckpt.get("ema") or ckpt.get("model")
        if not state:
            raise ValueError("checkpoint has neither a non-empty 'ema' nor 'model' state")
        net.load_state_dict(state, strict=True)
        _say(verbose, f"loaded {weights} ({'ema' if ckpt.get('ema') else 'model'} weights)")
        return net, True
    except Exception as exc:  # noqa: BLE001 - report one concise checkpoint diagnostic
        detail = f"could not load {weights} ({type(exc).__name__}: {_brief(exc)})"
        if allow_bicubic_fallback:
            _err(f"{detail}; demo fallback enabled, using bicubic x{SCALE} upsample")
        else:
            _err(f"{detail}; refusing to generate non-model outputs")
        return BicubicUpsampler(), False


# ======================================================================================
# Forward path
# ======================================================================================
def _autocast_forward(net: torch.nn.Module, x: torch.Tensor, dev: torch.device,
                      use_amp: bool, amp_dtype: torch.dtype, tta: bool) -> torch.Tensor:
    if use_amp:
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            return _tta_forward(net, x) if tta else net(x)
    return _tta_forward(net, x) if tta else net(x)


def try_compile(net: torch.nn.Module, probe_shape: tuple[int, int] | None, dev: torch.device,
                use_amp: bool, amp_dtype: torch.dtype, verbose: bool) -> torch.nn.Module:
    """Opt-in torch.compile with a guarded warm-up; reverts to eager on any failure.

    SPEC 11.2: "always keep the eager path working". A missing or too-old triton is the
    common failure (measured on this Windows dev box) and it raises at the FIRST FORWARD, not
    at the torch.compile() call, so the guard has to include a warm-up forward. The warm-up
    uses the modal input shape, so on the opt-in path it also moves that shape's compilation
    and cuDNN autotune ahead of the real batches.
    """
    if probe_shape is None:
        return net
    try:
        compiled = torch.compile(net)
        probe = torch.zeros((1, 1, probe_shape[0], probe_shape[1]), device=dev)
        probe = probe.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            _autocast_forward(compiled, probe, dev, use_amp, amp_dtype, False)
        _say(verbose, f"torch.compile warm-up ok at {probe_shape}")
        return compiled
    except Exception as exc:  # noqa: BLE001 - an unusable toolchain must not abort the run
        _err(f"torch.compile unusable ({type(exc).__name__}: {_brief(exc)}); running eager")
        return net


def _tta_forward(net: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """8x dihedral self-ensemble. Opt-in only (V42); off by default."""
    acc: torch.Tensor | None = None
    for k in range(4):
        for flip in (False, True):
            u = torch.rot90(x, k, (-2, -1))
            if flip:
                u = torch.flip(u, (-1,))
            v = net(u.contiguous(memory_format=torch.channels_last)).float()
            if flip:
                v = torch.flip(v, (-1,))
            v = torch.rot90(v, -k, (-2, -1))
            acc = v if acc is None else acc + v
    assert acc is not None
    return acc / 8.0


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


class _ModelOutputViolation(RuntimeError):
    """Raised when strict submission inference would otherwise substitute bicubic output."""


def infer_chunk(net: torch.nn.Module, arrays: list[np.ndarray], dev: torch.device,
                use_amp: bool, amp_dtype: torch.dtype, tta: bool,
                allow_bicubic_fallback: bool = False) -> np.ndarray:
    """Run one same-shape batch and return a (B, 2H, 2W) float32 array, clipped to [0,1].

    On CUDA OOM the batch is split and retried. A single image that still will not fit, or a
    model output with the wrong shape, is a hard error unless the caller explicitly enabled
    the demo-only bicubic fallback.
    """
    h, w = arrays[0].shape
    x = np.stack(arrays)[:, None]                      # B,1,H,W float32, unclipped
    t = torch.from_numpy(x)
    if dev.type == "cuda":
        t = t.pin_memory().to(dev, non_blocking=True)  # pinned + non_blocking H2D
    else:
        t = t.to(dev)
    t = t.contiguous(memory_format=torch.channels_last)
    try:
        y = _autocast_forward(net, t, dev, use_amp, amp_dtype, tta)
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        del t
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if len(arrays) > 1:
            mid = len(arrays) // 2
            _err(f"CUDA OOM at batch {len(arrays)}; retrying in two halves")
            a = infer_chunk(net, arrays[:mid], dev, use_amp, amp_dtype, tta,
                            allow_bicubic_fallback)
            b = infer_chunk(net, arrays[mid:], dev, use_amp, amp_dtype, tta,
                            allow_bicubic_fallback)
            return np.concatenate([a, b], axis=0)
        if not allow_bicubic_fallback:
            raise _ModelOutputViolation(
                f"CUDA OOM on a single {h}x{w} image; refusing to substitute bicubic output"
            ) from exc
        _err(f"CUDA OOM on a single {h}x{w} image; demo fallback enabled, using CPU "
             f"bicubic x{SCALE}")
        y = BicubicUpsampler()(torch.from_numpy(x))
    y = y.float()
    expect = (SCALE * h, SCALE * w)
    if tuple(y.shape[-2:]) != expect:
        msg = f"model returned {tuple(y.shape[-2:])} for a {h}x{w} input, expected {expect}"
        if not allow_bicubic_fallback:
            raise _ModelOutputViolation(f"{msg}; refusing to substitute bicubic output")
        _err(f"{msg}; demo fallback enabled, using bicubic x{SCALE} for this batch")
        y = BicubicUpsampler()(torch.from_numpy(x))
    y = y.clamp_(0.0, 1.0).contiguous()
    return y.detach().to("cpu").numpy()[:, 0]


# ======================================================================================
# Main
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    t_start = time.perf_counter()
    args = build_argparser().parse_args(argv)
    verbose = bool(args.verbose)

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not in_dir.is_dir():
        _err(f"--input_dir is not a directory: {in_dir}")
        return 1
    in_resolved, out_resolved = in_dir.resolve(), out_dir.resolve()
    if out_resolved == in_resolved or out_resolved.is_relative_to(in_resolved):
        # Unguarded, this either overwrites the degraded inputs with restored outputs IN
        # PLACE (a second run then restores the already-restored output again), or makes a
        # second invocation silently re-ingest the previous run's own output as new input
        # (adversarial review findings H2/H3). The only destructive write in this program
        # must not be reachable by accident.
        _err(f"--output_dir ({out_resolved}) is --input_dir or nested inside it; use a "
             f"separate directory for --output_dir")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy, script-relative import: keeps `src` off V23's module-level allowlist and keeps
    # `--help` cheap. sys.path[0] is already SCRIPT_DIR when run as a script; this makes the
    # import work when main() is called in-process from another CWD as well.
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from src.io_utils import list_inputs, load_array, mirror_path, save_array

    files = list_inputs(in_dir, EXTS)
    if not files:
        _err(f"no input files with extensions {sorted(EXTS)} under {in_dir}")
        return 1

    dev = resolve_device(args.device)
    use_amp, amp_dtype, prec = resolve_precision(args.precision, dev)
    tune_backends(dev, prec)

    allow_fallback = bool(args.allow_bicubic_fallback and not args.require_weights)
    net, weights_ok = load_net(Path(args.weights), dev, verbose, allow_fallback)
    if not weights_ok and not allow_fallback:
        if args.require_weights:
            _err("--require_weights was given and the checkpoint could not be loaded")
        else:
            _err("the default checkpoint is required for submission inference; "
                 "--allow_bicubic_fallback is available only for explicit demo runs")
        return 1
    net.eval()
    net.to(dev)
    net.to(memory_format=torch.channels_last)

    # ---- eager load, then group by shape (SPEC 7.3; 25 MB fits trivially in RAM) ----
    items: list[tuple[Path, np.ndarray]] = []
    unreadable: list[Path] = []
    for p in files:
        if p.suffix.lower() != ".npy":
            _err(f"skipping {p}: only .npy is decodable; this pipeline imports no image "
                 f"library by design (docs/io_contract.md)")
            continue
        try:
            items.append((p, load_array(p)))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the run (V20)
            unreadable.append(p)
            _err(f"unreadable input {p}: {type(exc).__name__}: {exc}")

    shape_counts: dict[tuple[int, int], int] = {}
    for _, arr in items:
        shape_counts[arr.shape] = shape_counts.get(arr.shape, 0) + 1
    modal = max(sorted(shape_counts), key=lambda s: shape_counts[s]) if shape_counts else None

    groups: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = {}
    for p, arr in items:
        groups.setdefault(arr.shape, []).append((p, arr))

    if args.compile:
        # Opt-in only (V41): 30-120 s of compilation never pays back over 400 images; the
        # crossover SPEC 11.2 quotes (~2000 images) is 5x the test set (docs/decisions.md D7).
        net = try_compile(net, modal, dev, use_amp, amp_dtype, verbose)

    n_threads = args.write_threads if args.write_threads > 0 else min(8, (os.cpu_count() or 4))
    pool = ThreadPoolExecutor(max_workers=n_threads)
    futures = []
    n_written = 0

    # V07 vs V20 reconciliation: V07 wants exactly one output per input file, V20 wants a
    # corrupt file to be logged and skipped rather than fatal. Both hold if an undecodable
    # input still receives a best-effort constant output at 2x the modal input shape: the
    # other N-1 images are produced normally (V20) and the output filename set stays
    # identical to the input set (V07, V08). A missing file could break a scorer that
    # iterates its own reference list; a flat image merely scores badly.
    for p in unreadable:
        if modal is None:
            _err(f"no readable input in this run, so no output shape can be inferred for {p}")
            continue
        ph = np.full((SCALE * modal[0], SCALE * modal[1]), PLACEHOLDER_VALUE, dtype=np.float32)
        _err(f"writing a flat {PLACEHOLDER_VALUE} placeholder for {p.name} "
             f"at {ph.shape} (2x the modal input shape {modal})")
        futures.append(pool.submit(save_array, mirror_path(in_dir, out_dir, p), ph))
        n_written += 1

    bs = max(1, int(args.batch_size))
    with torch.inference_mode():                       # not no_grad: also skips versioning
        for shape in sorted(groups):
            batch_items = groups[shape]
            for i in range(0, len(batch_items), bs):
                chunk = batch_items[i:i + bs]
                try:
                    y = infer_chunk(net, [a for _, a in chunk], dev, use_amp, amp_dtype,
                                    args.tta, allow_fallback)
                except _ModelOutputViolation as exc:
                    pool.shutdown(wait=True)
                    _err(str(exc))
                    return 1
                for (src, _), out in zip(chunk, y):
                    if not np.isfinite(out).all():
                        _err(f"non-finite prediction for {src.name}; neutralised on write")
                    futures.append(pool.submit(save_array, mirror_path(in_dir, out_dir, src),
                                               out))
                    n_written += 1

    pool.shutdown(wait=True)
    n_failed = 0
    for fut in futures:
        try:
            fut.result()
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            _err(f"write failed: {type(exc).__name__}: {exc}")

    n_ok = n_written - n_failed
    dt = time.perf_counter() - t_start
    if n_ok == 0:
        # One bad file is survivable (V20); zero usable outputs is a failed run and must not
        # masquerade as success -- exit 0 with an empty output dir is the worst possible
        # outcome on KLA's machine, because nothing would flag it.
        _err(f"produced no outputs for {len(files)} discovered input file(s) in {dt:.2f}s")
        return 1
    if n_failed > 0:
        # V07 requires exactly one output per input; a WRITE failure (disk full, quota,
        # permissions, a transient filesystem error) is not the "one corrupt input" case V20
        # exists to tolerate -- it means the output set KLA will read back is short, and an
        # exit 0 asserting success while that is true is the failure adversarial review
        # finding H4 identified.
        _err(f"{n_failed} of {n_written} outputs failed to write ({n_ok}/{len(files)} usable, "
             f"{dt:.2f}s); exiting non-zero because the output set is incomplete")
        return 1
    _say(verbose,
         f"restored {n_written - n_failed}/{len(files)} in {dt:.2f}s "
         f"({(n_written - n_failed) / max(dt, 1e-9):.1f} img/s) | device={dev.type} "
         f"precision={prec} batch={bs} shapes={sorted(groups)} "
         f"weights={'best' if weights_ok else 'bicubic-fallback'} "
         f"unreadable={len(unreadable)} write_errors={n_failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
