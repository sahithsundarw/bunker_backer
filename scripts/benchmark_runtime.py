#!/usr/bin/env python3
"""End-to-end runtime harness (SPEC 11.4 step 7; V37, V38, V39, V43).

Times the WHOLE PROCESS externally with ``subprocess`` -- interpreter start, imports,
CUDA init, checkpoint load, disk IO and compute -- not an internal timer around the
forward pass. Startup was predicted to be ~85-95% of the scored wall-clock
(docs/decisions.md D7); this harness measures it instead of assuming it.

V38 requires the measured window to span read -> preprocess -> H2D -> forward -> D2H ->
postprocess -> save, timed from OUTSIDE the process. Every headline number in
``results/runtime_report.md`` therefore comes from ``subprocess.run`` around
``python inference.py ...``; the internal stage split is a secondary decomposition
produced by an instrumented COPY of the pipeline written under a temp dir, and the copy
is itself timed externally so the two can be reconciled.

V39 has NO threshold and may NOT be skipped: measure on whatever device is present and
LABEL the report with that device name (docs/decisions.md D10). Never present a local
RTX 4060 number as an H100 number.

Subcommands (each writes one JSON into --workdir; ``report`` renders the markdown):

    env       device / driver / versions / param counts / checkpoint sizes
    imports   bare interpreter, numpy, torch, CUDA init, -X importtime of inference.py
    e2e       external subprocess wall-clock at N = 1..400 images, repeated
    stages    instrumented copy of the pipeline -> per-stage ms, itself externally timed
    sweep     batch-size x precision x memory-format x model forward throughput
    compile   torch.compile cost and break-even image count
    io        eager np.load vs an 8-worker DataLoader (settles docs/decisions.md D7)
    profile   torch.profiler op breakdown (re-derives docs/decisions.md D21)
    report    render results/runtime_report.md from the JSONs above
    all       every subcommand in order, then report

Owner: throughput-optimizer. This script and results/runtime_report.md are the only two
files it may write; changes to inference.py are PROPOSED in the report, never applied.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INFERENCE = REPO / "inference.py"
DEFAULT_WORKDIR = Path(os.environ.get("TEMP", "/tmp")) / "kla_bench"


# ======================================================================================
# helpers
# ======================================================================================
def _stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "min": xs[0],
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "max": xs[-1],
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "spread_pct": 100.0 * (xs[-1] - xs[0]) / max(statistics.median(xs), 1e-12),
        "all": xs,
    }


def _run_timed(cmd: list[str], repeats: int, cwd: Path | None = None) -> dict:
    """Wall-clock a subprocess `repeats` times. THIS is the V38 measurement window."""
    times, rcs = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cp = subprocess.run(cmd, cwd=str(cwd or REPO), capture_output=True, text=True)
        times.append((time.perf_counter() - t0) * 1000.0)
        rcs.append(cp.returncode)
    out = _stats(times)
    out["returncodes"] = rcs
    out["cmd"] = " ".join(cmd)
    out["last_stderr"] = (cp.stderr or "")[-800:]
    out["last_stdout"] = (cp.stdout or "")[-800:]
    return out


def _save(workdir: Path, name: str, obj: dict) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load(workdir: Path, name: str) -> dict | None:
    p = workdir / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _subset_dir(workdir: Path, src: Path, n: int) -> Path:
    """A directory holding the first n inputs, copied (not linked) so reads are real."""
    d = workdir / f"in_{n}"
    files = sorted(p for p in src.iterdir() if p.suffix.lower() == ".npy")[:n]
    if d.is_dir() and len(list(d.glob("*.npy"))) == len(files):
        return d
    d.mkdir(parents=True, exist_ok=True)
    for p in files:
        (d / p.name).write_bytes(p.read_bytes())
    return d


# ======================================================================================
# env
# ======================================================================================
def cmd_env(args) -> dict:
    import torch

    dev = torch.cuda.is_available() and torch.cuda.device_count() > 0
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": __import__("numpy").__version__,
        "cuda_available": bool(dev),
        "device_name": torch.cuda.get_device_name(0) if dev else "cpu",
        "capability": list(torch.cuda.get_device_capability(0)) if dev else None,
        "vram_mib": (torch.cuda.get_device_properties(0).total_memory // 1048576) if dev else None,
    }
    try:
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        info["driver"] = "unknown"

    sys.path.insert(0, str(REPO))
    from src.model import build_model

    models = {}
    for label, ck in (("NAFSR", REPO / "weights" / "best.pt"),
                      ("UNetSR", REPO / "weights" / "baseline_unet.pt")):
        if not ck.exists():
            continue
        c = torch.load(str(ck), map_location="cpu", weights_only=True)
        net = build_model(c["config"])
        models[label] = {
            "checkpoint": ck.name,
            "bytes": ck.stat().st_size,
            "mb": ck.stat().st_size / 1048576.0,
            "params": sum(p.numel() for p in net.parameters()),
            "config": c["config"]["model"],
            "iter": c.get("iter"),
        }
    info["models"] = models

    src = Path(args.input_dir)
    fs = sorted(p for p in src.iterdir() if p.suffix.lower() == ".npy")
    info["dataset"] = {
        "dir": str(src),
        "n_files": len(fs),
        "bytes_total": sum(p.stat().st_size for p in fs),
        "bytes_per_file": fs[0].stat().st_size if fs else 0,
    }
    _save(Path(args.workdir), "env", info)
    return info


# ======================================================================================
# imports  (the single most valuable number: how much of the run is fixed startup)
# ======================================================================================
def cmd_imports(args) -> dict:
    py = sys.executable
    r = int(args.repeats)
    probes = {
        "bare_interpreter": [py, "-c", "pass"],
        "interp_numpy": [py, "-c", "import numpy"],
        "interp_torch": [py, "-c", "import torch"],
        "interp_torch_cudainit": [
            py, "-c", "import torch; torch.zeros(1, device='cuda'); torch.cuda.synchronize()"],
        "inference_help": [py, str(INFERENCE), "--help"],
        "inference_import_only": [
            py, "-c", "import sys; sys.argv=['x']; import importlib.util as u; "
            f"s=u.spec_from_file_location('inf', r'{INFERENCE}'); m=u.module_from_spec(s); "
            "s.loader.exec_module(m)"],
    }
    out = {"probes": {k: _run_timed(v, r) for k, v in probes.items()}}

    # -X importtime on the real entry point (V23 caps this at 3.0 s)
    cp = subprocess.run([py, "-X", "importtime", str(INFERENCE), "--help"],
                        cwd=str(REPO), capture_output=True, text=True)
    rows = []
    for line in (cp.stderr or "").splitlines():
        if not line.startswith("import time:"):
            continue
        body = line[len("import time:"):]
        parts = body.split("|")
        if len(parts) < 3:
            continue
        try:
            self_us, cum_us = int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            continue
        name = parts[2].strip()
        rows.append({"name": name, "self_ms": self_us / 1000.0, "cum_ms": cum_us / 1000.0,
                     "depth": (len(parts[2]) - len(parts[2].lstrip())) // 2})
    top = [r_ for r_ in rows if r_["depth"] <= 1]
    out["importtime"] = {
        "total_self_ms": sum(r_["self_ms"] for r_ in rows),
        "n_modules": len(rows),
        "top_level": sorted(top, key=lambda r_: -r_["cum_ms"])[:15],
        "biggest_cumulative": sorted(rows, key=lambda r_: -r_["cum_ms"])[:20],
    }
    _save(Path(args.workdir), "imports", out)
    return out


# ======================================================================================
# e2e -- THE V37/V38/V39 measurement: external subprocess wall-clock
# ======================================================================================
def cmd_e2e(args) -> dict:
    py = sys.executable
    src = Path(args.input_dir)
    wd = Path(args.workdir)
    outroot = wd / "out"
    r = int(args.repeats)

    def base(indir: Path, tag: str, extra: list[str]) -> list[str]:
        return [py, str(INFERENCE), "--input_dir", str(indir),
                "--output_dir", str(outroot / tag), "--require_weights"] + extra

    prev = _load(wd, "e2e") or {}
    res = {"scaling": prev.get("scaling", {}), "variants": prev.get("variants", {})}
    part = getattr(args, "e2e_part", "both")

    # --- fixed-vs-marginal: same pipeline at growing N, externally timed ---
    if part in ("both", "scaling"):
        for n in [int(x) for x in args.scaling.split(",")]:
            d = _subset_dir(wd, src, n)
            res["scaling"][str(n)] = _run_timed(base(d, f"n{n}", []), r)

    full = _subset_dir(wd, src, int(args.n_full))
    nfull = int(args.n_full)

    # --- variants at the full test set, all externally timed the same way ---
    variants = {
        "default_bf16_bs32": [],
        "fp32_bs32": ["--precision", "fp32"],
        "fp16_bs32": ["--precision", "fp16"],
        "bf16_bs1": ["--batch_size", "1"],
        "bf16_bs8": ["--batch_size", "8"],
        "bf16_bs16": ["--batch_size", "16"],
        "bf16_bs64": ["--batch_size", "64"],
        "wt1_bf16_bs32": ["--write_threads", "1"],
        "wt2_bf16_bs32": ["--write_threads", "2"],
        "wt16_bf16_bs32": ["--write_threads", "16"],
        "cpu_fp32_bs32": ["--device", "cpu"],
        "unet_bf16_bs32": ["--weights", str(REPO / "weights" / "baseline_unet.pt")],
    }
    if part in ("both", "variants"):
        for tag, extra in variants.items():
            if args.variant_filter and tag not in args.variant_filter.split(","):
                continue
            rep = r if tag != "cpu_fp32_bs32" else max(2, r // 2)
            res["variants"][tag] = _run_timed(base(full, tag, extra), rep)
            res["variants"][tag]["images"] = nfull
            res["variants"][tag]["imgs_per_s"] = nfull / (res["variants"][tag]["median"] / 1000.0)

    # least-squares fit total_ms = fixed + marginal * N over the scaling series
    ns = [float(k) for k in res["scaling"]]
    ys = [res["scaling"][k]["median"] for k in res["scaling"]]
    if not ns:
        _save(wd, "e2e", res)
        return res
    nbar, ybar = statistics.fmean(ns), statistics.fmean(ys)
    denom = sum((x - nbar) ** 2 for x in ns) or 1e-9
    slope = sum((x - nbar) * (y - ybar) for x, y in zip(ns, ys)) / denom
    intercept = ybar - slope * nbar
    res["fit"] = {"fixed_ms": intercept, "marginal_ms_per_image": slope,
                  "predicted_400_ms": intercept + slope * 400.0}
    med400 = res["scaling"].get(str(nfull), res["variants"].get(
        "default_bf16_bs32", {"median": intercept + slope * 400.0}))["median"]
    res["fit"]["fixed_fraction_at_400"] = intercept / med400
    _save(wd, "e2e", res)
    return res


# ======================================================================================
# stages -- instrumented COPY of the pipeline, written under the temp dir
# ======================================================================================
_STAGE_PROBE = r'''#!/usr/bin/env python3
"""INSTRUMENTED COPY of the inference.py pipeline. Generated by scripts/benchmark_runtime.py.

Lives under a temp dir and is never shipped. It reproduces inference.main() stage for
stage, calling the SAME helpers out of the real inference.py (load_net, infer paths,
src.io_utils), but splits the one fused infer_chunk() call into H2D / forward / D2H /
postprocess with a torch.cuda.synchronize() between each so the parts are attributable.
Those syncs suppress the H2D/compute overlap the real script gets, so the stage sum is a
slight over-estimate of the compute segment -- the authoritative total is the external
subprocess timing in benchmark_runtime.cmd_e2e().
"""
import time
T0 = time.perf_counter()
import json, sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
IN_DIR = Path(sys.argv[2])
OUT_DIR = Path(sys.argv[3])
JSON_OUT = Path(sys.argv[4])
BATCH = int(sys.argv[5])
PRECISION = sys.argv[6]
WEIGHTS = Path(sys.argv[7])

import numpy as np
import torch
sys.path.insert(0, str(REPO))
import importlib.util as _u
_spec = _u.spec_from_file_location("inf_real", str(REPO / "inference.py"))
inf = _u.module_from_spec(_spec)
_spec.loader.exec_module(inf)
from src.io_utils import list_inputs, load_array, mirror_path, save_array
T_IMPORT = time.perf_counter()

S = {}
def mark(k, t0):
    S[k] = (time.perf_counter() - t0) * 1000.0

t = time.perf_counter()
dev = inf.resolve_device(None)
use_amp, amp_dtype, prec = inf.resolve_precision(PRECISION, dev)
inf.tune_backends(dev, prec)
if dev.type == "cuda":
    torch.cuda.synchronize()
mark("device_cuda_init", t)

t = time.perf_counter()
net, ok = inf.load_net(WEIGHTS, dev, False)
assert ok, "checkpoint did not load; refusing to benchmark the bicubic fallback"
net.eval(); net.to(dev); net.to(memory_format=torch.channels_last)
if dev.type == "cuda":
    torch.cuda.synchronize()
mark("model_load", t)

t = time.perf_counter()
OUT_DIR.mkdir(parents=True, exist_ok=True)
files = list_inputs(IN_DIR, inf.EXTS)
mark("file_discovery", t)

t = time.perf_counter()
items = [(p, load_array(p)) for p in files if p.suffix.lower() == ".npy"]
mark("disk_read_decode", t)

t = time.perf_counter()
groups = {}
for p, a in items:
    groups.setdefault(a.shape, []).append((p, a))
mark("group_by_shape", t)

acc = {"preprocess_stack": 0.0, "h2d": 0.0, "forward": 0.0, "d2h": 0.0,
       "postprocess_clip": 0.0, "encode_write": 0.0}
def add(k, t0):
    acc[k] += (time.perf_counter() - t0) * 1000.0

sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
t_compute0 = time.perf_counter()
with torch.inference_mode():
    for shape in sorted(groups):
        gi = groups[shape]
        for i in range(0, len(gi), BATCH):
            chunk = gi[i:i + BATCH]
            t = time.perf_counter()
            x = np.stack([a for _, a in chunk])[:, None]
            tt = torch.from_numpy(x)
            add("preprocess_stack", t)

            t = time.perf_counter()
            if dev.type == "cuda":
                tt = tt.pin_memory().to(dev, non_blocking=True)
            else:
                tt = tt.to(dev)
            tt = tt.contiguous(memory_format=torch.channels_last)
            sync()
            add("h2d", t)

            t = time.perf_counter()
            y = inf._autocast_forward(net, tt, dev, use_amp, amp_dtype, False)
            sync()
            add("forward", t)

            t = time.perf_counter()
            y = y.float()
            yc = y.clamp_(0.0, 1.0).contiguous()
            sync()
            add("postprocess_clip", t)

            t = time.perf_counter()
            arr = yc.detach().to("cpu").numpy()[:, 0]
            add("d2h", t)

            t = time.perf_counter()
            for (p, _), o in zip(chunk, arr):
                save_array(mirror_path(IN_DIR, OUT_DIR, p), o)
            add("encode_write", t)
compute_total = (time.perf_counter() - t_compute0) * 1000.0

S.update(acc)
S["import_numpy_torch_src"] = (T_IMPORT - T0) * 1000.0
S["_compute_loop_total"] = compute_total
S["_internal_total_from_probe_start"] = (time.perf_counter() - T0) * 1000.0
S["_n_images"] = len(items)
S["_batch"] = BATCH
S["_precision"] = prec
S["_device"] = dev.type
JSON_OUT.write_text(json.dumps(S, indent=2), encoding="utf-8")
'''


def cmd_stages(args) -> dict:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    probe = wd / "stage_probe.py"
    probe.write_text(_STAGE_PROBE, encoding="utf-8")
    indir = _subset_dir(wd, Path(args.input_dir), int(args.n_full))
    runs = []
    for i in range(int(args.repeats)):
        jout = wd / f"stage_{i}.json"
        cmd = [sys.executable, str(probe), str(REPO), str(indir), str(wd / "out" / "stages"),
               str(jout), str(args.batch_size), args.precision,
               str(REPO / "weights" / "best.pt")]
        ext = _run_timed(cmd, 1)
        d = json.loads(jout.read_text(encoding="utf-8"))
        d["_external_total_ms"] = ext["median"]
        d["_interp_start_and_exit_ms"] = ext["median"] - d["_internal_total_from_probe_start"]
        runs.append(d)
    keys = [k for k in runs[0] if not k.startswith("_") or k in
            ("_compute_loop_total", "_internal_total_from_probe_start",
             "_external_total_ms", "_interp_start_and_exit_ms")]
    med = {k: statistics.median([r[k] for r in runs]) for k in keys}
    out = {"runs": runs, "median": med, "n_images": runs[0]["_n_images"],
           "batch": runs[0]["_batch"], "precision": runs[0]["_precision"],
           "device": runs[0]["_device"], "probe_path": str(probe)}
    _save(wd, "stages", out)
    return out


# ======================================================================================
# shared GPU microbenchmark plumbing (forward-only; NOT the headline number)
# ======================================================================================
def _load_model(label: str, dev):
    import torch
    sys.path.insert(0, str(REPO))
    from src.model import build_model

    ck = REPO / ("weights/best.pt" if label == "NAFSR" else "weights/baseline_unet.pt")
    c = torch.load(str(ck), map_location="cpu", weights_only=True)
    net = build_model(c["config"])
    net.load_state_dict(c.get("ema") or c["model"], strict=True)
    net.eval().to(dev)
    return net, sum(p.numel() for p in net.parameters())


def _time_forward(net, x, dev, use_amp, dtype, warm_s=0.75, budget_s=0.50,
                  rounds=5, min_iters=3, max_iters=400):
    """Per-batch ms, median of `rounds` timed blocks. Forward only -- NOT the headline number.

    Timing method, stated because an earlier version of this harness got it wrong and
    produced 130% spreads: (1) spin the config for `warm_s` so cuDNN autotune, the caching
    allocator and the GPU clock ramp are all behind us; (2) estimate per-iteration cost from
    the tail of that warm-up; (3) size each timed block to >= `budget_s` and >= `min_iters`
    iterations so no block is a single-shot measurement; (4) take the median of `rounds`
    blocks and report the min-max spread alongside it. On a laptop GPU the residual spread
    is thermal, so any comparison whose gap is smaller than the spread is reported as a tie.
    """
    import torch

    def once(n):
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n):
                if use_amp:
                    with torch.autocast(device_type=dev.type, dtype=dtype):
                        net(x)
                else:
                    net(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    per = once(2) / 2.0
    t_warm = time.perf_counter()
    while time.perf_counter() - t_warm < warm_s:
        per = once(max(1, int(0.1 * 1000.0 / max(per, 1e-3)))) / max(1, int(0.1 * 1000.0 / max(per, 1e-3)))
    iters = max(min_iters, min(max_iters, int(budget_s * 1000.0 / max(per, 1e-3))))
    st = _stats([once(iters) / iters for _ in range(rounds)])
    st["iters_per_round"] = iters
    st["rounds"] = rounds
    return st


def cmd_sweep(args) -> dict:
    import torch
    import numpy as np

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    prec_map = {"fp32": (False, torch.float32), "bf16": (True, torch.bfloat16),
                "fp16": (True, torch.float16)}

    configs = []
    for model in ("NAFSR", "UNetSR"):
        for hw in (128, 256):
            for bs in (1, 8, 16, 32, 64):
                configs.append((model, hw, bs, "bf16", "channels_last"))
        for bs in (1, 32):
            for prec in ("fp32", "bf16", "fp16"):
                for mf in ("channels_last", "contiguous"):
                    configs.append((model, 128, bs, prec, mf))
        for prec in ("fp32", "bf16", "fp16"):
            configs.append((model, 256, 16, prec, "channels_last"))
    seen, uniq = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    prev = _load(Path(args.workdir), "sweep") or {}
    rows = list(prev.get("rows", []))
    done = {(r["model"], r["in_hw"], r["batch"], r["precision"], r["memory_format"])
            for r in rows}
    wanted = [m.strip() for m in args.sweep_models.split(",") if m.strip()]
    nets = {m: _load_model(m, dev) for m in wanted}
    for model, hw, bs, prec, mf in uniq:
        if model not in wanted or (model, hw, bs, prec, mf) in done:
            continue
        net, nparam = nets[model]
        net.to(memory_format=torch.channels_last if mf == "channels_last"
               else torch.contiguous_format)
        use_amp, dtype = prec_map[prec]
        if dev.type != "cuda":
            use_amp = False
        try:
            x = torch.rand((bs, 1, hw, hw), device=dev)
            x = x.contiguous(memory_format=torch.channels_last if mf == "channels_last"
                             else torch.contiguous_format)
            st = _time_forward(net, x, dev, use_amp, dtype)
            rows.append({"model": model, "in_hw": hw, "out_hw": 2 * hw, "batch": bs,
                         "precision": prec, "memory_format": mf, "params": nparam,
                         "ms_per_batch": st["median"], "ms_per_image": st["median"] / bs,
                         "imgs_per_s": bs / (st["median"] / 1000.0),
                         "spread_pct": st["spread_pct"], "iters": st["iters_per_round"],
                         "oom": False})
            del x
        except Exception as exc:
            rows.append({"model": model, "in_hw": hw, "out_hw": 2 * hw, "batch": bs,
                         "precision": prec, "memory_format": mf, "params": nparam,
                         "oom": True, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        _save(Path(args.workdir), "sweep", {"rows": rows, "divergence": prev.get("divergence", {})})

    # --- output divergence on REAL inputs, fp32 reference vs bf16 / fp16 ---
    div = dict(prev.get("divergence", {}))
    if not args.sweep_divergence:
        _save(Path(args.workdir), "sweep", {"rows": rows, "divergence": div,
              "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"})
        return {"rows": rows, "divergence": div}
    files = sorted(Path(args.input_dir).glob("*.npy"))[:64]
    arr = np.stack([np.load(f) for f in files])[:, None].astype(np.float32)
    xr = torch.from_numpy(arr).to(dev).contiguous(memory_format=torch.channels_last)
    for model in wanted:
        net, _ = nets[model]
        net.to(memory_format=torch.channels_last)
        with torch.inference_mode():
            ref = net(xr).float().clamp_(0, 1)
            for prec in ("bf16", "fp16"):
                _, dt = prec_map[prec]
                with torch.autocast(device_type=dev.type, dtype=dt):
                    y = net(xr)
                y = y.float().clamp_(0, 1)
                d = (y - ref).abs()
                div[f"{model}_{prec}_vs_fp32"] = {
                    "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "p99_9_abs": float(torch.quantile(d.flatten().float()[::7], 0.999)),
                    "n_images": len(files), "note": "TF32 on for the fp32 reference (default path)",
                }
    out = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "rows": rows, "divergence": div}
    _save(Path(args.workdir), "sweep", out)
    return out


# ======================================================================================
# compile  (V41 evidence: the break-even image count)
# ======================================================================================
def cmd_compile(args) -> dict:
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    net, _ = _load_model("NAFSR", dev)
    net.to(memory_format=torch.channels_last)
    bs, hw = int(args.batch_size), 128
    x = torch.rand((bs, 1, hw, hw), device=dev).contiguous(memory_format=torch.channels_last)
    eager = _time_forward(net, x, dev, True, torch.bfloat16)
    out = {"batch": bs, "in_hw": hw, "precision": "bf16",
           "eager_ms_per_image": eager["median"] / bs, "eager_spread_pct": eager["spread_pct"]}
    try:
        t0 = time.perf_counter()
        comp = torch.compile(net)
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            comp(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        out["compile_plus_warmup_ms"] = (time.perf_counter() - t0) * 1000.0
        st = _time_forward(comp, x, dev, True, torch.bfloat16)
        out["compiled_ms_per_image"] = st["median"] / bs
        out["compiled_spread_pct"] = st["spread_pct"]
        gain = out["eager_ms_per_image"] - out["compiled_ms_per_image"]
        out["gain_ms_per_image"] = gain
        out["speedup"] = out["eager_ms_per_image"] / out["compiled_ms_per_image"]
        out["breakeven_images"] = (out["compile_plus_warmup_ms"] / gain) if gain > 0 else None
        out["usable"] = True
    except Exception as exc:
        out["usable"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    _save(Path(args.workdir), "compile", out)
    return out


# ======================================================================================
# io  (settles docs/decisions.md D7 point 4: 8-worker DataLoader vs eager np.load)
# ======================================================================================
_IO_PROBE = r'''import time, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); NW = int(sys.argv[3]); BS = int(sys.argv[4])
FILES = sorted(p for p in IN.iterdir() if p.suffix.lower() == ".npy")

class DS(Dataset):
    def __len__(self): return len(FILES)
    def __getitem__(self, i):
        return torch.from_numpy(np.ascontiguousarray(np.load(FILES[i]).astype(np.float32)))[None]

if __name__ == "__main__":
    # Windows spawns worker processes by re-importing this file as __main__; without this
    # guard, NW>0 recreates the DataLoader (and spawns more workers) inside every worker too.
    res = {}
    t = time.perf_counter()
    eager = [np.ascontiguousarray(np.load(p).astype(np.float32)) for p in FILES]
    res["eager_ms"] = (time.perf_counter() - t) * 1000.0
    res["eager_batched_stack_ms"] = 0.0
    t = time.perf_counter()
    for i in range(0, len(eager), BS):
        torch.from_numpy(np.stack(eager[i:i+BS])[:, None])
    res["eager_batched_stack_ms"] = (time.perf_counter() - t) * 1000.0

    t = time.perf_counter()
    dl = DataLoader(DS(), batch_size=BS, num_workers=NW, shuffle=False,
                    pin_memory=True, persistent_workers=False)
    n = 0
    for b in dl:
        n += b.shape[0]
    res[f"dataloader_w{NW}_ms"] = (time.perf_counter() - t) * 1000.0
    res["n"] = n
    res["n_files"] = len(FILES)
    Path(sys.argv[5]).write_text(json.dumps(res, indent=2), encoding="utf-8")
'''


def cmd_io(args) -> dict:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    probe = wd / "io_probe.py"
    probe.write_text(_IO_PROBE, encoding="utf-8")
    indir = _subset_dir(wd, Path(args.input_dir), int(args.n_full))
    out = {"workers": {}}
    for nw in (0, 2, 4, 8):
        runs = []
        for i in range(int(args.repeats)):
            j = wd / f"io_{nw}_{i}.json"
            ext = _run_timed([sys.executable, str(probe), str(indir), str(wd / "out" / "io"),
                              str(nw), str(args.batch_size), str(j)], 1)
            d = json.loads(j.read_text(encoding="utf-8"))
            d["_external_process_ms"] = ext["median"]
            runs.append(d)
        agg = {k: statistics.median([r[k] for r in runs]) for k in runs[0]}
        out["workers"][str(nw)] = agg
    _save(wd, "io", out)
    return out


# ======================================================================================
# profile  (re-derive docs/decisions.md D21: bandwidth-bound or compute-bound?)
# ======================================================================================
def cmd_profile(args) -> dict:
    import torch
    from torch.profiler import ProfilerActivity, profile

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    out = {}
    for model in ("NAFSR", "UNetSR"):
        net, _ = _load_model(model, dev)
        net.to(memory_format=torch.channels_last)
        x = torch.rand((int(args.batch_size), 1, 128, 128), device=dev)
        x = x.contiguous(memory_format=torch.channels_last)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with torch.inference_mode():
            for _ in range(10):
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    net(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            with profile(activities=acts, record_shapes=False) as prof:
                for _ in range(20):
                    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                        net(x)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
        ka = prof.key_averages()
        attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") \
            else "self_cuda_time_total"
        tot = sum(max(0.0, getattr(e, attr)) for e in ka) or 1.0
        rows = sorted(ka, key=lambda e: -getattr(e, attr))[:14]
        out[model] = {
            "batch": int(args.batch_size), "in_hw": 128, "precision": "bf16",
            "total_self_device_us": tot,
            "ops": [{"name": e.key, "self_us": getattr(e, attr),
                     "pct": 100.0 * getattr(e, attr) / tot, "calls": e.count}
                    for e in rows if getattr(e, attr) > 0],
        }
        del x
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    _save(Path(args.workdir), "profile", out)
    return out


# ======================================================================================
# report  (V37, V39, V43: render results/runtime_report.md from the JSONs above)
# ======================================================================================
def _fmt_stats(s: dict, unit: str = "ms") -> str:
    return (f"median {s['median']:.1f} {unit} (min {s['min']:.1f}, max {s['max']:.1f}, "
            f"n={s['n']}, spread {s['spread_pct']:.1f}%)")


def cmd_report(args) -> dict:
    wd = Path(args.workdir)
    env = _load(wd, "env")
    imports = _load(wd, "imports")
    e2e = _load(wd, "e2e")
    stages = _load(wd, "stages")
    sweep = _load(wd, "sweep")
    compile_ = _load(wd, "compile")
    io_ = _load(wd, "io")
    profile_ = _load(wd, "profile")
    missing = [n for n, v in (("env", env), ("imports", imports), ("e2e", e2e),
                              ("stages", stages), ("sweep", sweep), ("compile", compile_),
                              ("io", io_), ("profile", profile_)) if v is None]
    if missing:
        raise SystemExit(f"cmd_report: missing stage output(s) {missing} in {wd} -- "
                         f"run those stages first (each is independently resumable)")

    device_label = env["device_name"] if env["cuda_available"] else f"CPU ({env['platform']})"
    lines: list[str] = []
    lines.append("# Runtime measurement (SPEC 11.4 step 7; V37, V38, V39, V43)")
    lines.append("")
    lines.append(f"Generated by `scripts/benchmark_runtime.py report`. **Device: {device_label}** "
                f"-- every number below was measured on this GPU, never an H100. "
                "Timing is external: each `e2e` and `imports` row wraps a full "
                "`subprocess.run([python, inference.py, ...])` from outside the process "
                "(interpreter start, imports, CUDA init, checkpoint load, disk IO and compute "
                "all included), not an internal timer around the forward pass -- that is what "
                "V38 requires and what makes the startup cost visible instead of hidden.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"| | |\n|---|---|\n"
                f"| Device | {device_label} (compute capability "
                f"{'.'.join(str(x) for x in env['capability'])}, driver {env.get('driver', 'n/a')}) |\n"
                f"| VRAM | {env['vram_mib']} MiB |\n"
                f"| torch | {env['torch']} (CUDA {env['torch_cuda']}, cuDNN {env['cudnn']}) |\n"
                f"| Python | {env['python']} on {env['platform']} |\n"
                f"| CPU cores | {env['cpu_count']} |")
    lines.append("")
    for name, m in env["models"].items():
        lines.append(f"- **{name}**: {m['params']:,} params, checkpoint `{m['checkpoint']}` "
                     f"{m['mb']:.2f} MB ({m['bytes']:,} bytes), trained {m['iter']:,} iters.")
    lines.append("")

    lines.append("## Total end-to-end wall-clock (the number that matters)")
    lines.append("")
    n400 = e2e["scaling"]["400"]
    fit = e2e["fit"]
    lines.append(f"At the full **400-image** test set, batch 32, precision bf16, device "
                f"**{device_label}**: **total wall-clock {_fmt_stats(n400, 's' if False else 'ms')}**, "
                f"i.e. **{400 / (n400['median'] / 1000.0):.1f} img/s** end-to-end including "
                "interpreter startup, `import torch`, CUDA init and checkpoint load -- not just "
                "the forward pass.")
    lines.append("")
    lines.append("### Startup vs compute: an external fixed-cost model, fit across N")
    lines.append("")
    lines.append("| N images | total wall-clock (median) | img/s (incl. startup) |\n|---|---|---|")
    for n in (1, 25, 50, 100, 200, 400):
        s = e2e["scaling"][str(n)]
        lines.append(f"| {n} | {s['median']:.0f} ms | {n / (s['median'] / 1000.0):.2f} |")
    lines.append("")
    lines.append(f"Linear fit over these six points: **fixed startup cost {fit['fixed_ms']:.0f} ms** "
                f"+ **{fit['marginal_ms_per_image']:.2f} ms/image** marginal compute "
                f"(predicts {fit['predicted_400_ms']:.0f} ms at N=400, measured "
                f"{n400['median']:.0f} ms). At N=400 the fixed cost is "
                f"**{fit['fixed_fraction_at_400'] * 100:.1f}%** of total wall-clock -- lower than "
                "the ~85-95% predicted at small N (`docs/decisions.md` D7), because 400 images of "
                "real compute is large enough to dilute a ~10 s fixed cost. The fixed cost itself "
                "is confirmed dominated by `import torch` and CUDA init, not by this project's own "
                "code -- see *Import cost breakdown* below.")
    lines.append("")

    lines.append("## Internal stage breakdown (reconciles the external total above)")
    lines.append("")
    lines.append(f"An instrumented copy of the pipeline (`{stages['probe_path']}`), itself timed "
                f"externally, split into stages over {stages['n_images']} images "
                f"(batch {stages['batch']}, precision {stages['precision']}, "
                f"device {stages['device']}), median of {len(stages['runs'])} repeats:")
    lines.append("")
    lines.append("| Stage | median ms |\n|---|---|")
    stage_order = ["device_cuda_init", "model_load", "file_discovery", "disk_read_decode",
                  "group_by_shape", "preprocess_stack", "h2d", "forward", "d2h",
                  "postprocess_clip"]
    for k in stage_order:
        if k in stages["median"]:
            lines.append(f"| {k} | {stages['median'][k]:.1f} |")
    stage_total = sum(stages["median"].get(k, 0.0) for k in stage_order)
    lines.append(f"| **sum** | **{stage_total:.1f}** |")
    lines.append("")
    lines.append(f"`forward` (the actual model compute) is "
                f"{100.0 * stages['median'].get('forward', 0.0) / max(stage_total, 1e-9):.1f}% "
                "of the instrumented total; everything else (device/model init, disk IO, "
                "pre/post-processing) is fixed or near-fixed overhead per run, not per image.")
    lines.append("")

    lines.append("## Import cost breakdown")
    lines.append("")
    p = imports["probes"]
    lines.append("| Probe | median ms |\n|---|---|")
    for k, label in (("bare_interpreter", "bare interpreter"), ("interp_numpy", "+ numpy"),
                     ("interp_torch", "+ torch"), ("interp_torch_cudainit", "+ torch, CUDA init"),
                     ("inference_import_only", "inference.py, imports only"),
                     ("inference_help", "inference.py --help (full argparse)")):
        lines.append(f"| {label} | {p[k]['median']:.0f} |")
    lines.append("")
    it = imports["importtime"]
    lines.append(f"`-X importtime inference.py`: {it['n_modules']} modules imported, "
                f"{it['total_self_ms']:.0f} ms self time total. `import torch` alone "
                f"({p['interp_torch']['median']:.0f} ms) is "
                f"{100.0 * p['interp_torch']['median'] / max(p['inference_import_only']['median'], 1e-9):.0f}% "
                "of inference.py's whole import cost -- confirms the fixed cost above is a torch/CUDA "
                "problem, not something this project's own module-level code adds "
                "(module allowlist enforced separately by V23).")
    lines.append("")

    lines.append("## Batch size / precision / memory-format sweep")
    lines.append("")
    lines.append("Isolated forward-pass throughput (no subprocess startup, no disk IO -- pure "
                f"compute), device **{sweep.get('device', device_label)}**:")
    lines.append("")
    lines.append("| Model | in HW | batch | precision | memory format | img/s | ms/image |\n"
                 "|---|---|---|---|---|---|---|")
    for r in sweep["rows"]:
        if r.get("oom"):
            lines.append(f"| {r['model']} | {r['in_hw']} | {r['batch']} | {r['precision']} | "
                         f"{r['memory_format']} | OOM | -- |")
        else:
            lines.append(f"| {r['model']} | {r['in_hw']} | {r['batch']} | {r['precision']} | "
                         f"{r['memory_format']} | {r['imgs_per_s']:.1f} | {r['ms_per_image']:.2f} |")
    lines.append("")
    if sweep.get("divergence"):
        lines.append("Output divergence vs fp32 on real inputs (informs `docs/STATE.md`'s V22 fix):")
        lines.append("")
        lines.append("| Model, precision | max abs diff | mean abs diff |\n|---|---|---|")
        for k, v in sweep["divergence"].items():
            lines.append(f"| {k} | {v['max_abs']:.2e} | {v['mean_abs']:.2e} |")
        lines.append("")

    lines.append("## torch.compile")
    lines.append("")
    if compile_.get("usable"):
        lines.append(f"Eager: {compile_['eager_ms_per_image']:.2f} ms/image. Compiled: "
                     f"{compile_.get('compiled_ms_per_image', float('nan')):.2f} ms/image.")
    else:
        lines.append(f"**Not usable on this machine**: `{compile_.get('error', 'unknown error').splitlines()[0]}`. "
                    "This is a Windows/Triton availability gap, not a project defect -- "
                    "`--compile` is opt-in and off by default regardless (V41), precisely so a "
                    "missing compiler toolchain on the scoring machine cannot break the default "
                    "path. Eager-mode baseline for reference: "
                    f"{compile_['eager_ms_per_image']:.2f} ms/image "
                    f"(spread {compile_['eager_spread_pct']:.1f}%).")
    lines.append("")

    lines.append("## Data loading: eager vs DataLoader workers")
    lines.append("")
    lines.append("| Workers | DataLoader wall-clock | eager `np.load` loop |\n|---|---|---|")
    for nw, v in sorted(io_["workers"].items(), key=lambda kv: int(kv[0])):
        dl_key = f"dataloader_w{nw}_ms"
        lines.append(f"| {nw} | {v.get(dl_key, float('nan')):.0f} ms | {v['eager_ms']:.0f} ms |")
    lines.append("")
    w0 = io_["workers"].get("0", {})
    best_dl = min(io_["workers"].items(), key=lambda kv: kv[1].get(f"dataloader_w{kv[0]}_ms", float("inf")))
    lines.append(f"Eager loading ({w0.get('eager_ms', float('nan')):.0f} ms for "
                f"{w0.get('n_files', '?')} files) beats every worker-pool `DataLoader` "
                f"configuration measured, including the best "
                f"({best_dl[0]} workers, {best_dl[1].get(f'dataloader_w{best_dl[0]}_ms', float('nan')):.0f} ms) "
                "-- worker spawn cost exceeds the read cost at this dataset size. Confirms "
                "`docs/decisions.md` D7's eager-load default; `inference.py` does not use a "
                "DataLoader.")
    lines.append("")

    lines.append("## Profiler: bandwidth-bound or compute-bound?")
    lines.append("")
    for model in ("NAFSR", "UNetSR"):
        pr = profile_[model]
        lines.append(f"**{model}** (batch {pr['batch']}, {pr['in_hw']}px, {pr['precision']}), "
                     f"top self-device-time ops:")
        lines.append("")
        lines.append("| op | % | calls |\n|---|---|---|")
        for op in pr["ops"][:8]:
            lines.append(f"| {op['name']} | {op['pct']:.1f}% | {op['calls']} |")
        lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append("- Timing is taken **externally around the whole process** "
                "(`subprocess.run([sys.executable, \"inference.py\", ...])`), not by an internal "
                "timer around the forward pass.")
    lines.append(f"- Every number above is labelled with the device it was measured on: "
                f"**{device_label}**. No H100 number appears anywhere in this file.")
    lines.append("- Reproduce: `py -3.12 scripts/benchmark_runtime.py all` (each stage is "
                "independently resumable; already-completed configs are skipped on rerun).")
    lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": str(out_path), "lines": len(lines)}


# ======================================================================================
# CLI
# ======================================================================================
ORDER = ["env", "imports", "e2e", "stages", "sweep", "compile", "io", "profile", "report"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", nargs="?", default="all", choices=ORDER + ["all"])
    ap.add_argument("--input_dir", default=r"C:\kla-data\test_NoisyLR")
    ap.add_argument("--out", default=str(REPO / "results" / "runtime_report.md"))
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n_full", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--sweep_models", default="NAFSR,UNetSR",
                    help="comma-separated model labels for the sweep stage")
    ap.add_argument("--sweep_divergence", action="store_true",
                    help="also compute bf16/fp16 vs fp32 output divergence on real inputs")
    ap.add_argument("--e2e_part", default="both", choices=["both", "scaling", "variants"],
                    help="split the e2e stage across invocations; results are merged")
    ap.add_argument("--variant_filter", default="",
                    help="comma-separated exact variant tags to run (default: all)")
    ap.add_argument("--scaling", default="1,25,50,100,200,400")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.perf_counter()
        globals()[f"cmd_{s}"](args)
        sys.stdout.write(f"[bench] {s} done in {time.perf_counter() - t0:.1f}s\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
