#!/usr/bin/env python3
"""verify_all.py — the executable form of docs/VERIFICATION_CONTRACT.md.

Implements every check V01-V52 from the contract, plus V00 (self-hash integrity).

Design rules (LOOP_PROMPT.md B3):
  * One function per check, named check_V##(ctx) -> CheckResult, auto-discovered by
    prefix. Adding a check must not require editing a dispatch table.
  * A check that cannot run because the code it tests does not exist returns FAIL with
    detail="not implemented yet" — never SKIP, never a silent pass.
  * SKIP is permitted ONLY for the whitelist in the contract.
  * Every check is wrapped: any exception becomes FAIL with the traceback in detail, so a
    project-side crash can never be swallowed or crash the verifier.

Usage:
    python scripts/verify_all.py
    python scripts/verify_all.py --strict
    python scripts/verify_all.py --only V07,V19
    python scripts/verify_all.py --tier 0
    python scripts/verify_all.py --fresh-clone
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_PATH = Path(__file__).resolve()

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
NOT_IMPL = "not implemented yet"

# --------------------------------------------------------------------------------------
# Tier map. V23 sits in Tier 0 by human authorisation (docs/decisions.md D6); its ID is
# deliberately not renumbered because IDs are stable identifiers.
# --------------------------------------------------------------------------------------
TIERS: dict[str, int] = {"V00": 0}
for _i in range(1, 15):
    TIERS[f"V{_i:02d}"] = 0
TIERS["V23"] = 0
for _i in list(range(15, 23)) + [24]:
    TIERS[f"V{_i:02d}"] = 1
for _i in range(25, 37):
    TIERS[f"V{_i:02d}"] = 2
for _i in range(37, 44):
    TIERS[f"V{_i:02d}"] = 3
for _i in range(44, 53):
    TIERS[f"V{_i:02d}"] = 4

# Whitelisted SKIPs, verbatim from the contract. V39's CUDA allowance was REMOVED by human
# authorisation (docs/decisions.md D10) — threshold-free wall-clock is measurable anywhere.
SKIP_WHITELIST: dict[str, str] = {
    "V40": "No CUDA device available in the dev environment. Must still pass static-scan portions.",
    "V06": "No git remote configured yet — permitted only before first push.",
}

# inference.py module-level import allowlist (CLAUDE.md §STYLE, tightened 2026-08-15).
IMPORT_ALLOWLIST = {
    "argparse", "os", "sys", "time", "pathlib", "concurrent", "concurrent.futures",
    "numpy", "torch", "__future__",
}
IMPORT_TIME_BUDGET_S = 3.0


@dataclass
class CheckResult:
    id: str
    status: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ctx:
    root: Path
    args: argparse.Namespace
    fixtures: Path

    # ---- small helpers ----
    def p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.p(*parts).exists()

    def read(self, *parts: str) -> str:
        return self.p(*parts).read_text(encoding="utf-8", errors="replace")

    def run(self, cmd: list[str], cwd: Path | None = None, timeout: int = 300,
            env: dict[str, str] | None = None) -> tuple[int, str, str]:
        e = os.environ.copy()
        if env:
            e.update(env)
        try:
            pr = subprocess.run(cmd, cwd=str(cwd or self.root), capture_output=True,
                                text=True, timeout=timeout, env=e)
            return pr.returncode, pr.stdout, pr.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return 125, "", f"{type(exc).__name__}: {exc}"

    def run_inference(self, in_dir: Path, out_dir: Path, extra: list[str] | None = None,
                      cwd: Path | None = None, timeout: int = 300,
                      env: dict[str, str] | None = None) -> tuple[int, str, str]:
        cmd = [sys.executable, str(self.p("inference.py")),
               "--input_dir", str(in_dir), "--output_dir", str(out_dir)]
        if extra:
            cmd += extra
        return self.run(cmd, cwd=cwd, timeout=timeout, env=env)

    def inference_ast(self) -> ast.Module | None:
        f = self.p("inference.py")
        if not f.exists():
            return None
        try:
            return ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            return None

    def tmpdir(self, name: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix=f"kla_{name}_"))
        return d


def not_impl(cid: str, what: str) -> CheckResult:
    """Uniform 'the code under test does not exist yet' failure. Never SKIP."""
    return CheckResult(cid, FAIL, NOT_IMPL, {"missing": what})


def _is_stub(path: Path) -> bool:
    """True if the file exists but every entry point raises NotImplementedError."""
    if not path.exists():
        return True
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return True
    return "NotImplementedError" in txt


# ======================================================================================
# FIXTURE CORPUS
# ======================================================================================
def build_fixtures(root: Path) -> Path:
    """Build a synthetic .npy corpus so robustness checks never touch the real dataset.

    Deliberately .npy float32 with values OUTSIDE [0,1], matching the real data
    (observed [-0.28, 2.16]). NOT PNG — a PNG corpus would prove nothing about this
    pipeline and would mask exactly the dtype/clipping bugs V10/V11/V12 exist to catch.
    """
    import numpy as np

    fx = root / "tests" / "fixtures"
    rng = np.random.default_rng(20260815)

    def w(p: Path, a) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, a.astype(np.float32))

    def img(h: int, w_: int, lo: float = -0.28, hi: float = 2.16):
        base = rng.random((h, w_), dtype=np.float64)
        return (base * (hi - lo) + lo)

    # mixed/: both resolutions, a nested subdir, a corrupt file, a non-array file
    w(fx / "mixed" / "a_128.npy", img(128, 128))
    w(fx / "mixed" / "b_128.npy", img(128, 128))
    w(fx / "mixed" / "c_256.npy", img(256, 256))
    w(fx / "mixed" / "nested" / "d_128.npy", img(128, 128))
    (fx / "mixed" / "corrupt.npy").write_bytes(b"\x93NUMPY\x01\x00truncated-garbage")
    (fx / "mixed" / "notes.txt").write_text("not an array\n", encoding="utf-8")

    # single/: exactly one file (batching edge case, V16)
    w(fx / "single" / "only_128.npy", img(128, 128))

    # size256/: 256 -> 512, the size-agnosticism fixture (SPEC T6). No such pair exists in
    # the real dataset (SPEC_ADDENDUM §1), so this is the ONLY guard against hard-coding.
    w(fx / "size256" / "e_256.npy", img(256, 256))

    # extreme/: degenerate and out-of-range content
    w(fx / "extreme" / "zeros_128.npy", np.zeros((128, 128)))
    w(fx / "extreme" / "ones_128.npy", np.ones((128, 128)))
    w(fx / "extreme" / "wide_128.npy", img(128, 128, -5.0, 5.0))
    w(fx / "extreme" / "nonsquare_96x160.npy", img(96, 160))

    # unicode/space filenames
    w(fx / "oddnames" / "with space.npy", img(128, 128))
    w(fx / "oddnames" / "unicode_éè_128.npy", img(128, 128))

    # large/: 200+ files for the batch check (V17), tiny so it stays fast
    large = fx / "large"
    for i in range(210):
        w(large / f"{i:05d}.npy", img(32, 32))

    return fx


# ======================================================================================
# TIER 0
# ======================================================================================
def check_V00(ctx: Ctx) -> CheckResult:
    """Verifier self-hash integrity pin."""
    digest = hashlib.sha256(SELF_PATH.read_bytes()).hexdigest()
    pin_file = ctx.p("docs", "VERIFIER_SHA256")
    if not pin_file.exists():
        return CheckResult("V00", FAIL, "docs/VERIFIER_SHA256 missing", {"computed": digest})
    pinned = None
    for line in pin_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[1].endswith("scripts/verify_all.py"):
            pinned = parts[0]
    if pinned is None:
        return CheckResult("V00", FAIL, "no pin recorded for scripts/verify_all.py",
                           {"computed": digest})
    if pinned == digest:
        return CheckResult("V00", PASS, "hash matches pin", {"sha256": digest})
    dec = ctx.read("docs", "decisions.md") if ctx.exists("docs", "decisions.md") else ""
    if digest in dec:
        return CheckResult("V00", PASS, "hash changed but documented in decisions.md",
                           {"computed": digest, "pinned": pinned})
    return CheckResult("V00", FAIL,
                       "verifier changed without a matching docs/decisions.md entry",
                       {"computed": digest, "pinned": pinned})


def check_V01(ctx: Ctx) -> CheckResult:
    f = ctx.p("inference.py")
    if not f.exists():
        return not_impl("V01", "inference.py")
    if list(ctx.root.glob("*.ipynb")):
        return CheckResult("V01", FAIL, "a notebook is present at repo root",
                           {"notebooks": [p.name for p in ctx.root.glob("*.ipynb")]})
    try:
        ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return CheckResult("V01", FAIL, f"ast.parse failed: {e}")
    return CheckResult("V01", PASS, "inference.py exists and parses", {"path": str(f)})


def check_V02(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V02", "inference.py")
    required: list[str] = []
    optional: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            name = None
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("--"):
                    name = a.value
            if name is None:
                continue
            req = any(k.arg == "required" and isinstance(k.value, ast.Constant)
                      and k.value.value is True for k in node.keywords)
            (required if req else optional).append(name)
    if sorted(required) != ["--input_dir", "--output_dir"]:
        return CheckResult("V02", FAIL,
                           f"required args are {sorted(required)}, expected ['--input_dir', '--output_dir']",
                           {"required": sorted(required), "optional": sorted(optional)})
    out = ctx.tmpdir("v02")
    rc, so, se = ctx.run_inference(ctx.fixtures / "single", out / "o")
    if rc != 0:
        d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc}: {se.strip()[-300:]}"
        return CheckResult("V02", FAIL, d, {"required": sorted(required)})
    return CheckResult("V02", PASS, "two required args, exit 0",
                       {"required": sorted(required), "optional": sorted(optional)})


def check_V03(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V03", "inference.py")
    results = {}
    for label, cwd in (("root", Path(ctx.root.anchor)), ("tmp", ctx.tmpdir("v03cwd"))):
        out = ctx.tmpdir(f"v03_{label}")
        rc, _, se = ctx.run_inference(ctx.fixtures / "single", out / "o", cwd=cwd)
        results[label] = rc
        if rc != 0:
            return CheckResult("V03", FAIL, f"failed from cwd={cwd}: {se.strip()[-300:]}", results)
    return CheckResult("V03", PASS, "runs from arbitrary CWD", results)


def check_V04(ctx: Ctx) -> CheckResult:
    if not ctx.args.fresh_clone:
        return CheckResult("V04", FAIL,
                           "requires --fresh-clone (not run; this is a FAIL, not a SKIP)")
    if not ctx.exists("requirements.txt"):
        return not_impl("V04", "requirements.txt")
    return _fresh_clone_run(ctx, "V04")


def _fresh_clone_run(ctx: Ctx, cid: str) -> CheckResult:
    work = ctx.tmpdir(f"{cid}_clone")
    rc, so, se = ctx.run(["git", "clone", "--quiet", str(ctx.root), str(work / "repo")])
    if rc != 0:
        return CheckResult(cid, FAIL, f"git clone failed: {se.strip()[-300:]}")
    repo = work / "repo"
    venv = repo / ".venv"
    rc, _, se = ctx.run([sys.executable, "-m", "venv", str(venv)], cwd=repo, timeout=600)
    if rc != 0:
        return CheckResult(cid, FAIL, f"venv creation failed: {se.strip()[-300:]}")
    py = venv / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    rc, _, se = ctx.run([str(py), "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                        cwd=repo, timeout=1800)
    if rc != 0:
        return CheckResult(cid, FAIL, f"pip install failed: {se.strip()[-500:]}")
    fx = repo / "tests" / "fixtures" / "single"
    if not fx.exists():
        build_fixtures(repo)
    rc, _, se = ctx.run([str(py), str(repo / "inference.py"),
                         "--input_dir", str(repo / "tests" / "fixtures" / "single"),
                         "--output_dir", str(work / "out")], cwd=repo, timeout=900)
    if rc != 0:
        d = NOT_IMPL if "NotImplementedError" in se else f"inference exit {rc}: {se.strip()[-300:]}"
        return CheckResult(cid, FAIL, d, {"clone": str(repo)})
    return CheckResult(cid, PASS, "fresh clone + fresh venv end-to-end", {"clone": str(repo)})


def check_V05(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V05", "inference.py")
    src = ctx.read("inference.py")
    if "Path(__file__)" not in src:
        return CheckResult("V05", FAIL, "weight path is not derived from Path(__file__)")
    if "os.getcwd()" in src:
        return CheckResult("V05", FAIL, "os.getcwd() used in path resolution")
    bad = re.findall(r'["\'](?:[A-Za-z]:\\\\|/(?:home|Users|mnt|opt))[^"\']*["\']', src)
    if bad:
        return CheckResult("V05", FAIL, "absolute path literal present", {"literals": bad[:5]})
    return CheckResult("V05", PASS, "weights resolved via Path(__file__)")


def check_V06(ctx: Ctx) -> CheckResult:
    ck = ctx.p("weights", "best.pt")
    if ck.exists():
        size = ck.stat().st_size
        head = ck.read_bytes()[:64]
        if head.startswith(b"version https://git-lfs"):
            return CheckResult("V06", FAIL, "weights/best.pt is an unresolved LFS pointer stub",
                               {"size": size})
        if size <= 1024:
            return CheckResult("V06", FAIL, f"weights/best.pt is {size} B (<= 1 KB)", {"size": size})
        return CheckResult("V06", PASS, "checkpoint present in clone", {"size": size})
    rd = ctx.p("weights", "README.md")
    if rd.exists():
        txt = rd.read_text(encoding="utf-8", errors="replace")
        urls = re.findall(r"https?://\S+", txt)
        has_sha = re.search(r"\b[0-9a-f]{64}\b", txt) is not None
        if urls and has_sha:
            return CheckResult("V06", FAIL,
                               "URL+sha256 present but not verified (needs a logged-out fetch)",
                               {"urls": urls[:3]})
    rc, so, _ = ctx.run(["git", "remote", "-v"])
    if rc == 0 and not so.strip():
        return CheckResult("V06", SKIP, SKIP_WHITELIST["V06"])
    return not_impl("V06", "weights/best.pt or a verified download URL + sha256")


def _npy_shape(p: Path) -> tuple[int, ...] | None:
    import numpy as np
    try:
        return tuple(np.load(p, mmap_mode="r", allow_pickle=False).shape)
    except Exception:  # noqa: BLE001
        return None


def _io_roundtrip(ctx: Ctx, src_dir: Path, cid: str) -> tuple[CheckResult | None, Path, Path]:
    """Run inference over src_dir; return (early_failure_or_None, in_dir, out_dir)."""
    out = ctx.tmpdir(f"{cid}_out")
    rc, _, se = ctx.run_inference(src_dir, out)
    if rc != 0:
        d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc}: {se.strip()[-300:]}"
        return CheckResult(cid, FAIL, d), src_dir, out
    return None, src_dir, out


def check_V07(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V07", "inference.py")
    src = ctx.fixtures / "mixed"
    early, ind, out = _io_roundtrip(ctx, src, "V07")
    if early:
        return early
    n_in = len([p for p in ind.rglob("*") if p.is_file() and p.suffix == ".npy"])
    n_out = len([p for p in out.rglob("*") if p.is_file()])
    if n_in != n_out:
        return CheckResult("V07", FAIL, f"{n_in} inputs -> {n_out} outputs",
                           {"n_in": n_in, "n_out": n_out})
    return CheckResult("V07", PASS, f"{n_in} in, {n_out} out", {"n": n_in})


def check_V08(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V08", "inference.py")
    src = ctx.fixtures / "mixed"
    early, ind, out = _io_roundtrip(ctx, src, "V08")
    if early:
        return early
    ins = {str(p.relative_to(ind)) for p in ind.rglob("*.npy") if p.is_file()}
    outs = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    if ins != outs:
        return CheckResult("V08", FAIL, "output filename set differs from input",
                           {"only_in": sorted(ins - outs)[:5], "only_out": sorted(outs - ins)[:5]})
    return CheckResult("V08", PASS, "filenames byte-identical incl. subdir paths", {"n": len(ins)})


def check_V09(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V09", "inference.py")
    bad: list[dict[str, Any]] = []
    unreadable: list[str] = []
    checked = 0
    for sub in ("mixed", "size256"):
        src = ctx.fixtures / sub
        early, ind, out = _io_roundtrip(ctx, src, "V09")
        if early:
            return early
        for p in ind.rglob("*.npy"):
            si = _npy_shape(p)
            q = out / p.relative_to(ind)
            so = _npy_shape(q) if q.exists() else None
            if si is None or len(si) < 2:
                # The INPUT is unreadable, so no (in, out) pair exists and the contract's
                # "for every pair" has nothing to say about it. V20 declares exactly this
                # case survivable, so failing it here would put V09 in direct conflict
                # with V20. Counted and reported, never silently dropped.
                unreadable.append(p.name)
                continue
            if so is None or len(so) < 2:
                bad.append({"file": p.name, "in": si, "out": so, "why": "no readable output"})
            elif (so[0], so[1]) != (2 * si[0], 2 * si[1]):
                bad.append({"file": p.name, "in": si, "out": so, "why": "not 2x"})
            else:
                checked += 1
    if bad:
        return CheckResult("V09", FAIL, f"{len(bad)} outputs are not exactly 2x",
                           {"violations": bad[:5], "unreadable_inputs": unreadable,
                            "pairs_checked": checked})
    if checked == 0:
        # Anti-vacuity: skipping unreadable inputs must never let V09 pass by checking
        # nothing at all.
        return CheckResult("V09", FAIL,
                           "no readable input/output pair was checked — V09 cannot pass vacuously",
                           {"unreadable_inputs": unreadable})
    return CheckResult("V09", PASS,
                       f"all {checked} pairs are exactly 2x "
                       f"({len(unreadable)} unreadable inputs excluded, see V20)",
                       {"pairs_checked": checked, "unreadable_inputs": unreadable})


def check_V10(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V10", "inference.py")
    import numpy as np
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "mixed", "V10")
    if early:
        return early
    bad = []
    for q in out.rglob("*"):
        if not q.is_file():
            continue
        if q.suffix != ".npy":
            bad.append({"file": q.name, "why": "not .npy"})
            continue
        try:
            a = np.load(q, mmap_mode="r", allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            bad.append({"file": q.name, "why": f"unreadable: {e}"})
            continue
        if a.dtype != np.float32:
            bad.append({"file": q.name, "why": f"dtype {a.dtype}, expected float32"})
    if bad:
        return CheckResult("V10", FAIL, "format/dtype mismatch vs docs/io_contract.md",
                           {"violations": bad[:5]})
    return CheckResult("V10", PASS, ".npy float32 throughout, per io_contract")


def check_V11(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V11", "inference.py")
    import numpy as np
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "extreme", "V11")
    if early:
        return early
    bad = []
    for q in out.rglob("*.npy"):
        a = np.load(q, allow_pickle=False)
        if not np.isfinite(a).all():
            bad.append({"file": q.name, "why": "NaN or Inf"})
        elif float(a.min()) < 0.0 or float(a.max()) > 1.0:
            bad.append({"file": q.name, "min": float(a.min()), "max": float(a.max())})
    if bad:
        return CheckResult("V11", FAIL, "output outside [0,1] or non-finite", {"violations": bad[:5]})
    return CheckResult("V11", PASS, "all outputs finite and within [0,1]")


def check_V12(ctx: Ctx) -> CheckResult:
    """Input must NOT be clipped — out-of-range values are intentional (SPEC F5)."""
    if _is_stub(ctx.p("src", "io_utils.py")) and _is_stub(ctx.p("inference.py")):
        return not_impl("V12", "src/io_utils.py load path")
    import numpy as np
    probe = ctx.tmpdir("v12") / "in"
    probe.mkdir(parents=True, exist_ok=True)
    arr = np.array([[-0.28, 2.16], [0.5, 1.5]], dtype=np.float32)
    arr = np.pad(arr, ((0, 126), (0, 126)), mode="wrap").astype(np.float32)
    np.save(probe / "probe_128.npy", arr)
    sys.path.insert(0, str(ctx.root))
    try:
        from src.io_utils import load_array  # type: ignore
    except Exception as e:  # noqa: BLE001
        return CheckResult("V12", FAIL, f"cannot import src.io_utils.load_array: {e}")
    finally:
        sys.path.pop(0)
    try:
        got = load_array(probe / "probe_128.npy")
    except NotImplementedError:
        return not_impl("V12", "src/io_utils.load_array")
    if float(got.min()) > -0.27 or float(got.max()) < 2.15:
        return CheckResult("V12", FAIL, "input was clipped on load",
                           {"min": float(got.min()), "max": float(got.max())})
    return CheckResult("V12", PASS, "out-of-range input values preserved",
                       {"min": float(got.min()), "max": float(got.max())})


def check_V13(ctx: Ctx) -> CheckResult:
    required = ["README.md", "inference.py", "train.py", "requirements.txt"]
    missing = [r for r in required if not ctx.exists(r)]
    rto = ctx.p("results", "restored_test_outputs")
    real = [p for p in rto.glob("*") if p.is_file() and p.name != ".gitkeep"] if rto.exists() else []
    if not real:
        missing.append("results/restored_test_outputs/ (non-empty)")
    if not ctx.exists("weights", "best.pt") and not ctx.exists("weights", "README.md"):
        missing.append("weights/")
    rc, so, _ = ctx.run(["git", "remote", "-v"])
    remote = so.strip()
    if missing:
        return CheckResult("V13", FAIL, f"missing: {', '.join(missing)}",
                           {"missing": missing, "remote": remote})
    if not remote:
        return CheckResult("V13", FAIL, "no git remote configured; repo cannot be public yet")
    return CheckResult("V13", PASS, "all mandatory items present", {"remote": remote})


def check_V14(ctx: Ctx) -> CheckResult:
    if not ctx.exists("requirements.txt"):
        return not_impl("V14", "requirements.txt")
    lines = [l.strip() for l in ctx.read("requirements.txt").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return CheckResult("V14", FAIL, "requirements.txt is empty")
    unpinned = [l for l in lines if "==" not in l]
    if unpinned:
        return CheckResult("V14", FAIL, f"{len(unpinned)} unpinned lines", {"unpinned": unpinned[:8]})
    listed = {re.split(r"[=<>\[]", l)[0].strip().lower().replace("_", "-") for l in lines}
    # Local first-party modules are not third-party distributions: any bare `import foo`
    # that resolves to a foo.py inside this repo (e.g. a sibling import between scripts/)
    # must not be demanded of requirements.txt.
    local_mods = {p.stem for p in ctx.root.rglob("*.py") if ".venv" not in p.parts}
    local_pkgs = {p.parent.name for p in ctx.root.rglob("__init__.py") if ".venv" not in p.parts}
    stdlib = set(sys.stdlib_module_names) | {"src", "__future__"} | local_mods | local_pkgs
    top: set[str] = set()
    for py in list(ctx.root.rglob("*.py")):
        if ".venv" in py.parts or "tests" in py.parts:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    top.add(a.name.split(".")[0])
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                top.add(n.module.split(".")[0])
    alias = {"cv2": "opencv-python", "skimage": "scikit-image", "yaml": "pyyaml",
             "PIL": "pillow", "sklearn": "scikit-learn"}
    missing = sorted({alias.get(m, m).lower().replace("_", "-") for m in top - stdlib} - listed)
    if missing:
        return CheckResult("V14", FAIL, f"imports not covered: {missing}", {"missing": missing})
    return CheckResult("V14", PASS, f"{len(lines)} pinned lines, all imports covered")


# ======================================================================================
# TIER 1
# ======================================================================================
def check_V15(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V15", "inference.py")
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "mixed", "V15")
    if early:
        return early
    shapes = {_npy_shape(p) for p in out.rglob("*.npy")}
    if len(shapes) < 2:
        return CheckResult("V15", FAIL, "mixed-resolution folder did not yield mixed outputs",
                           {"shapes": [list(s) if s else None for s in shapes]})
    return CheckResult("V15", PASS, "128 and 256 coexist in one run",
                       {"shapes": [list(s) for s in shapes if s]})


def check_V16(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V16", "inference.py")
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "single", "V16")
    if early:
        return early
    n = len(list(out.rglob("*.npy")))
    if n != 1:
        return CheckResult("V16", FAIL, f"single-image dir produced {n} outputs")
    return CheckResult("V16", PASS, "single-image folder handled")


def check_V17(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V17", "inference.py")
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "large", "V17")
    if early:
        return early
    n = len(list(out.rglob("*.npy")))
    if n < 210:
        return CheckResult("V17", FAIL, f"expected 210 outputs, got {n}")
    return CheckResult("V17", PASS, f"{n} images at default batch size, no OOM")


def check_V18(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V18", "inference.py")
    early, ind, out = _io_roundtrip(ctx, ctx.fixtures / "mixed", "V18")
    if early:
        return early
    if not (out / "nested" / "d_128.npy").exists():
        return CheckResult("V18", FAIL, "nested subdirectory structure not mirrored")
    return CheckResult("V18", PASS, "subdirs found and mirrored")


def check_V19(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V19", "inference.py")
    out = ctx.tmpdir("v19")
    rc, _, se = ctx.run_inference(ctx.fixtures / "single", out / "o",
                                  env={"CUDA_VISIBLE_DEVICES": ""})
    if rc != 0:
        d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc}: {se.strip()[-300:]}"
        return CheckResult("V19", FAIL, d)
    return CheckResult("V19", PASS, "CPU fallback completes")


def check_V20(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V20", "inference.py")
    src = ctx.fixtures / "mixed"  # contains corrupt.npy and notes.txt
    early, ind, out = _io_roundtrip(ctx, src, "V20")
    if early:
        return early
    good = len([p for p in ind.rglob("*.npy") if p.name != "corrupt.npy"])
    got = len([p for p in out.rglob("*.npy") if p.name != "corrupt.npy"])
    if got < good:
        return CheckResult("V20", FAIL, f"corrupt file aborted the run: {got}/{good} produced")
    return CheckResult("V20", PASS, f"corrupt file skipped, {got} others produced")


def check_V21(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V21", "inference.py")
    import numpy as np
    out = ctx.tmpdir("v21") / "o"
    for _ in range(2):
        rc, _, se = ctx.run_inference(ctx.fixtures / "single", out)
        if rc != 0:
            d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc}: {se.strip()[-300:]}"
            return CheckResult("V21", FAIL, d)
    first = sorted(out.rglob("*.npy"))
    if not first:
        return CheckResult("V21", FAIL, "no outputs produced")
    a = np.load(first[0], allow_pickle=False)
    out2 = ctx.tmpdir("v21b") / "o"
    ctx.run_inference(ctx.fixtures / "single", out2)
    b = np.load(sorted(out2.rglob("*.npy"))[0], allow_pickle=False)
    if not np.array_equal(a, b):
        return CheckResult("V21", FAIL, "repeat runs are not byte-identical",
                           {"max_abs_diff": float(np.abs(a - b).max())})
    return CheckResult("V21", PASS, "idempotent across runs")


def check_V22(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V22", "inference.py")
    import numpy as np
    o1, o2 = ctx.tmpdir("v22a") / "o", ctx.tmpdir("v22b") / "o"
    rc1, _, se1 = ctx.run_inference(ctx.fixtures / "single", o1, ["--precision", "bf16"])
    rc2, _, se2 = ctx.run_inference(ctx.fixtures / "single", o2, ["--precision", "fp32"])
    if rc1 != 0 or rc2 != 0:
        se = se1 or se2
        d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc1}/{rc2}: {se.strip()[-300:]}"
        return CheckResult("V22", FAIL, d)
    a = np.load(sorted(o1.rglob("*.npy"))[0], allow_pickle=False).astype(np.float64)
    b = np.load(sorted(o2.rglob("*.npy"))[0], allow_pickle=False).astype(np.float64)
    mad, mx = float(np.abs(a - b).mean()), float(np.abs(a - b).max())
    if mad >= 1e-3 or mx >= 1e-2:
        return CheckResult("V22", FAIL, f"bf16 vs fp32 diverge: mean {mad:.2e}, max {mx:.2e}",
                           {"mean_abs": mad, "max_abs": mx})
    return CheckResult("V22", PASS, f"mean {mad:.2e}, max {mx:.2e}", {"mean_abs": mad, "max_abs": mx})


def check_V23(ctx: Ctx) -> CheckResult:
    """TIER 0 (promoted; docs/decisions.md D6). Module-level imports + import time."""
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V23", "inference.py")
    offenders: list[str] = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in IMPORT_ALLOWLIST and a.name not in IMPORT_ALLOWLIST:
                    offenders.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and mod.split(".")[0] not in IMPORT_ALLOWLIST \
                    and mod not in IMPORT_ALLOWLIST:
                offenders.append(mod)
    if offenders:
        return CheckResult("V23", FAIL, f"non-allowlisted module-level imports: {offenders}",
                           {"offenders": offenders, "allowlist": sorted(IMPORT_ALLOWLIST)})
    rc, so, se = ctx.run([sys.executable, "-X", "importtime", str(ctx.p("inference.py")), "--help"],
                         timeout=180)
    total_us = 0
    for line in (se or "").splitlines():
        m = re.match(r"import time:\s+\d+\s+\|\s+(\d+)\s+\|\s+(\S+)", line)
        if m and "." not in m.group(2):
            total_us = max(total_us, int(m.group(1)))
    secs = total_us / 1e6
    if secs >= IMPORT_TIME_BUDGET_S:
        return CheckResult("V23", FAIL, f"import time {secs:.2f}s >= {IMPORT_TIME_BUDGET_S}s",
                           {"import_seconds": secs})
    return CheckResult("V23", PASS, f"allowlist clean, import {secs:.2f}s",
                       {"import_seconds": secs, "offenders": []})


def check_V24(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V24", "inference.py")
    import numpy as np
    outs = []
    for i in range(2):
        o = ctx.tmpdir(f"v24_{i}") / "o"
        rc, _, se = ctx.run_inference(ctx.fixtures / "single", o)
        if rc != 0:
            d = NOT_IMPL if "NotImplementedError" in se else f"exit {rc}: {se.strip()[-300:]}"
            return CheckResult("V24", FAIL, d)
        outs.append(np.load(sorted(o.rglob("*.npy"))[0], allow_pickle=False))
    if not np.array_equal(outs[0], outs[1]):
        return CheckResult("V24", FAIL, "separate processes give different outputs")
    src = ctx.read("inference.py")
    if ".train()" in src:
        return CheckResult("V24", FAIL, "model.train() called in inference path")
    return CheckResult("V24", PASS, "deterministic across processes")


# ======================================================================================
# TIER 2
# ======================================================================================
def _needs(ctx: Ctx, cid: str, *rel: str) -> CheckResult | None:
    for r in rel:
        if _is_stub(ctx.p(*r.split("/"))):
            return not_impl(cid, r)
    return None


def _import_project(ctx: Ctx, modname: str) -> Any:
    """Import a project module fresh, from the repo under test.

    Re-imported each call so a check never scores a stale module left in sys.modules by
    an earlier check. Any exception propagates to the runner, which turns it into FAIL
    with the traceback — a project-side crash must never be swallowed.
    """
    import importlib

    root = str(ctx.root)
    if root not in sys.path:
        sys.path.insert(0, root)
    for k in [k for k in sys.modules if k == modname or k.startswith(modname + ".")]:
        del sys.modules[k]
    return importlib.import_module(modname)


def _baseline_metrics(ctx: Ctx, name: str) -> dict[str, dict[str, Any]] | None:
    """Read results/baselines/<name>/metrics.json into {metric: {mean,std,n}}."""
    p = ctx.p("results", "baselines", name, "metrics.json")
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    m = raw.get("metrics", raw)
    out: dict[str, dict[str, Any]] = {}
    for k in ("psnr", "ssim", "lpips"):
        v = m.get(k)
        if isinstance(v, dict) and "mean" in v:
            out[k] = {"mean": float(v["mean"]),
                      "std": (float(v["std"]) if v.get("std") is not None else None),
                      "n": v.get("n") or raw.get("n")}
        elif isinstance(v, (int, float)):
            out[k] = {"mean": float(v), "std": None, "n": raw.get("n")}
    return out or None


def _data_root(ctx: Ctx) -> Path | None:
    """The dataset root, if this machine has it. Never required to exist.

    The dataset lives OUTSIDE the repo by design, so any check that needs it must
    degrade honestly (to FAIL with a clear reason) rather than pretend to verify.
    """
    cand = [os.environ.get("KLA_DATA_ROOT"), r"C:\kla-data", "/kla-data"]
    for c in cand:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _sem(d: dict[str, Any]) -> float | None:
    """Standard error of the mean, or None if std/n are unavailable."""
    std, n = d.get("std"), d.get("n")
    if std is None or not n or n <= 1:
        return None
    return float(std) / (float(n) ** 0.5)


#: V25's bar, owned by the VERIFIER and hash-pinned, not read from train.py. Same governance
#: reasoning as V33_THRESHOLDS (ml-skeptic F2): the subject under test must not own its own
#: pass mark. The contract says PSNR > 40 dB; this is that number, not a copy of train.py's.
V25_TARGET_DB = 40.0


def _extract_json_object(text: str, must_contain: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of mixed log output. Returns None if absent."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and must_contain in obj:
            return obj
    return None


def check_V25(ctx: Ctx) -> CheckResult:
    """Pipeline sanity: overfit 2 fixed pairs to PSNR > 40 dB.

    A failure here means alignment, normalisation or the loss is broken, and nothing
    downstream is trustworthy -- so this check runs the real training path rather than
    trusting a recorded number.
    """
    r = _needs(ctx, "V25", "train.py", "src/model.py")
    if r:
        return r
    if _data_root(ctx) is None:
        return CheckResult("V25", FAIL,
                           "dataset root not found (set KLA_DATA_ROOT); the overfit gate "
                           "trains on real pairs and cannot run without them")
    cfg = ctx.p("configs", "final.yaml")
    if not cfg.exists():
        return not_impl("V25", "configs/final.yaml")
    rc, so, se = ctx.run([sys.executable, str(ctx.p("train.py")),
                          "--config", str(cfg), "--overfit", "2"], timeout=3600)
    rep = _extract_json_object(so + "\n" + se, "best_psnr_db")
    if rep is None:
        return CheckResult("V25", FAIL,
                           f"train.py --overfit 2 produced no parsable report (rc={rc})",
                           {"stdout_tail": so[-800:], "stderr_tail": se[-800:]})
    best = rep.get("best_psnr_db")
    ev = {k: rep.get(k) for k in
          ("pass", "best_psnr_db", "best_at_iter", "iters", "n_pairs", "pair_names",
           "split_of_pairs", "structural_kind", "seed", "device", "wall_clock_s")}
    if rep.get("n_pairs") != 2:
        return CheckResult("V25", FAIL,
                           f"overfit ran on {rep.get('n_pairs')} pairs, the contract says 2",
                           ev)
    if rep.get("split_of_pairs") != "train":
        return CheckResult("V25", FAIL,
                           f"overfit pairs came from '{rep.get('split_of_pairs')}', not train",
                           ev)
    if best is None or not float(best) > V25_TARGET_DB:
        return CheckResult("V25", FAIL,
                           f"overfit reached {best} dB, below the {V25_TARGET_DB} dB gate — "
                           "alignment, normalisation or the loss is broken", ev)
    if rc != 0:
        return CheckResult("V25", FAIL,
                           f"overfit cleared {best} dB but train.py exited {rc}", ev)
    return CheckResult("V25", PASS,
                       f"overfit 2 pairs reached {float(best):.4f} dB at iter "
                       f"{rep.get('best_at_iter')} (gate {V25_TARGET_DB} dB)", ev)


def check_V26(ctx: Ctx) -> CheckResult:
    """Paired-crop alignment: the GT crop is exactly the 2x region of the LR crop.

    Asserted by running the marker test in src/dataset.py, which crops a synthetic
    image carrying a known marker and checks the marker's position in both members.
    """
    r = _needs(ctx, "V26", "src/dataset.py")
    if r:
        return r
    mod = _import_project(ctx, "src.dataset")
    fn = getattr(mod, "selftest_paired_crop", None)
    if fn is None:
        return CheckResult("V26", FAIL,
                           "src/dataset.py exposes no selftest_paired_crop(); V26 requires a "
                           "marker-based paired-crop test callable from the verifier")
    res = fn()
    if not isinstance(res, dict) or "pass" not in res:
        return CheckResult("V26", FAIL,
                           f"selftest_paired_crop() returned {type(res).__name__}, "
                           "expected a dict containing 'pass'")
    if not res.get("pass"):
        return CheckResult("V26", FAIL, "paired-crop marker test FAILED", res)
    # Anti-vacuity: a test that checked nothing must not pass.
    n = res.get("n_crops") or res.get("n") or res.get("n_checked") or 0
    if not n:
        return CheckResult("V26", FAIL,
                           "paired-crop test reported pass but checked zero crops", res)
    return CheckResult("V26", PASS, f"paired-crop alignment verified over {n} crops", res)


def check_V27(ctx: Ctx) -> CheckResult:
    """Final model must beat bicubic on PSNR and SSIM, and be lower on LPIPS.

    "By a margin, not noise" is enforced statistically rather than by an invented
    constant: the PSNR gain must exceed two standard errors of the mean, computed from
    the std and n the contract already requires to be reported.
    """
    r = _needs(ctx, "V27", "scripts/evaluate.py", "src/metrics.py")
    if r:
        return r
    ref = _baseline_metrics(ctx, "bicubic")
    if ref is None:
        return not_impl("V27", "results/baselines/bicubic/metrics.json")
    cand = _baseline_metrics(ctx, "final")
    if cand is None:
        return not_impl("V27", "results/baselines/final/metrics.json (no trained model yet)")
    need = ("psnr", "ssim", "lpips")
    missing = [k for k in need if k not in cand or k not in ref]
    if missing:
        return CheckResult("V27", FAIL, f"metrics missing: {missing}",
                           {"final": cand, "bicubic": ref})
    nostd = [k for k in need if cand[k].get("std") is None or ref[k].get("std") is None]
    if nostd:
        return CheckResult("V27", FAIL,
                           f"std not reported for {nostd}; the contract requires mean +/- std "
                           "over the split", {"final": cand, "bicubic": ref})
    ev = {k: {"final_mean": cand[k]["mean"], "final_std": cand[k]["std"],
              "bicubic_mean": ref[k]["mean"], "bicubic_std": ref[k]["std"],
              "n": cand[k].get("n")} for k in need}
    losses = []
    if not cand["psnr"]["mean"] > ref["psnr"]["mean"]:
        losses.append("psnr")
    if not cand["ssim"]["mean"] > ref["ssim"]["mean"]:
        losses.append("ssim")
    if not cand["lpips"]["mean"] < ref["lpips"]["mean"]:
        losses.append("lpips")
    if losses:
        return CheckResult("V27", FAIL, f"does not beat bicubic on {losses}", ev)
    margin = cand["psnr"]["mean"] - ref["psnr"]["mean"]
    sems = [s for s in (_sem(cand["psnr"]), _sem(ref["psnr"])) if s is not None]
    ev["psnr_margin_db"] = margin
    if sems:
        need_margin = 2.0 * max(sems)
        ev["required_margin_db"] = need_margin
        if margin <= need_margin:
            return CheckResult("V27", FAIL,
                               f"PSNR margin {margin:.4f} dB does not exceed two standard "
                               f"errors ({need_margin:.4f} dB) — that is noise, not a margin",
                               ev)
    return CheckResult("V27", PASS,
                       f"beats bicubic by {margin:.4f} dB PSNR, SSIM "
                       f"{cand['ssim']['mean']:.5f} vs {ref['ssim']['mean']:.5f}, LPIPS "
                       f"{cand['lpips']['mean']:.5f} vs {ref['lpips']['mean']:.5f}", ev)


def check_V28(ctx: Ctx) -> CheckResult:
    """Final model must beat the U-Net baseline on at least 2 of the 3 metrics.

    The contract provides one escape hatch and it is deliberately narrow: a loss
    converts FAIL->PASS only if the negative result is documented in
    docs/decisions.md AND the better model is the one shipped. Both are checked.
    """
    r = _needs(ctx, "V28", "scripts/make_baselines.py")
    if r:
        return r
    ref = None
    for nm in ("unet_baseline", "unet", "baseline_unet"):
        ref = _baseline_metrics(ctx, nm)
        if ref is not None:
            break
    if ref is None:
        return not_impl("V28", "results/baselines/unet_baseline/metrics.json")
    cand = _baseline_metrics(ctx, "final")
    if cand is None:
        return not_impl("V28", "results/baselines/final/metrics.json (no trained model yet)")
    need = ("psnr", "ssim", "lpips")
    missing = [k for k in need if k not in cand or k not in ref]
    if missing:
        return CheckResult("V28", FAIL, f"metrics missing: {missing}")
    wins = []
    if cand["psnr"]["mean"] > ref["psnr"]["mean"]:
        wins.append("psnr")
    if cand["ssim"]["mean"] > ref["ssim"]["mean"]:
        wins.append("ssim")
    if cand["lpips"]["mean"] < ref["lpips"]["mean"]:
        wins.append("lpips")
    ev = {"wins": wins,
          "final": {k: cand[k]["mean"] for k in need},
          "unet": {k: ref[k]["mean"] for k in need}}
    if len(wins) >= 2:
        return CheckResult("V28", PASS,
                           f"beats the U-Net baseline on {len(wins)}/3 metrics: {wins}", ev)
    dec = ctx.read("docs", "decisions.md") if ctx.exists("docs", "decisions.md") else ""
    documented = "V28" in dec and "negative result" in dec.lower()
    ev["negative_result_documented"] = documented
    if documented:
        return CheckResult("V28", PASS,
                           f"loses to the U-Net baseline ({len(wins)}/3) but the negative "
                           "result is documented in docs/decisions.md as the contract permits",
                           ev)
    return CheckResult("V28", FAIL,
                       f"beats the U-Net baseline on only {len(wins)}/3 metrics and no honest "
                       "negative result is documented in docs/decisions.md", ev)


def check_V29(ctx: Ctx) -> CheckResult:
    sv = ctx.p("configs", "split_val.txt")
    if not sv.exists():
        return not_impl("V29", "configs/split_val.txt")
    names = [l.strip() for l in sv.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not names:
        return CheckResult("V29", FAIL, "configs/split_val.txt has no file entries")
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        return CheckResult("V29", FAIL,
                           f"configs/split_val.txt has {len(dupes)} duplicate entries",
                           {"duplicates": dupes[:10]})
    if _is_stub(ctx.p("src", "dataset.py")):
        return not_impl("V29", "src/dataset.py train-list intersection")

    # The split must be READ from the committed file, never regenerated at runtime.
    txt = ctx.read("src", "dataset.py")
    if "split_val" not in txt:
        return CheckResult("V29", FAIL,
                           "src/dataset.py never references configs/split_val.txt, so the "
                           "committed split cannot be the one actually used")

    mod = _import_project(ctx, "src.dataset")
    fn = getattr(mod, "check_split_integrity", None)
    if fn is None:
        return CheckResult("V29", FAIL,
                           "src/dataset.py exposes no check_split_integrity(); V29 requires "
                           "the train and val lists to be intersected, which needs the module "
                           "to report what it actually trains on")
    root = _data_root(ctx)
    res = fn(data_root=str(root)) if root else fn()
    if not isinstance(res, dict) or "pass" not in res:
        return CheckResult("V29", FAIL,
                           f"check_split_integrity() returned {type(res).__name__}, "
                           "expected a dict containing 'pass'")
    inter = res.get("intersection_size")
    n_val, n_train = res.get("n_val"), res.get("n_train")
    if root is None:
        # The contract asserts V29 "by intersecting the two lists". Without the dataset
        # there is no train list to intersect, so this is unverifiable -- report that
        # honestly rather than passing on the file-only invariants.
        return CheckResult("V29", FAIL,
                           "dataset root not found (set KLA_DATA_ROOT); the train/val "
                           "intersection the contract requires cannot be computed", res)
    if inter is None or n_val is None or n_train is None:
        return CheckResult("V29", FAIL,
                           "check_split_integrity() must report n_train, n_val and "
                           "intersection_size", res)
    if inter != 0:
        return CheckResult("V29", FAIL,
                           f"LEAKAGE: {inter} filenames appear in BOTH the train and val lists",
                           res)
    if not n_val or not n_train:
        return CheckResult("V29", FAIL,
                           f"degenerate split (n_train={n_train}, n_val={n_val}); an empty "
                           "side makes a zero intersection meaningless", res)
    if len(names) != n_val:
        return CheckResult("V29", FAIL,
                           f"committed split_val.txt has {len(names)} entries but the module "
                           f"reports n_val={n_val} — the file is not the split in use", res)
    if not res.get("pass"):
        return CheckResult("V29", FAIL, "check_split_integrity() reported failure", res)
    return CheckResult("V29", PASS,
                       f"no leakage: {n_val} val / {n_train} train, intersection 0", res)


def check_V30(ctx: Ctx) -> CheckResult:
    p = ctx.p("scripts", "evaluate.py")
    if _is_stub(p):
        return not_impl("V30", "scripts/evaluate.py")
    src = p.read_text(encoding="utf-8")
    if "np.load" not in src and "load_array" not in src:
        return CheckResult("V30", FAIL, "evaluate.py does not reload artifacts from disk")
    return CheckResult("V30", PASS, "evaluate.py reloads from disk")


def check_V31(ctx: Ctx) -> CheckResult:
    p = ctx.p("src", "metrics.py")
    if _is_stub(p):
        return not_impl("V31", "src/metrics.py")
    src = p.read_text(encoding="utf-8")
    need = ["data_range=1.0", "gaussian_weights=True", "sigma=1.5",
            "use_sample_covariance=False", "net='alex'"]
    missing = [n for n in need if n not in src.replace('"', "'")]
    if missing:
        return CheckResult("V31", FAIL, f"pinned settings missing: {missing}", {"missing": missing})
    return CheckResult("V31", PASS, "metric settings pinned per SPEC §10")


def check_V32(ctx: Ctx) -> CheckResult:
    if _is_stub(ctx.p("src", "model.py")):
        return not_impl("V32", "src/model.py")
    for py in ctx.root.rglob("*.py"):
        if ".venv" in py.parts:
            continue
        t = py.read_text(encoding="utf-8", errors="replace")
        if "cv2.imread(" in t and "IMREAD_UNCHANGED" not in t:
            return CheckResult("V32", FAIL, f"plain cv2.imread in {py.name}")
    import torch  # local: the verifier's own import cost is not scored

    mod = _import_project(ctx, "src.model")
    m = mod.build_model({"name": "NAFSR", "width": 16, "num_blocks": 2, "scale": 2,
                         "in_ch": 1, "out_ch": 1})
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 1, 32, 32))
    if tuple(y.shape) != (1, 1, 64, 64):
        return CheckResult("V32", FAIL,
                           f"single-channel forward gave {tuple(y.shape)}, expected (1,1,64,64)")
    # A model that silently accepts 3 channels is not single-channel end to end; it would
    # let an accidental BGR/RGB path through without ever raising.
    try:
        with torch.no_grad():
            m(torch.zeros(1, 3, 32, 32))
    except Exception:  # noqa: BLE001
        pass
    else:
        return CheckResult("V32", FAIL,
                           "model accepted a 3-channel input; not single-channel end to end")
    return CheckResult("V32", PASS,
                       "model is 1-channel in and 1-channel out, rejects 3-channel input, "
                       "and no plain cv2.imread exists anywhere",
                       {"in_shape": [1, 1, 32, 32], "out_shape": list(y.shape)})


#: V33's acceptance thresholds, owned by the VERIFIER and therefore covered by
#: docs/VERIFIER_SHA256. They were previously read only from src/degrade.py's own
#: FIDELITY_TOLERANCE, which is not a pinned file — so a future iteration could have widened
#: the bar to turn V33 green without touching anything pinned and without tripping Prime
#: Directive 1. That governance hole was reported by ml-skeptic (iteration 1, finding F2).
#: The module's own `pass` flag is still required; these are applied ON TOP of it, so the
#: check is the AND of both and can only ever be stricter than before.
#:
#: Measured values at the time of pinning (2800 non-val pairs, 45,875,200 px), stable to
#: <0.003 across noise seeds (seed 0 vs seed 7):
#:   mean_abs_rel_err 0.38849 | _x_ge_0p1 0.27639 | resid_std_ratio 1.05540
#:   binned_r2 0.98038 | gain_over_spec_2par 1.8943 | gain_worst_bin 5.9169
V33_THRESHOLDS: dict[str, tuple[str, float]] = {
    "mean_abs_rel_err": ("<=", 0.50),
    "mean_abs_rel_err_x_ge_0p1": ("<=", 0.35),
    "binned_r2": (">=", 0.97),
    "gain_over_spec_2par": (">=", 1.5),
    # Tightened from the module's 3.0 to 4.5. ml-skeptic measured 97% headroom at 3.0 and
    # called it near-vacuous; the observed 5.917/5.931 leaves 24% headroom at 4.5, which
    # still tolerates seed noise an order of magnitude larger than what was measured.
    "gain_over_spec_2par_worst_bin": (">=", 4.5),
}
V33_STD_RATIO_RANGE = (0.90, 1.15)


def check_V33(ctx: Ctx) -> CheckResult:
    """Degradation simulator fidelity, plus the mandatory evidence figure.

    Acceptance is the AND of the module's own `pass` flag and the verifier-owned
    thresholds above, so widening src/degrade.py's FIDELITY_TOLERANCE alone can no longer
    turn this check green.
    """
    r = _needs(ctx, "V33", "src/degrade.py")
    if r:
        return r
    fig = None
    for cand in (("results", "degrade_fidelity", "degrade_fidelity.png"),
                 ("results", "eda", "degrade_fidelity.png")):
        if ctx.exists(*cand):
            fig = "/".join(cand)
            break
    if fig is None:
        return CheckResult("V33", FAIL,
                           "no degradation-fidelity evidence figure on disk; the contract "
                           "requires one to be saved")
    mod = _import_project(ctx, "src.degrade")
    fn = getattr(mod, "fidelity_report", None)
    if fn is None:
        return CheckResult("V33", FAIL,
                           "src/degrade.py exposes no fidelity_report(); V33 requires the "
                           "variance-vs-intensity comparison to be callable from the verifier")
    root = _data_root(ctx)
    if root is not None:
        # Recompute live against the real pairs -- far stronger than trusting an artifact.
        # fidelity_report() writes results/degrade_fidelity/degrade_fidelity.json
        # unconditionally, which would leave the tree dirty and break Definition-of-Done
        # criterion 5 ("git status is clean"). Reported by ml-skeptic (iteration 1, F3).
        # The verifier must not mutate the repository it is verifying, so snapshot the
        # committed artifact and restore it byte-for-byte afterwards.
        _art = ctx.p("results", "degrade_fidelity", "degrade_fidelity.json")
        _before = _art.read_bytes() if _art.exists() else None
        try:
            res = fn(str(root), make_figure=False)
        finally:
            if _before is not None and _art.exists() and _art.read_bytes() != _before:
                _art.write_bytes(_before)
        source = "recomputed live"
    else:
        # No dataset on this machine (e.g. a fresh clone). Fall back to the committed
        # report, and say so in the detail so nobody mistakes it for a live measurement.
        rp = ctx.p("results", "degrade_fidelity", "degrade_fidelity.json")
        if not rp.exists():
            return CheckResult("V33", FAIL,
                               "dataset root not found (set KLA_DATA_ROOT) and no committed "
                               "degrade_fidelity.json to fall back on")
        res = json.loads(rp.read_text(encoding="utf-8"))
        source = "from committed degrade_fidelity.json (dataset not present)"
    if not isinstance(res, dict) or "pass" not in res:
        return CheckResult("V33", FAIL,
                           f"fidelity_report() returned {type(res).__name__}, "
                           "expected a dict containing 'pass'")
    met = res.get("metrics", {}) if isinstance(res.get("metrics"), dict) else {}
    npx = (res.get("n_pixels") or res.get("n_px")
           or met.get("n_pixels") or met.get("n_px") or 0)
    if not npx:
        return CheckResult("V33", FAIL,
                           "fidelity_report() reported no pixel count; a tolerance claim over "
                           "an unstated corpus is not evidence", res)
    if not res.get("pass"):
        return CheckResult("V33", FAIL,
                           "synthetic degradation does not match the real variance-vs-intensity "
                           "curve within the documented tolerance", res)
    # Independent, verifier-owned acceptance. Applied on top of the module's own flag.
    breaches = []
    for key, (op, lim) in V33_THRESHOLDS.items():
        val = met.get(key, res.get(key))
        if val is None:
            breaches.append(f"{key} not reported")
            continue
        if op == "<=" and not float(val) <= lim:
            breaches.append(f"{key}={float(val):.5f} > {lim}")
        elif op == ">=" and not float(val) >= lim:
            breaches.append(f"{key}={float(val):.5f} < {lim}")
    ratio = met.get("resid_std_ratio", res.get("resid_std_ratio"))
    if ratio is None:
        breaches.append("resid_std_ratio not reported")
    elif not (V33_STD_RATIO_RANGE[0] <= float(ratio) <= V33_STD_RATIO_RANGE[1]):
        breaches.append(f"resid_std_ratio={float(ratio):.5f} outside {V33_STD_RATIO_RANGE}")
    if breaches:
        return CheckResult("V33", FAIL,
                           f"verifier-owned fidelity thresholds breached: {breaches}",
                           {"breaches": breaches, **{k: met.get(k) for k in V33_THRESHOLDS}})
    ev = {k: v for k, v in res.items() if not isinstance(v, (list, dict))}
    ev.update({k: v for k, v in met.items() if not isinstance(v, (list, dict))})
    ev["figure"] = fig
    ev["source"] = source
    return CheckResult("V33", PASS,
                       f"synthetic degradation matches the real curve within tolerance "
                       f"over {npx} px, {source}; evidence {fig}", ev)


def check_V34(ctx: Ctx) -> CheckResult:
    """Reproducibility: two identical seeded smoke runs must give identical losses."""
    r = _needs(ctx, "V34", "train.py")
    if r:
        return r
    if _data_root(ctx) is None:
        return CheckResult("V34", FAIL,
                           "dataset root not found (set KLA_DATA_ROOT); the smoke run trains "
                           "on real pairs and cannot run without them")
    cfg = ctx.p("configs", "final.yaml")
    if not cfg.exists():
        return not_impl("V34", "configs/final.yaml")
    runs = []
    for _ in range(2):
        rc, so, se = ctx.run([sys.executable, str(ctx.p("train.py")),
                              "--config", str(cfg), "--seed", "42", "--smoke"], timeout=1800)
        out = so + "\n" + se
        losses = digest = None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("SMOKE_LOSSES "):
                try:
                    losses = json.loads(s[len("SMOKE_LOSSES "):])
                except ValueError:
                    losses = None
            elif s.startswith("SMOKE_DIGEST "):
                digest = s[len("SMOKE_DIGEST "):].strip()
        runs.append({"rc": rc, "losses": losses, "digest": digest,
                     "tail": out[-500:]})
    a, b = runs
    if a["rc"] != 0 or b["rc"] != 0:
        return CheckResult("V34", FAIL,
                           f"smoke runs exited {a['rc']} and {b['rc']}",
                           {"tail_a": a["tail"], "tail_b": b["tail"]})
    if not a["losses"] or not b["losses"]:
        return CheckResult("V34", FAIL,
                           "train.py --smoke emitted no SMOKE_LOSSES line",
                           {"tail_a": a["tail"], "tail_b": b["tail"]})
    if len(a["losses"]) < 2:
        return CheckResult("V34", FAIL,
                           f"smoke run produced only {len(a['losses'])} loss value(s); a "
                           "single step cannot demonstrate reproducibility",
                           {"losses": a["losses"]})
    if a["losses"] != b["losses"]:
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a["losses"], b["losses"])) if x != y]
        return CheckResult("V34", FAIL,
                           f"seeded smoke runs diverge at {len(diff)} of "
                           f"{len(a['losses'])} steps",
                           {"first_divergences": diff[:5]})
    if a["digest"] != b["digest"]:
        return CheckResult("V34", FAIL,
                           "loss values match but the run digests differ, so something "
                           "outside the loss is non-deterministic",
                           {"digest_a": a["digest"], "digest_b": b["digest"]})
    return CheckResult("V34", PASS,
                       f"two seeded smoke runs identical across {len(a['losses'])} steps",
                       {"steps": len(a["losses"]), "digest": a["digest"],
                        "first_loss": a["losses"][0], "last_loss": a["losses"][-1]})


def check_V35(ctx: Ctx) -> CheckResult:
    ck = ctx.p("weights", "best.pt")
    if not ck.exists():
        return not_impl("V35", "weights/best.pt")
    if _is_stub(ctx.p("src", "model.py")):
        return not_impl("V35", "src/model.py")
    import torch  # local: the verifier's own import cost is not scored

    # weights_only=True deliberately: it is what inference.py uses, so a checkpoint that
    # required arbitrary unpickling would pass a lax V35 and then break the shipped script.
    # It is also the safe load path -- torch.load with weights_only=False executes pickle.
    d = torch.load(ck, map_location="cpu", weights_only=True)
    if not isinstance(d, dict):
        return CheckResult("V35", FAIL,
                           f"weights/best.pt holds {type(d).__name__}, not a checkpoint dict")
    required = ["model", "ema", "config", "iter", "metrics", "git"]
    missing = [k for k in required if k not in d]
    if missing:
        return CheckResult("V35", FAIL, f"checkpoint missing keys: {missing}",
                           {"present": sorted(d.keys())})
    mod = _import_project(ctx, "src.model")
    m = mod.build_model(d["config"])
    m.load_state_dict(d["model"], strict=True)
    ev = {"keys": sorted(d.keys()), "iter": d.get("iter"), "git": d.get("git"),
          "strict_load": "model"}
    # The shipped weights are the EMA ones, so those must load strictly too when present.
    if d.get("ema"):
        mod.build_model(d["config"]).load_state_dict(d["ema"], strict=True)
        ev["strict_load"] = "model+ema"
    return CheckResult("V35", PASS,
                       "checkpoint self-describes and build_model(ckpt['config']) loads it "
                       "with strict=True", ev)


def check_V36(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V36", "inference.py")
    src = ctx.read("inference.py")
    bad = [t for t in (".backward(", "requires_grad_(True)", "optim.", "Optimizer(") if t in src]
    if bad:
        return CheckResult("V36", FAIL, f"training constructs in inference.py: {bad}", {"found": bad})
    return CheckResult("V36", PASS, "no optimizer, backward, or grad enabling in inference.py")


# ======================================================================================
# TIER 3
# ======================================================================================
def check_V37(ctx: Ctx) -> CheckResult:
    rp = ctx.p("results", "runtime_report.md")
    if not rp.exists():
        return not_impl("V37", "results/runtime_report.md")
    txt = rp.read_text(encoding="utf-8", errors="replace").lower()
    need = ["img/s", "batch", "precision", "torch", "timing"]
    missing = [n for n in need if n not in txt]
    if missing:
        return CheckResult("V37", FAIL, f"runtime_report.md missing: {missing}")
    return CheckResult("V37", PASS, "runtime report present with required fields")


def check_V38(ctx: Ctx) -> CheckResult:
    p = ctx.p("scripts", "benchmark_runtime.py")
    if _is_stub(p):
        return not_impl("V38", "scripts/benchmark_runtime.py")
    src = p.read_text(encoding="utf-8")
    if "subprocess" not in src:
        return CheckResult("V38", FAIL, "timing is not external to the process")
    return CheckResult("V38", PASS, "timing wraps the whole process externally")


def check_V39(ctx: Ctx) -> CheckResult:
    """Revised (docs/decisions.md D6/D10): measure and report. No threshold. NO SKIP."""
    rp = ctx.p("results", "runtime_report.md")
    if not rp.exists():
        return not_impl("V39", "results/runtime_report.md")
    txt = rp.read_text(encoding="utf-8", errors="replace")
    low = txt.lower()
    has_total = ("total" in low and "wall" in low)
    has_breakdown = ("startup" in low and "compute" in low)
    has_device = bool(re.search(r"(device|gpu|cpu)\s*[:|]", low))
    missing = []
    if not has_total:
        missing.append("total end-to-end wall-clock")
    if not has_breakdown:
        missing.append("startup-vs-compute breakdown")
    if not has_device:
        missing.append("device label")
    if missing:
        return CheckResult("V39", FAIL, f"runtime_report.md missing: {missing}")
    return CheckResult("V39", PASS, "wall-clock measured, broken down, and device-labelled")


def check_V40(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V40", "inference.py")
    src = ctx.read("inference.py")
    need = {
        "channels_last": "channels_last",
        "inference_mode": "inference_mode",
        "tf32": "allow_tf32",
        "cudnn_benchmark": "cudnn.benchmark",
        "amp": "autocast",
        "threaded_writes": "ThreadPoolExecutor",
    }
    missing = [k for k, v in need.items() if v not in src]
    if missing:
        return CheckResult("V40", FAIL, f"default optimizations absent: {missing}",
                           {"missing": missing})
    return CheckResult("V40", PASS, "free optimizations enabled by default")


def check_V41(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V41", "inference.py")
    src = ctx.read("inference.py")
    if "torch.compile" not in src:
        return CheckResult("V41", PASS, "torch.compile not used at all")
    if 'action="store_true"' not in src and "action='store_true'" not in src:
        return CheckResult("V41", FAIL, "torch.compile present without an opt-in flag")
    if re.search(r'--compile["\'].*default\s*=\s*True', src):
        return CheckResult("V41", FAIL, "--compile defaults to True")
    return CheckResult("V41", PASS, "torch.compile is opt-in and off by default")


def check_V42(ctx: Ctx) -> CheckResult:
    tree = ctx.inference_ast()
    if tree is None:
        return not_impl("V42", "inference.py")
    src = ctx.read("inference.py")
    if "tta" not in src.lower():
        return CheckResult("V42", PASS, "no TTA implemented")
    if re.search(r'--tta["\'].*default\s*=\s*True', src):
        return CheckResult("V42", FAIL, "--tta defaults to True")
    return CheckResult("V42", PASS, "TTA is flag-gated and off by default")


def check_V43(ctx: Ctx) -> CheckResult:
    rp = ctx.p("results", "runtime_report.md")
    ck = ctx.p("weights", "best.pt")
    if not ck.exists():
        return not_impl("V43", "weights/best.pt")
    size_mb = ck.stat().st_size / (1024 * 1024)
    if size_mb >= 100:
        return CheckResult("V43", FAIL, f"checkpoint {size_mb:.1f} MB >= 100 MB (needs LFS/hosting)",
                           {"size_mb": size_mb})
    if not rp.exists() or "param" not in rp.read_text(encoding="utf-8", errors="replace").lower():
        return CheckResult("V43", FAIL, "param count / checkpoint size not recorded in runtime_report.md")
    return CheckResult("V43", PASS, f"checkpoint {size_mb:.1f} MB, params recorded")


# ======================================================================================
# TIER 4
# ======================================================================================
def check_V44(ctx: Ctx) -> CheckResult:
    p = ctx.p("train.py")
    u = ctx.p("src", "utils.py")
    if _is_stub(p) and _is_stub(u):
        return not_impl("V44", "train.py / src/utils.py seeding")
    src = (p.read_text(encoding="utf-8") if p.exists() else "") + \
          (u.read_text(encoding="utf-8") if u.exists() else "")
    need = ["random.seed", "np.random.seed", "torch.manual_seed", "cuda.manual_seed"]
    missing = [n for n in need if n not in src]
    if missing:
        return CheckResult("V44", FAIL, f"seeding incomplete: {missing}", {"missing": missing})
    return CheckResult("V44", PASS, "all four RNGs seeded")


def check_V45(ctx: Ctx) -> CheckResult:
    p = ctx.p("results", "experiments.csv")
    if not p.exists():
        return not_impl("V45", "results/experiments.csv")
    rows = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 3:
        return CheckResult("V45", FAIL, f"{max(0, len(rows)-1)} data rows, need >= 2")
    return CheckResult("V45", PASS, f"{len(rows)-1} runs logged")


def check_V46(ctx: Ctx) -> CheckResult:
    if not ctx.exists("README.md"):
        return not_impl("V46", "README.md")
    cmds = re.findall(r"```(?:bash|sh|shell|powershell)\n(.*?)```", ctx.read("README.md"), re.S)
    if not cmds:
        return CheckResult("V46", FAIL, "no fenced shell commands found in README.md")
    if not ctx.args.fresh_clone:
        return CheckResult("V46", FAIL, "requires --fresh-clone to execute README commands")
    return _fresh_clone_run(ctx, "V46")


def check_V47(ctx: Ctx) -> CheckResult:
    si = ctx.p("sample_inputs")
    files = [p for p in si.glob("*") if p.is_file() and p.name != ".gitkeep"] if si.exists() else []
    if not files:
        return not_impl("V47", "sample_inputs/ (4-6 small degraded images)")
    if _is_stub(ctx.p("inference.py")):
        return not_impl("V47", "inference.py")
    import time as _t
    out = ctx.tmpdir("v47") / "o"
    t0 = _t.perf_counter()
    rc, _, se = ctx.run_inference(si, out, timeout=120)
    dt = _t.perf_counter() - t0
    if rc != 0:
        return CheckResult("V47", FAIL, f"exit {rc}: {se.strip()[-200:]}")
    if dt >= 60:
        return CheckResult("V47", FAIL, f"took {dt:.1f}s, budget 60s", {"seconds": dt})
    return CheckResult("V47", PASS, f"{len(files)} samples in {dt:.1f}s", {"seconds": dt})


def check_V48(ctx: Ctx) -> CheckResult:
    p = ctx.p("results", "metrics_summary.md")
    if not p.exists():
        return not_impl("V48", "results/metrics_summary.md")
    txt = p.read_text(encoding="utf-8", errors="replace").lower()
    rows = [l for l in txt.splitlines() if l.strip().startswith("|")]
    if len(rows) < 6:
        return CheckResult("V48", FAIL, "fewer than 3 baselines + final in the table")
    return CheckResult("V48", PASS, "results table present")


def check_V49(ctx: Ctx) -> CheckResult:
    q = ctx.p("results", "qualitative")
    files = [p for p in q.glob("*") if p.is_file() and p.name != ".gitkeep"] if q.exists() else []
    if len(files) < 5:
        return not_impl("V49", "results/qualitative/ (>=4 successes + >=1 labelled failure)")
    if not any("fail" in p.name.lower() for p in files):
        return CheckResult("V49", FAIL, "no labelled failure case present")
    return CheckResult("V49", PASS, f"{len(files)} qualitative artifacts")


def check_V50(ctx: Ctx) -> CheckResult:
    if not ctx.exists("README.md"):
        return not_impl("V50", "README.md")
    txt = ctx.read("README.md")
    low = txt.lower()
    if "none used" in low:
        return CheckResult("V50", PASS, "explicit empty disclosure present")
    if "licence" in low or "license" in low:
        return CheckResult("V50", PASS, "external-resources section present")
    return CheckResult("V50", FAIL,
                       "no external-resources disclosure; Phase 1 must say 'No external "
                       "datasets or pretrained weights used.' (docs/decisions.md D13)")


#: The only place a tracked ``.npy`` is permitted, and the bound on that permission.
#: SPEC section 12 requires ``sample_inputs/`` and V47 runs inference against it from a
#: clean clone, so those files MUST be in the clone. The exemption is deliberately narrow.
SAMPLE_INPUTS_MAX_FILES = 8
SAMPLE_INPUTS_MAX_BYTES = 512 * 1024

#: Blob extensions that may never be tracked. Broader than the original four.
BLOB_EXTS = (".npz", ".pt", ".pth", ".zip", ".env", ".tar", ".gz", ".7z", ".ckpt",
             ".onnx", ".safetensors", ".bin", ".raw", ".dat", ".h5", ".hdf5",
             ".parquet", ".mat", ".pkl", ".pickle")

#: Path segments that would mean a slice of the dataset tree got committed.
DATASET_DIR_TOKENS = ("/gt/", "/noisylr/", "/ground_truth/", "/test_noisylr/")

#: No single tracked file may exceed this, and no repo may exceed the total. Catches a
#: dataset dump under ANY extension, which an extension blacklist alone cannot.
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024
MAX_TRACKED_TOTAL_BYTES = 25 * 1024 * 1024


def check_V51(ctx: Ctx) -> CheckResult:
    if not ctx.exists(".gitignore"):
        return CheckResult("V51", FAIL, ".gitignore missing")
    rc, so, _ = ctx.run(["git", "ls-files"])
    tracked = so.splitlines()

    junk = [f for f in tracked
            if f.endswith(BLOB_EXTS) or "__pycache__" in f or ".ipynb_checkpoints" in f
            or f.endswith(".DS_Store")]
    # .npy is banned everywhere EXCEPT the bounded sample_inputs/ exemption below.
    junk += [f for f in tracked
             if f.endswith(".npy") and not f.startswith("sample_inputs/")]
    junk = [f for f in junk if not f.startswith("tests/fixtures/")]
    if junk:
        return CheckResult("V51", FAIL, f"{len(junk)} junk/dataset files tracked",
                           {"files": sorted(set(junk))[:10]})

    # A committed slice of the dataset tree, under any extension.
    tree = [f for f in tracked
            if any(t in ("/" + f.lower() + "/") for t in DATASET_DIR_TOKENS)]
    if tree:
        return CheckResult("V51", FAIL,
                           f"{len(tree)} tracked files sit inside a dataset directory",
                           {"files": sorted(tree)[:10]})

    # The sample_inputs/ exemption is bounded in both count and total bytes.
    samples = [f for f in tracked if f.startswith("sample_inputs/") and f.endswith(".npy")]
    sample_bytes = 0
    for f in samples:
        p = ctx.p(*f.split("/"))
        if p.exists():
            sample_bytes += p.stat().st_size
    if len(samples) > SAMPLE_INPUTS_MAX_FILES:
        return CheckResult("V51", FAIL,
                           f"sample_inputs/ has {len(samples)} .npy files, "
                           f"cap is {SAMPLE_INPUTS_MAX_FILES}",
                           {"count": len(samples), "bytes": sample_bytes})
    if sample_bytes > SAMPLE_INPUTS_MAX_BYTES:
        return CheckResult("V51", FAIL,
                           f"sample_inputs/ totals {sample_bytes} B, "
                           f"cap is {SAMPLE_INPUTS_MAX_BYTES} B",
                           {"count": len(samples), "bytes": sample_bytes})

    # Size caps catch a dataset dump regardless of extension.
    total = 0
    oversized = []
    for f in tracked:
        p = ctx.p(*f.split("/"))
        if not p.exists():
            continue
        sz = p.stat().st_size
        total += sz
        if sz > MAX_TRACKED_FILE_BYTES:
            oversized.append(f"{f} ({sz} B)")
    if oversized:
        return CheckResult("V51", FAIL,
                           f"{len(oversized)} tracked files exceed "
                           f"{MAX_TRACKED_FILE_BYTES} B", {"files": oversized[:10]})
    if total > MAX_TRACKED_TOTAL_BYTES:
        return CheckResult("V51", FAIL,
                           f"tracked tree is {total} B, cap is {MAX_TRACKED_TOTAL_BYTES} B",
                           {"total_bytes": total})

    keys = []
    for f in tracked:
        p = ctx.p(*f.split("/"))
        if p.suffix in {".py", ".md", ".txt", ".yaml", ".yml"} and p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})", t):
                keys.append(f)
    if keys:
        return CheckResult("V51", FAIL, f"possible secrets in {keys}")
    return CheckResult("V51", PASS,
                       f"{len(tracked)} tracked files ({total} B), no secrets or dataset "
                       f"blobs; sample_inputs/ = {len(samples)} files, {sample_bytes} B",
                       {"tracked": len(tracked), "total_bytes": total,
                        "sample_count": len(samples), "sample_bytes": sample_bytes})


def check_V52(ctx: Ctx) -> CheckResult:
    need = ["docs/STATE.md", "docs/dataset_findings.md", "docs/io_contract.md", "docs/decisions.md"]
    missing = [n for n in need if not ctx.exists(*n.split("/"))]
    if missing:
        return CheckResult("V52", FAIL, f"missing docs: {missing}", {"missing": missing})
    stubs = [n for n in need if ctx.p(*n.split("/")).stat().st_size < 200]
    if stubs:
        return CheckResult("V52", FAIL, f"stub docs: {stubs}", {"stubs": stubs})
    findings = ctx.read("docs", "dataset_findings.md") + ctx.read("docs", "decisions.md")
    blockers = ctx.read("docs", "BLOCKERS.md") if ctx.exists("docs", "BLOCKERS.md") else ""
    unanswered = [f"U{i}" for i in range(1, 10)
                  if f"U{i}" not in findings and f"U{i}" not in blockers]
    if unanswered:
        return CheckResult("V52", FAIL, f"U-items neither answered nor in BLOCKERS: {unanswered}",
                           {"unanswered": unanswered})
    return CheckResult("V52", PASS, "docs current; U1-U9 all accounted for")


# ======================================================================================
# RUNNER
# ======================================================================================
def discover() -> dict[str, Callable[[Ctx], CheckResult]]:
    g = globals()
    return {n[len("check_"):]: g[n] for n in sorted(g) if n.startswith("check_V")}


def run_one(cid: str, fn: Callable[[Ctx], CheckResult], ctx: Ctx) -> CheckResult:
    """Wrap every check: an exception is a FAIL with the traceback, never a crash."""
    try:
        r = fn(ctx)
        if not isinstance(r, CheckResult):
            return CheckResult(cid, FAIL, f"check returned {type(r).__name__}, not CheckResult")
        if r.status == SKIP and cid not in SKIP_WHITELIST:
            return CheckResult(cid, FAIL,
                               f"SKIP not permitted for {cid} (not in whitelist). "
                               f"original detail: {r.detail}", r.evidence)
        return r
    except Exception:  # noqa: BLE001
        return CheckResult(cid, FAIL, "exception in check:\n" + traceback.format_exc()[-1500:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the KLA verification contract.")
    ap.add_argument("--strict", action="store_true", help="fail on un-whitelisted SKIP")
    ap.add_argument("--only", default=None, help="comma-separated check IDs, e.g. V07,V19")
    ap.add_argument("--tier", type=int, default=None, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--fresh-clone", action="store_true", dest="fresh_clone")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--json", default="results/verification_report.json")
    args = ap.parse_args(argv)

    checks = discover()
    selected = sorted(checks)
    if args.only:
        want = {s.strip().upper() for s in args.only.split(",") if s.strip()}
        selected = [c for c in selected if c in want]
        unknown = want - set(checks)
        if unknown:
            print(f"unknown check ids: {sorted(unknown)}", file=sys.stderr)
            return 2
    if args.tier is not None:
        selected = [c for c in selected if TIERS.get(c) == args.tier]

    fixtures = build_fixtures(REPO_ROOT)
    ctx = Ctx(root=REPO_ROOT, args=args, fixtures=fixtures)

    results = [run_one(cid, checks[cid], ctx) for cid in selected]

    rc, so, _ = ctx.run(["git", "rev-parse", "--short", "HEAD"])
    commit = so.strip() if rc == 0 else "unknown"

    n_pass = sum(1 for r in results if r.status == PASS)
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_skip = sum(1 for r in results if r.status == SKIP)

    width = max((len(r.detail.splitlines()[0]) if r.detail else 0) for r in results) if results else 0
    width = min(max(width, 20), 88)
    print()
    print(f"{'ID':<5} {'TIER':<5} {'STATUS':<7} DETAIL")
    print("-" * (20 + width))
    for r in results:
        d = (r.detail.splitlines()[0] if r.detail else "")[:width]
        print(f"{r.id:<5} {TIERS.get(r.id, '?'):<5} {r.status:<7} {d}")
    print("-" * (20 + width))
    print(f"implemented={len(checks)}  run={len(results)}  "
          f"PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  commit={commit}")

    by_tier: dict[int, dict[str, int]] = {}
    for r in results:
        t = TIERS.get(r.id, -1)
        by_tier.setdefault(t, {PASS: 0, FAIL: 0, SKIP: 0})[r.status] += 1
    print("per tier: " + "  ".join(
        f"T{t}[P{v[PASS]}/F{v[FAIL]}/S{v[SKIP]}]" for t, v in sorted(by_tier.items())))

    report = {
        "iteration": args.iteration,
        "commit": commit,
        "checks": [{"id": r.id, "status": r.status, "detail": r.detail, "evidence": r.evidence}
                   for r in results],
        "summary": {
            "implemented": len(checks), "run": len(results),
            "pass": n_pass, "fail": n_fail, "skip": n_skip,
            "strict": bool(args.strict), "fresh_clone": bool(args.fresh_clone),
            "by_tier": {str(k): v for k, v in sorted(by_tier.items())},
        },
    }
    jp = REPO_ROOT / args.json
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {jp}")

    if n_fail:
        return 1
    if args.strict and n_skip:
        unwl = [r.id for r in results if r.status == SKIP and r.id not in SKIP_WHITELIST]
        if unwl:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
