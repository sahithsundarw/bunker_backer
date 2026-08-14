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
    bad = []
    for sub in ("mixed", "size256"):
        src = ctx.fixtures / sub
        early, ind, out = _io_roundtrip(ctx, src, "V09")
        if early:
            return early
        for p in ind.rglob("*.npy"):
            si = _npy_shape(p)
            q = out / p.relative_to(ind)
            so = _npy_shape(q) if q.exists() else None
            if si is None or so is None or len(si) < 2 or len(so) < 2:
                bad.append({"file": p.name, "in": si, "out": so})
            elif (so[0], so[1]) != (2 * si[0], 2 * si[1]):
                bad.append({"file": p.name, "in": si, "out": so})
    if bad:
        return CheckResult("V09", FAIL, f"{len(bad)} outputs are not exactly 2x", {"violations": bad[:5]})
    return CheckResult("V09", PASS, "every output is exactly 2x its input")


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


def check_V25(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V25", "train.py", "src/model.py")
    return r or CheckResult("V25", FAIL, "overfit-2-pairs harness not wired up yet")


def check_V26(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V26", "src/dataset.py")
    return r or CheckResult("V26", FAIL, "paired-crop marker test not wired up yet")


def check_V27(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V27", "scripts/evaluate.py", "src/metrics.py")
    return r or CheckResult("V27", FAIL, "no metrics_summary.md to compare against bicubic")


def check_V28(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V28", "scripts/make_baselines.py")
    return r or CheckResult("V28", FAIL, "U-Net baseline comparison not available yet")


def check_V29(ctx: Ctx) -> CheckResult:
    sv = ctx.p("configs", "split_val.txt")
    if not sv.exists():
        return not_impl("V29", "configs/split_val.txt")
    names = [l.strip() for l in sv.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not names:
        return CheckResult("V29", FAIL, "configs/split_val.txt has no file entries")
    if _is_stub(ctx.p("src", "dataset.py")):
        return not_impl("V29", "src/dataset.py train-list intersection")
    return CheckResult("V29", FAIL, "train/val intersection not verifiable yet")


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
    return CheckResult("V32", FAIL, "single-channel end-to-end not verifiable until model exists")


def check_V33(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V33", "src/degrade.py")
    return r or CheckResult("V33", FAIL, "degradation-fidelity figure not produced yet")


def check_V34(ctx: Ctx) -> CheckResult:
    r = _needs(ctx, "V34", "train.py")
    return r or CheckResult("V34", FAIL, "seeded smoke run not reproducible yet")


def check_V35(ctx: Ctx) -> CheckResult:
    ck = ctx.p("weights", "best.pt")
    if not ck.exists():
        return not_impl("V35", "weights/best.pt")
    return CheckResult("V35", FAIL, "checkpoint key/strict-load check not wired up yet")


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


def check_V51(ctx: Ctx) -> CheckResult:
    if not ctx.exists(".gitignore"):
        return CheckResult("V51", FAIL, ".gitignore missing")
    rc, so, _ = ctx.run(["git", "ls-files"])
    tracked = so.splitlines()
    junk = [f for f in tracked
            if f.endswith((".npy", ".npz", ".pt", ".pth", ".zip", ".env"))
            or "__pycache__" in f or ".ipynb_checkpoints" in f]
    junk = [f for f in junk if not f.startswith("tests/fixtures/")]
    if junk:
        return CheckResult("V51", FAIL, f"{len(junk)} junk/dataset files tracked", {"files": junk[:10]})
    keys = []
    for f in tracked:
        p = ctx.p(*f.split("/"))
        if p.suffix in {".py", ".md", ".txt", ".yaml", ".yml"} and p.exists():
            t = p.read_text(encoding="utf-8", errors="replace")
            if re.search(r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})", t):
                keys.append(f)
    if keys:
        return CheckResult("V51", FAIL, f"possible secrets in {keys}")
    return CheckResult("V51", PASS, f"{len(tracked)} tracked files, no secrets or dataset blobs")


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
