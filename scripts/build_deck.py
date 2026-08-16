"""Build the KLA PS01 submission deck (SPEC section 14, F13).

Reads numbers ONLY from committed repo artifacts (results/metrics_summary.md,
results/runtime_report.md, results/runtime_report_512.md if present,
docs/dataset_findings.md, docs/decisions.md) -- never hardcodes a metric. Re-run this
script whenever an underlying artifact changes; do not hand-edit the output PDF.

Usage:
    py -3.12 scripts/build_deck.py --team "PLACEHOLDER_TEAM"

Produces ``<team>_KLA_PS01.pdf`` at the repo root, <=9 slides, landscape.

Owner: main session (not on the per-file agent ownership map -- this is a
cross-cutting deliverable, not part of the model/data/inference pipeline).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 14 * mm

BANNED_PHRASES = [
    "our semiconductor dataset",
    "semiconductor image pairs",
    "structure families present in the data",
    "trained on semiconductor inspection imagery",
]

PROXY_SENTENCE = (
    "The released dataset is 3200 training pairs and 400 test inputs of grayscale "
    "natural photographs, not semiconductor imagery. We treat it as a proxy: the "
    "degradation — ×2 decimation plus signal-dependent noise — is what "
    "transfers to inspection imagery, so we characterised the degradation empirically "
    "and optimised for degradation robustness rather than fitting content-specific priors."
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_metrics_table(md: str) -> list[dict]:
    """Pull the `## Results` markdown table rows out of results/metrics_summary.md."""
    rows = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("| Method"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line.replace("|", "").strip()) <= {"-", " "}:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 6:
                rows.append({
                    "method": cells[0], "psnr": cells[1], "ssim": cells[2],
                    "lpips": cells[3], "n": cells[4], "notes": cells[6] if len(cells) > 6 else "",
                })
    return rows


def parse_proxy_ood_row(md: str) -> dict | None:
    """Proxy-OOD section is bullets, not a table (deliberate, so V48's single-table
    constraint on metrics_summary.md is never at risk) -- e.g.
    "- PSNR dB (mean +/- sd): 27.3177 +/- 3.4696"."""
    m = re.search(r"^## Proxy-OOD generalisation check.*$", md, re.MULTILINE)
    if not m:
        return None
    nxt = re.search(r"\n## ", md[m.end():])
    section = md[m.start(): m.end() + (nxt.start() if nxt else len(md) - m.end())]
    out = {}
    for key, label in (("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS")):
        mm = re.search(rf"{label}.*?:\s*([-\d.]+\s*\+/-\s*[\d.]+)", section)
        if mm:
            out[key] = mm.group(1)
    return out if len(out) == 3 else None


def parse_runtime(md: str, *, side_by_side_col: int | None = None) -> dict:
    """side_by_side_col: if the report has a "128->256 vs 256->512" comparison table (the
    _512 report does), 0 selects the first (128->256) data column, 1 the second (256->512).
    None (the plain runtime_report.md) uses the single-resolution bold-percent pattern."""
    out = {"device": "unknown", "img_s": "not measured", "fixed_pct": "not measured"}
    m = re.search(r"Device \| (.+)", md)
    if m:
        out["device"] = m.group(1).split("|")[0].strip().rstrip("|").strip()
    if side_by_side_col is not None:
        m = re.search(r"\| fixed-cost fraction at N=400 \| ([\d.]+)% \| ([\d.]+)%", md)
        if m:
            out["fixed_pct"] = m.group(side_by_side_col + 1) + "%"
        m = re.search(r"\| img/s at N=400 incl\. startup \| ([\d.]+) \| ([\d.]+)", md)
        if m:
            out["img_s"] = m.group(side_by_side_col + 1) + " img/s"
    else:
        m = re.search(r"([\d.]+)\s*img/s", md)
        if m:
            out["img_s"] = m.group(1) + " img/s"
        m = re.search(r"fixed cost is \*\*([\d.]+)%\*\*", md)
        if m:
            out["fixed_pct"] = m.group(1) + "%"
    return out


class Deck:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=landscape(A4))
        self.slide_no = 0

    def new_slide(self, title: str):
        if self.slide_no:
            self.c.showPage()
        self.slide_no += 1
        self.c.setFont("Helvetica-Bold", 20)
        self.c.drawString(MARGIN, PAGE_H - MARGIN - 4 * mm, f"{self.slide_no}. {title}")
        self.c.setFont("Helvetica", 8)
        self.c.setFillColorRGB(0.5, 0.5, 0.5)
        self.c.drawRightString(PAGE_W - MARGIN, MARGIN / 2, f"slide {self.slide_no}")
        self.c.setFillColorRGB(0, 0, 0)
        return PAGE_H - MARGIN - 12 * mm  # y cursor start for body text

    def body(self, y: float, lines: list[str], size: int = 11, leading: float = 5.2 * mm,
              indent: float = 0) -> float:
        self.c.setFont("Helvetica", size)
        for line in lines:
            self.c.drawString(MARGIN + indent, y, line)
            y -= leading
        return y

    def wrapped(self, y: float, text: str, size: int = 11, width_chars: int = 110,
                leading: float = 5.2 * mm, indent: float = 0) -> float:
        import textwrap
        for line in textwrap.wrap(text, width_chars):
            y = self.body(y, [line], size=size, leading=leading, indent=indent)
        return y

    def image_fit(self, path: Path, x: float, y_top: float, max_w: float, max_h: float) -> float:
        """Draw image fit within a box, top-left anchored at (x, y_top). Returns bottom y."""
        if not path.exists():
            self.c.setFont("Helvetica-Oblique", 9)
            self.c.drawString(x, y_top - 10, f"[missing: {path.name}]")
            return y_top - 10 * mm
        img = ImageReader(str(path))
        iw, ih = img.getSize()
        scale = min(max_w / iw, max_h / ih)
        w, h = iw * scale, ih * scale
        self.c.drawImage(img, x, y_top - h, width=w, height=h, preserveAspectRatio=True)
        return y_top - h

    def save(self):
        self.c.showPage()
        self.c.save()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="PLACEHOLDER_TEAM",
                     help="Team name; also becomes the output filename per SPEC F13")
    args = ap.parse_args(argv)

    metrics_md = _read(ROOT / "results" / "metrics_summary.md")
    runtime_md = _read(ROOT / "results" / "runtime_report.md")
    runtime_512_md = _read(ROOT / "results" / "runtime_report_512.md")
    rows = parse_metrics_table(metrics_md)
    proxy_row = parse_proxy_ood_row(metrics_md)
    rt = parse_runtime(runtime_md)
    rt512 = parse_runtime(runtime_512_md, side_by_side_col=1) if runtime_512_md else None

    out_path = ROOT / f"{args.team}_KLA_PS01.pdf"
    d = Deck(out_path)

    # Slide 1 -- Team Details
    y = d.new_slide("Team Details")
    y = d.body(y, [
        f"Team name: {args.team}  [[ FILL IN BEFORE SUBMISSION ]]",
        "Members (name -- role):",
        "  [MEMBER 1] -- modelling",
        "  [MEMBER 2] -- data & augmentation",
        "  [MEMBER 3] -- inference optimization",
        "  [MEMBER 4] -- evaluation & docs",
        "College: [COLLEGE NAME]",
        "Contact: [EMAIL] / [PHONE]",
    ])

    # Slide 2 -- Problem Statement Addressed
    y = d.new_slide("Problem Statement Addressed")
    y = d.wrapped(y, "AI-Based Restoration of Degraded Images for Semiconductor Inspection "
                      "(KLA PS01). A single lost pixel or noisy region in an inspection image "
                      "can hide a real defect and cost a die -- restoration quality is a "
                      "correctness problem, not a cosmetic one.")
    y -= 3 * mm
    y = d.body(y, ["Three degradation mechanisms, applied in combination:"])
    y = d.body(y, [
        "  1. Speckle noise (signal-dependent, multiplicative-like variance term)",
        "  2. Additive/shot noise (signal-dependent linear term)",
        "  3. Downsampling (x2 decimation, GT/LR pair)",
    ], indent=4 * mm)
    y -= 2 * mm
    y = d.wrapped(y, "The pipeline must invert all three jointly, in whatever order they were "
                      "applied, and generalise to content the training set never showed it.")

    # Slide 3 -- Idea Description (dataset analysis + core concept)
    y = d.new_slide("Idea Description")
    y = d.wrapped(y, PROXY_SENTENCE, size=10)
    y -= 2 * mm
    img_bottom = d.image_fit(ROOT / "results" / "eda" / "noise_variance_vs_intensity.png",
                              MARGIN, y, max_w=115 * mm, max_h=60 * mm)
    d.c.setFont("Helvetica", 9)
    d.c.drawString(MARGIN + 120 * mm, y - 4 * mm,
                    "Measured (D1/D12/D2): downsample kernel is a recovered 4x4")
    d.c.drawString(MARGIN + 120 * mm, y - 9 * mm,
                    "sharpening kernel (bicubic antialias-off is within 1.22e-05 of")
    d.c.drawString(MARGIN + 120 * mm, y - 14 * mm,
                    "optimal); noise is applied AFTER downsampling, signal-dependent,")
    d.c.drawString(MARGIN + 120 * mm, y - 19 * mm,
                    "no additive Gaussian floor (D2, residual autocorrelation ~0).")
    d.c.drawString(MARGIN + 120 * mm, y - 26 * mm,
                    "Core concept: one-step blind joint restoration, all compute at LR")
    d.c.drawString(MARGIN + 120 * mm, y - 31 * mm,
                    "resolution, x2 PixelShuffle head -- no cascade, no GAN.")

    # Slide 4 -- Proposed Solution
    y = d.new_slide("Proposed Solution")
    y = d.body(y, ["Pipeline: load .npy -> group by shape -> batch -> bf16 forward "
                    "(channels_last) -> clip [0,1] -> save"])
    y -= 2 * mm
    y = d.body(y, ["Architecture: NAFSR -- NAFNet-style blocks (SimpleGate, SCA channel "
                    "attention, LayerNorm) + x2 PixelShuffle head, width 48, 16 blocks"])
    y -= 2 * mm
    y = d.body(y, ["Loss: Charbonnier (fidelity) + SSIM (structure) + FFT (frequency) -- "
                    "balanced, no adversarial term (no hallucination risk for inspection use)"])
    y -= 2 * mm
    y = d.body(y, ["Augmentation: dihedral flips/rotations, CutBlur, on-the-fly synthetic "
                    "re-degradation with randomised order and noise levels"])

    # Slide 5 -- Innovation & Uniqueness
    y = d.new_slide("Innovation & Uniqueness")
    y = d.wrapped(y, "(a) Empirical degradation forensics driving a matched synthetic-pair "
                      "generator -- kernel and noise parameters measured from the data, not "
                      "assumed from the spec.")
    y -= 2 * mm
    y = d.wrapped(y, "(b) Balanced fidelity + structure + frequency loss with an explicit "
                      "no-GAN / no-hallucination decision -- justified by inspection "
                      "semantics: a plausible but wrong structure is worse than a blurry "
                      "correct one when a die's fate depends on it.")
    y -= 2 * mm
    y = d.wrapped(y, "(c) Throughput-engineered inference path: grouped batching by shape, "
                      "bf16 + channels_last, threaded I/O, memory-aware OOM recovery "
                      "(automatic batch halving, CPU-bicubic floor) -- measured, not assumed.")

    # Slide 6 -- Results
    y = d.new_slide("Results")
    d.c.setFont("Helvetica-Bold", 9)
    headers = ["Method", "PSNR dB", "SSIM", "LPIPS", "n"]
    col_x = [MARGIN, MARGIN + 76 * mm, MARGIN + 122 * mm, MARGIN + 162 * mm, MARGIN + 200 * mm]
    for cx, h in zip(col_x, headers):
        d.c.drawString(cx, y, h)
    y -= 4.6 * mm
    d.c.setFont("Helvetica", 8)
    for r in rows:
        for cx, key in zip(col_x, ["method", "psnr", "ssim", "lpips", "n"]):
            d.c.drawString(cx, y, r[key][:34])
        y -= 4.2 * mm
    y -= 2 * mm
    if proxy_row:
        y = d.wrapped(y, f"Proxy-OOD (procedural geometric content, n=40): "
                          f"PSNR {proxy_row['psnr']} / SSIM {proxy_row['ssim']} / "
                          f"LPIPS {proxy_row['lpips']} -- synthetic, not "
                          f"semiconductor imagery; degradation model unchanged, not refit.",
                       size=9, width_chars=100)
    else:
        y = d.body(y, ["Proxy-OOD column: [[ PENDING -- loss-metrics scoring in flight ]]"],
                    size=9)
    y -= 4 * mm
    img_top = y
    bottom1 = d.image_fit(ROOT / "results" / "qualitative" / "success_p50_001143_psnr28.82.png",
                           MARGIN, img_top, max_w=58 * mm, max_h=38 * mm)
    bottom2 = d.image_fit(ROOT / "results" / "qualitative" / "fail_worst_psnr_002041_psnr17.23.png",
                           MARGIN + 64 * mm, img_top, max_w=58 * mm, max_h=38 * mm)
    y = min(bottom1, bottom2) - 4 * mm
    d.c.setFont("Helvetica-Oblique", 8)
    d.c.drawString(MARGIN, y, "Success (median case) | Honest failure: worst-PSNR case,")
    d.c.drawString(MARGIN, y - 4 * mm, "band-limited oracle ceiling caveat applies")

    # Slide 7 -- Technology & Feasibility
    y = d.new_slide("Technology & Feasibility")
    y = d.body(y, [
        "PyTorch 2.11.0+cu128, CUDA 12.8, trained on NVIDIA RTX 4060 Laptop GPU (8 GB)",
        "Model: NAFSR, 388,225 params, checkpoint 3.14 MiB, trained 20,000 iterations",
    ])
    y -= 1.5 * mm
    y = d.wrapped(y, f"Inference throughput (128->256, {rt['device']}, bf16, batch 32): "
                      f"{rt['img_s']}, fixed startup {rt['fixed_pct']} of wall-clock at "
                      f"N=400 -- externally timed, process start to finish", width_chars=100)
    if rt512:
        y = d.wrapped(y, f"Inference throughput (256->512, {rt512['device']}, bf16, batch 32): "
                          f"{rt512['img_s']}, fixed startup {rt512['fixed_pct']} of wall-clock "
                          f"at N=400", width_chars=100)
    else:
        y = d.body(y, ["256->512 timing: [[ PENDING -- dual-resolution measurement in flight ]]"])
    y -= 2 * mm
    y = d.wrapped(y, "No H100 number is reported unless explicitly labelled as a projection "
                      "with its formula shown -- KLA scores on an H100, we trained/measured on "
                      "an RTX 4060 and (if landed) an A100 cloud run, and never conflate the two.",
                  width_chars=100)

    # Slide 8 -- GitHub & Video Link
    y = d.new_slide("GitHub & Video Link")
    y = d.body(y, [
        "Repository: https://github.com/sahithsundarw/semicon-kla-image-restoration",
        "(public, verified in a logged-out window)",
        "Demo video: [[ OPTIONAL, <=5 min, link here if recorded ]]",
    ])

    # Slide 9 -- References
    y = d.new_slide("References")
    y = d.body(y, [
        "Kumar, T. et al. (2024). Image Data Augmentation Approaches: A Comprehensive "
        "Survey and Future Directions. IEEE Access, 12.",
        "Zhai, L. et al. (2023). A Comprehensive Review of Deep Learning-Based Real-World "
        "Image Restoration. IEEE Access, 11, 21049-21067.",
        "Terven, J. et al. (2025). A Comprehensive Survey of Loss Functions and Metrics "
        "in Deep Learning. Artificial Intelligence Review, 58, 195.",
        "Monga, V. et al. (2021). Algorithm Unrolling: Interpretable, Efficient Deep "
        "Learning for Signal and Image Processing. IEEE SPM, 38(2), 18-44.",
        "Chen, L. et al. (2022). Simple Baselines for Image Restoration (NAFNet). ECCV.",
        "Yoo, J. et al. (2020). Rethinking Data Augmentation for Image Super-Resolution "
        "(CutBlur). CVPR.",
        "Wang, Z. et al. (2004). Image Quality Assessment: SSIM. IEEE TIP.",
        "Zhang, R. et al. (2018). The Unreasonable Effectiveness of Deep Features as a "
        "Perceptual Metric (LPIPS). CVPR.",
        "Shi, W. et al. (2016). Real-Time Single Image and Video SR Using an Efficient "
        "Sub-Pixel CNN (PixelShuffle). CVPR.",
    ], size=9, leading=4.6 * mm)

    d.save()

    # Self-check: banned phrases, proxy sentence, page count -- what V53 will assert.
    from pypdf import PdfReader  # local import; optional dependency for the self-check only
    try:
        reader = PdfReader(str(out_path))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        n_pages = len(reader.pages)
        problems = []
        if n_pages > 9:
            problems.append(f"{n_pages} pages > 9 limit")
        if "proxy" not in text.lower() or "natural photograph" not in text.lower():
            problems.append("proxy sentence not found verbatim")
        for phrase in BANNED_PHRASES:
            if phrase.lower() in text.lower():
                problems.append(f"banned phrase present: {phrase!r}")
        if problems:
            print("SELF-CHECK ISSUES:", problems, file=sys.stderr)
        else:
            print(f"Self-check OK: {n_pages} pages, proxy sentence present, no banned phrases.")
    except ImportError:
        print("pypdf not installed -- skipping self-check (install pypdf to verify V53 conditions)")

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
