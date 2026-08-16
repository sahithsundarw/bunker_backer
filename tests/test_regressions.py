from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dataset_paths import resolve_test_input_dir
from scripts.benchmark_runtime import _render
from src.blocks import NAFBlock
from src.dataset import DataConfig, PairedRestorationDataset
from src.io_utils import save_array
from src.metrics import paired_verdict
from src.utils import (EMA, capture_rng_state, capture_source_provenance, restore_rng_state,
                       save_checkpoint, seed_everything)
from train import (build_argparser, fixed_overfit_regions, initialize_from_checkpoint,
                   promote_checkpoint_if_accepted)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ArtifactTests(unittest.TestCase):
    def test_runtime_report_template_records_target_environment(self) -> None:
        run = {
            "external_s": 2.0, "internal_s": 1.5, "startup_import_s": 0.5,
            "images": 400, "device": "cuda", "precision": "bf16", "batch": 32,
        }
        checkpoint = {
            "path": Path("weights/best.pt"), "sha256": "a" * 64, "bytes": 1,
            "parameters": 2, "hardware": "NVIDIA H100", "python": "3.12.0",
            "numpy": "2.4.0", "torch": "2.11.0+cu128", "cuda_runtime": "12.8",
            "cudnn": "91002",
        }
        report = _render([run], ["python", "benchmark.py"], checkpoint)
        for expected in ("NVIDIA H100", "Python", "NumPy", "PyTorch", "CUDA runtime",
                         "cuDNN", "400 images in 2.00 s"):
            self.assertIn(expected, report)

    def test_release_manifest_and_checkout_checksum(self) -> None:
        manifest_dir = ROOT / "results" / "restored_test_outputs"
        manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
        csv_bytes = (manifest_dir / "manifest.csv").read_bytes()
        self.assertEqual(csv_bytes.count(b"\r"), 0)
        self.assertEqual(hashlib.sha256(csv_bytes).hexdigest(), manifest["manifest_csv_sha256"])
        self.assertEqual(manifest["n_files"], 400)
        self.assertTrue(manifest["release_url"].startswith("https://github.com/"))
        self.assertIn("--require_weights", manifest["command"])
        self.assertEqual(len((manifest_dir / "sha256sums.txt").read_text().splitlines()), 400)

    def test_checkpoint_config_and_source_provenance(self) -> None:
        checkpoint = torch.load(ROOT / "weights" / "best.pt", map_location="cpu",
                                weights_only=True)
        config = yaml.safe_load((ROOT / "configs" / "final.yaml").read_text())
        self.assertEqual(checkpoint["config"], config)
        provenance = checkpoint["provenance"]
        commit = provenance["canonical_training_source_commit"]
        self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertTrue(provenance["original_worktree_marker_preserved"])
        for entry in provenance["canonical_training_source_files"]:
            content = subprocess.check_output(
                ["git", "show", f"{commit}:{entry['path']}"], cwd=ROOT
            )
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])

    def test_selected_model_is_not_the_regressed_checkpoint(self) -> None:
        final = json.loads((ROOT / "results/baselines/final/metrics.json").read_text())
        unet = json.loads((ROOT / "results/baselines/unet_baseline/metrics.json").read_text())
        fm, um = final["metrics"], unet["metrics"]
        wins = sum((fm["psnr"]["mean"] > um["psnr"]["mean"],
                    fm["ssim"]["mean"] > um["ssim"]["mean"],
                    fm["lpips"]["mean"] < um["lpips"]["mean"]))
        self.assertGreaterEqual(wins, 2)
        self.assertGreater(fm["psnr"]["mean"], 28.7)
        self.assertEqual(final["checkpoint_sha256"], digest(ROOT / "weights/best.pt"))

    def test_readme_uses_reproducible_verifier_and_no_final_test_metric_claim(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("results/verification_report.json", readme)
        self.assertIn("python scripts/verify_all.py --strict --fresh-clone", readme)
        manifest = json.loads(
            (ROOT / "results/restored_test_outputs/manifest.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(manifest["metrics"])
        self.assertIn("no final-test psnr", manifest["metrics_note"].lower())


class ResumeAndGateTests(unittest.TestCase):
    @staticmethod
    def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, ema: EMA) -> None:
        scale = random.random() + float(np.random.random()) + float(torch.rand(()))
        x = torch.tensor([[scale]], dtype=torch.float32)
        loss = model(x).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(model)

    def test_complete_resume_state_reproduces_ema_and_model(self) -> None:
        seed_everything(123)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ema = EMA(model, decay=0.999)
        for _ in range(4):
            self._step(model, optimizer, ema)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "resume.pt"
            training_state = {
                "optimizer": optimizer.state_dict(),
                "global_step": 4,
                "scheduler": {"kind": "test"},
                "best": {"psnr": 1.0, "ssim": 0.5, "iter": 4},
                "ema_num_updates": ema.num_updates,
                "epoch": 2,
                "batch_in_epoch": 3,
                "rng": capture_rng_state(),
            }
            save_checkpoint(checkpoint_path, model=model, ema=ema, config={"model": {}},
                            iteration=4, metrics={}, training_state=training_state)
            for _ in range(5):
                self._step(model, optimizer, ema)
            expected_model = {k: v.clone() for k, v in model.state_dict().items()}
            expected_ema = ema.state_dict()

            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            resumed = torch.nn.Linear(1, 1)
            resumed.load_state_dict(payload["model"], strict=True)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
            resumed_optimizer.load_state_dict(payload["training_state"]["optimizer"])
            resumed_ema = EMA(resumed, decay=0.999)
            resumed_ema.load_state_dict(payload["ema"])
            resumed_ema.num_updates = payload["training_state"]["ema_num_updates"]
            seed_everything(999)
            restore_rng_state(payload["training_state"]["rng"])
            for _ in range(5):
                self._step(resumed, resumed_optimizer, resumed_ema)

        self.assertEqual(resumed_ema.num_updates, 9)
        for key, value in resumed.state_dict().items():
            torch.testing.assert_close(value, expected_model[key], rtol=0, atol=0)
        for key, value in resumed_ema.state_dict().items():
            torch.testing.assert_close(value, expected_ema[key], rtol=0, atol=0)

    def test_inference_checkpoint_can_initialize_but_not_fake_resume(self) -> None:
        source_model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            source_model.weight.fill_(0.25)
            source_model.bias.fill_(-0.5)
        source_ema = EMA(source_model, decay=0.999)
        config = {"model": {"name": "test"}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference-only.pt"
            save_checkpoint(path, model=source_model, ema=source_ema, config=config,
                            iteration=20, metrics={})
            target = torch.nn.Linear(2, 1)
            target_ema = EMA(target, decay=0.999)
            metadata = initialize_from_checkpoint(target, target_ema, path, config)

        self.assertEqual(metadata["source_iter"], 20)
        self.assertEqual(metadata["source_weights"], "ema")
        self.assertFalse(metadata["optimizer_state_reused"])
        self.assertEqual(metadata["new_run_starts_at_iter"], 0)
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, source_ema.state_dict()[key], rtol=0, atol=0)

    def test_init_checkpoint_and_resume_are_mutually_exclusive(self) -> None:
        parser = build_argparser()
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--config", "configs/final.yaml",
                "--resume", "resume.pt",
                "--init_checkpoint", "weights/best.pt",
            ])

    def test_rejected_quality_gate_preserves_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.write_bytes(b"known-good")
            before = digest(path)
            called = False

            def writer(_: Path) -> None:
                nonlocal called
                called = True
                path.write_bytes(b"bad-candidate")

            self.assertFalse(promote_checkpoint_if_accepted(False, path, writer))
            self.assertFalse(called)
            self.assertEqual(digest(path), before)

    def test_overfit_regions_are_centered_and_exactly_aligned(self) -> None:
        lr = torch.arange(2 * 1 * 40 * 42, dtype=torch.float32).reshape(2, 1, 40, 42)
        gt = torch.repeat_interleave(torch.repeat_interleave(lr, 2, -2), 2, -1)
        lr_crop, gt_crop, detail = fixed_overfit_regions(lr, gt, lr_side=32)
        self.assertEqual(tuple(lr_crop.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(gt_crop.shape), (2, 1, 64, 64))
        self.assertEqual(detail["lr_shape"], [32, 32])
        torch.testing.assert_close(
            gt_crop,
            torch.repeat_interleave(torch.repeat_interleave(lr_crop, 2, -2), 2, -1),
            rtol=0,
            atol=0,
        )

    def test_dirty_or_external_training_source_content_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "run.yaml"
            source.write_text("optim:\n  lr: 0.001\n", encoding="utf-8")
            provenance = capture_source_provenance([source])
        record = provenance["source_files"][0]
        self.assertFalse(record["tracked"])
        self.assertEqual(record["dirty_content_utf8"], "optim:\n  lr: 0.001\n")
        self.assertRegex(provenance["git_commit"], r"^[0-9a-f]{40}$")


class InferenceAndIoTests(unittest.TestCase):
    def test_mixed_extensions_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, out = root / "in", root / "out"
            inp.mkdir()
            np.save(inp / "valid.npy", np.zeros((4, 4), dtype=np.float32))
            (inp / "advertised.png").write_bytes(b"not-an-image")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "inference.py"), "--input_dir", str(inp),
                 "--output_dir", str(out)], cwd=ROOT, text=True, capture_output=True
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("unsupported raster image", proc.stderr)
            self.assertFalse(out.exists())

    def test_success_reconciles_stale_contract_outputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, out = root / "in", root / "out"
            inp.mkdir()
            out.mkdir()
            source = next((ROOT / "sample_inputs").glob("*.npy"))
            (inp / source.name).write_bytes(source.read_bytes())
            np.save(out / "stale.npy", np.zeros((2, 2), dtype=np.float32))
            (out / "notes.txt").write_text("preserve me", encoding="utf-8")
            subprocess.run(
                [sys.executable, str(ROOT / "inference.py"), "--input_dir", str(inp),
                 "--output_dir", str(out), "--require_weights"], cwd=ROOT, check=True
            )
            self.assertEqual({p.name for p in out.glob("*.npy")}, {source.name})
            self.assertEqual((out / "notes.txt").read_text(), "preserve me")

    def test_failed_promotion_rolls_back_complete_previous_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp, out = root / "in", root / "out"
            inp.mkdir()
            out.mkdir()
            sources = sorted((ROOT / "sample_inputs").glob("*.npy"))[:2]
            for source in sources:
                (inp / source.name).write_bytes(source.read_bytes())
            (out / sources[1].name).mkdir()
            prior = out / "prior.npy"
            prior.write_bytes(sources[0].read_bytes())
            before = digest(prior)
            proc = subprocess.run(
                [sys.executable, str(ROOT / "inference.py"), "--input_dir", str(inp),
                 "--output_dir", str(out), "--require_weights"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual([p.name for p in out.glob("*.npy") if p.is_file()], [prior.name])
            self.assertEqual(digest(prior), before)

    def test_atomic_array_write_does_not_replace_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.npy"
            path.write_bytes(b"known-good")
            with mock.patch("numpy.save", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    save_array(path, np.zeros((4, 4), dtype=np.float32))
            self.assertEqual(path.read_bytes(), b"known-good")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_scored_naf_forward_has_no_item_sync(self) -> None:
        self.assertNotIn(".item(", inspect.getsource(NAFBlock.forward))

    def test_cuda_baseline_timing_is_synchronized_and_labeled(self) -> None:
        source = (ROOT / "scripts/make_baselines.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("torch.cuda.synchronize()"), 2)
        self.assertIn("timing_method", source)


class DataMetricsAndToolsTests(unittest.TestCase):
    def test_second_invalid_pair_fails_construction(self) -> None:
        valid_lr = np.zeros((8, 8), dtype=np.float32)
        valid_gt = np.zeros((16, 16), dtype=np.float32)
        invalid_gt = np.zeros((15, 16), dtype=np.float32)
        cfg = DataConfig(lr_patch=4, preload=True)
        with self.assertRaisesRegex(ValueError, "mem_000001"):
            PairedRestorationDataset.from_arrays(
                [valid_lr, valid_lr], [valid_gt, invalid_gt], cfg=cfg, split="train"
            )

    def test_constant_paired_difference_is_significant(self) -> None:
        ref = {f"{i}.npy": {"psnr": 1.0} for i in range(30)}
        better = {f"{i}.npy": {"psnr": 1.25} for i in range(30)}
        same = {f"{i}.npy": {"psnr": 1.0} for i in range(30)}
        verdict = paired_verdict(better, ref, "psnr", higher_is_better=True)
        self.assertTrue(verdict["significant"])
        self.assertTrue(verdict["win"])
        self.assertEqual(verdict["t"], float("inf"))
        self.assertFalse(paired_verdict(same, ref, "psnr", True)["significant"])

    def test_requested_missing_baseline_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            (root / "train" / "GT").mkdir(parents=True)
            (root / "train" / "NoisyLR").mkdir(parents=True)
            np.save(root / "train" / "NoisyLR" / "000000.npy",
                    np.zeros((4, 4), dtype=np.float32))
            np.save(root / "train" / "GT" / "000000.npy",
                    np.zeros((8, 8), dtype=np.float32))
            split = Path(tmp) / "split.txt"
            split.write_text("000000.npy\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/make_baselines.py"),
                 "--data_root", str(root), "--split", str(split),
                 "--baselines", "unet_baseline", "--unet_ckpt", str(Path(tmp) / "missing.pt"),
                 "--out", str(Path(tmp) / "preds"), "--device", "cpu"],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("0/1 baselines produced", proc.stdout)
            self.assertIn(str(split.resolve()), proc.stdout)

    def test_current_and_historical_test_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "NoisyLR"
            current.mkdir()
            self.assertEqual(resolve_test_input_dir(root), current.resolve())
            current.rmdir()
            historical = root / "test_NoisyLR"
            historical.mkdir()
            self.assertEqual(resolve_test_input_dir(root), historical.resolve())


if __name__ == "__main__":
    unittest.main()
