# Environment

Verified 2026-08-15.

## Python

| Interpreter | Version | Path |
|---|---|---|
| `python` (default on PATH) | **3.14.3** | `C:\Python314\python.exe` |
| `py -3.14` | 3.14.x | `C:\Users\sahit\AppData\Local\Python\pythoncore-3.14-64\python.exe` |
| `py -3.12` | **3.12.10** | `C:\Users\sahit\AppData\Local\Programs\Python\Python312\python.exe` |
| `py -3.11` | 3.11.x | `C:\Users\sahit\AppData\Local\Programs\Python\Python311\python.exe` |

## PyTorch availability — no blocker

The default interpreter is 3.14, which is ≥3.13, so wheel availability was checked directly
against PyPI rather than assumed. Latest `torch` is **2.13.0**, and its Windows wheels are:

```
torch-2.13.0-cp310-cp310-win_amd64.whl
torch-2.13.0-cp311-cp311-win_amd64.whl
torch-2.13.0-cp312-cp312-win_amd64.whl
torch-2.13.0-cp313-cp313-win_amd64.whl
torch-2.13.0-cp314-cp314-win_amd64.whl
torch-2.13.0-cp314-cp314t-win_amd64.whl
```

**A `cp314` Windows wheel exists.** Python 3.14 is supported. Installing Python 3.11/3.12 is
not required, and moving GPU work to Colab/Kaggle is not required.

Recommendation anyway: **build on 3.12**. It has the widest wheel coverage for the rest of a
vision stack (timm, kornia, basicsr, opencv variants), which lags core torch by months. 3.12
is already installed and already has the inspection dependencies.

## GPU — present

```
NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 610.47
```

8 GB VRAM, local CUDA training is viable. At `(128,128)` → `(256,256)` the tensors are small;
8 GB is comfortable for this task.

## Installed (into `py -3.12`)

```
numpy      2.5.2
opencv-python (cv2) 5.0.0
tifffile   2026.7.31
```

PyTorch is **not** installed yet — deliberately, per the setup brief.

Note: `C:\Users\sahit\AppData\Local\Programs\Python\Python312\Scripts` is not on PATH, so
pip-installed console scripts for 3.12 are not directly callable. Invoke modules via
`py -3.12 -m <module>` instead.

## Reproducing

```powershell
py -3.12 -m pip install numpy opencv-python tifffile
py -3.12 scripts\inspect_dataset.py C:\kla-data
py -3.12 scripts\probe_quantization.py C:\kla-data
```
