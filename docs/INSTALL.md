# Install guide

## Why the install was taking 1.5+ hours

Almost always it's one of these, not a real "install duration" — it's a slow
or stuck download:

1. **Plain `pip install torch` with no `--index-url`.** PyPI's default torch
   wheel bundles CUDA libraries and is 2-3GB. On a slow/unstable connection
   this can crawl or stall entirely, and pip's retry/resume behavior is poor.
2. **conda's default solver** (not mamba) can spend a very long time just
   *resolving* the environment before it even starts downloading anything —
   this is a well-known conda pain point, not a download speed issue.
3. Installing a **CUDA build when you don't even have an NVIDIA GPU** — you
   download 2-3GB you can't use. The CPU-only wheel is ~200MB.

## Fastest path: conda + mamba

```bash
# one-time: install mamba into your base env (much faster solver than conda)
conda install -n base -c conda-forge mamba -y

# check if you have an NVIDIA GPU and what CUDA version your driver supports
nvidia-smi   # look at "CUDA Version" in the top-right box; if this command
             # doesn't exist, you don't have an NVIDIA GPU — use the CPU file

# GPU:
mamba env create -f environment-gpu.yml
# CPU-only:
mamba env create -f environment-cpu.yml

conda activate mer
```

If `nvidia-smi` shows a CUDA version lower than 12.4, edit
`environment-gpu.yml` and change `pytorch-cuda=12.4` to `12.1` or `11.8`
before running the command above.

## Plain pip / venv path

```bash
python3.12 -m venv venv
source venv/bin/activate

# GPU (adjust cu124 -> cu121 / cu118 to match nvidia-smi's CUDA version):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-only (fast, ~200MB):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# then everything else:
pip install -r requirements.txt
```

The `--index-url` is the important part — it points pip directly at the
prebuilt wheel for your platform instead of letting it resolve through
PyPI's default index, which is usually what causes the multi-hour hangs.

## If the download itself is just slow (bad connection, not resolver)

Download the wheel once with a resumable downloader, then install from the
local file:

```bash
# find the exact wheel URL for your Python/OS/CUDA combo at:
# https://download.pytorch.org/whl/torch/
aria2c -x 8 -s 8 <wheel_url>
pip install ./torch-<version>.whl
```

`aria2c` (or `wget -c` for resume support) handles flaky connections far
better than pip's built-in downloader.

## Sanity check after install

```bash
Channels:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Should print `2.6.0` or newer. If `torch.cuda.is_available()` is `False` and
you expected a GPU, you installed the CPU build by mistake — reinstall with
the CUDA `--index-url` above.
