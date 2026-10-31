# Environments

The two experiments keep independent environments. The CPU data/metric utilities and
unit tests use only the Python standard library. Python 3.11 is recommended for training.

The supplied pins are **candidate environments, not recovered historical lockfiles**.
The historical TimeLens files listed PyTorch 2.6, CUDA 12.4, TRL 0.19.1 and
Transformers 4.57.1; that torch/CUDA combination does not establish Blackwell support.
The native SFT backup did not contain a complete environment export.

For an NVIDIA Blackwell Linux host, begin with a CUDA 12.8-capable PyTorch build:

```bash
python -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/sft.txt
python -m pip check
python tools/doctor.py --cuda
```

Create another environment for `requirements/timelens.txt`, or `requirements/qlora.txt`
for 4-bit training. Confirm the actual server's NVIDIA driver supports the selected
CUDA runtime. The configs default to SDPA, so FlashAttention and Liger are optional.
Change `--attention flash_attention_2` only in an environment with a compatible build.

After the first successful GPU smoke run, save `pip freeze`, CUDA/driver versions,
model revision and the run configuration next to that run. Do not claim these pins
reproduce the historical environment before that verification.
