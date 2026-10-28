# Torch training and native export

Think of Torch as the school and the C++ engine as the runner. Torch learns the
numbers inside the model. We then copy those numbers into the native file format
and check that the runner gives the same answers. The agreed one-block architecture
stays fixed; this phase adds learning, saving and export around it.

## What the pieces do

- `tools/torch_forecaster.py` expresses the same model using Torch tensor operations.
  Torch supplies matrix multiplication and automatic differentiation. Native C++
  remains an inference library; it has no Torch dependency.
- `tools/training_state.py` groups examples into batches, accumulates gradients,
  clips their norm and asks Torch AdamW to update parameters. It stores model,
  optimizer moments/steps and the dedicated shuffle RNG in an epoch checkpoint.
- `tools/train_ettm1.py` checks the original data and training-only scaler, runs
  the frozen recipe, chooses the earliest best validation epoch, exports its
  weights and checks both native kernels. A separate evaluation command requires
  all preregistered seeds to be complete before scoring the final period.

The forward reference uses FP64 reductions for normalization/attention/GELU, with
FP32 tensors between layers, to follow the native arithmetic contract. Torch owns
backpropagation through those operations. This is a correctness reference, not an
optimized Torch latency baseline. No dropout, new depth or architecture change is
introduced. A performance comparison must also include an ordinary optimized Torch
implementation and appropriate established runtimes under matched conditions.

## Fixed first experiment

The recipe in `benchmarks/ettm1/training_v1.json` is committed before training:
three seeds (7, 19, 43), exactly six epochs each, batch size 64 and two accumulated
batches per update. The final smaller group is divided by its actual example
count. AdamW uses learning rate 0.001, weight decay 0.01, betas 0.9/0.999 and epsilon
1e-8. Global gradient norm is clipped at 1.0. All trainable parameters receive
weight decay. Matrices start with Xavier uniform values, biases/shifts with zero
and normalization scales with one.

Training minimizes squared error in standardized target units. Selection uses
validation MAE in original temperature units, with the earliest epoch winning a
tie. Test reporting includes MAE, RMSE and every seed. The two baselines predict
that temperature stays at its latest observation, or repeats the temperature at
the target time one day earlier. This is an end-to-end learning experiment with a
bounded compute budget, not a tuned model or a reproduction of published ETT
multi-horizon scores. Overlapping forecasts are dependent; no IID significance or
trading conclusion follows from these scores.

## Setup and commands

Training is an explicit optional installation. The local Intel Mac uses Python
3.10, NumPy 1.26.4 and Torch 2.2.2, the last official Intel Mac Torch release.
See [PyTorch's platform notice](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).
The Linux CPU checks also exercise Torch 2.13.0 from the
[official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/).
Check the CI result before calling that configuration verified.

```sh
python3.10 -m venv local/torch-reference-env
local/torch-reference-env/bin/python -m pip install -r requirements-training-macos-intel.txt
local/torch-reference-env/bin/python -m pip install --no-deps path/to/the/built/stealth_cpu.whl
local/torch-reference-env/bin/python -m unittest discover -s tests/training -v
```

On Linux, install `requirements-training-common.txt`, then explicitly install
`torch==2.13.0+cpu` with `--index-url https://download.pytorch.org/whl/cpu`.
Build the native wheel as described in [the package guide](python-stream-v1.md).
Use the matching interpreter for all steps. If this device's Python needs an
explicit certificate bundle, pass an existing trusted bundle with `--cert`; never
disable TLS verification.

Commit the code and protocol, then run each seed into a new directory:

```sh
local/torch-reference-env/bin/python tools/train_ettm1.py train \
  --raw local/datasets/ettm1-v1 --prepared local/datasets/ettm1-prepared-v1 \
  --seed 7 --output local/ettm1-training-v1/seed7
```

Repeat with seeds 19 and 43 and their own directories. Once all complete:

```sh
local/torch-reference-env/bin/python tools/train_ettm1.py evaluate \
  --raw local/datasets/ettm1-v1 --prepared local/datasets/ettm1-prepared-v1 \
  --runs local/ettm1-training-v1/seed7 local/ettm1-training-v1/seed19 local/ettm1-training-v1/seed43 \
  --output local/ettm1-training-v1/evaluation
```

Each `model.stlf` is directly loadable with `stealth_cpu.ForecastModel.load` or the
C++ loader. Its `export.json` binds it to feature order, cadence, target, scaler,
data/protocol/task hashes, selected checkpoint, environment and code commit.
Raw data, derivatives, checkpoints and trained binaries remain local; public
results contain compact aggregate evidence. Dataset/model redistribution rights
are not established by this workflow.

## Recovery and evidence limits

Checkpoints contain ordinary numeric NPZ arrays and JSON, never pickle. Reads
bound bytes/archive entries, verify hashes, reject malformed tensors and require
the same experiment/environment. They return fresh objects. Manifest files are
written last; partial directories are incomplete, and existing destinations are
refused. Epoch checkpoints do not include an interrupted partial optimizer step.
Resume a complete epoch with `--resume local/previous-run/epoch-003`, the same
seed and a new output directory. The prior completed epoch history is checked and
copied; the original run remains intact. Use the recorded code commit/environment
for exact continuation. Cross-device or cross-version bitwise training equality
is not promised.

The stochastic operations use dedicated CPU Torch generators: one for
initialization and one for epoch shuffling. There is no data-loader worker, NumPy
random stream or dropout state to restore in this version. Changing those requires
extending the checkpoint contract. Gradient accumulation is checked against an
ordinary larger SGD batch without loosening tolerance. AdamW updates can differ
slightly across batch partitions: roundoff in the analytically zero attention
key-bias gradient is amplified by division by its tiny second moment. Resume tests
keep the same partition and demand exact equality of weights and optimizer state.

Component tests cover independent equations, native kernels, causal prefixes,
batch agreement and finite-difference gradients for every parameter group. The
miniature learning test is a software diagnostic. It does not substitute for the
public-data experiment. Native parity on trained ETTm1 exports samples 32 fixed,
evenly spaced windows per split, checking all 96 origin outputs in both kernels
at the predeclared absolute 5e-5 plus relative 2e-4 tolerance; it does not exhaust
all possible model inputs.
