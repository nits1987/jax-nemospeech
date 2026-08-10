# FastConformer RNN-T on JAX

`fcrnnt-jax` is an engineering qualification PoC for running the trainable core of a Parakeet-style FastConformer + RNN-T model on Cloud TPU. It contains a pure-JAX classic RNN-T loss, a configuration-driven Flax model, an Optax training step, Orbax checkpointing, strict weight-conversion primitives, and repeatable CPU/TPU smoke commands.

This repository deliberately separates **runtime proof** from **model-compatibility proof**. The tiny synthetic path can establish that model, loss, gradients, optimizer, and checkpoints execute on XLA. It does not by itself prove that NVIDIA's public Parakeet checkpoint was converted exactly or that WER is unchanged. Those claims require pinned NeMo fixtures, complete tensor mapping, decoder parity, and matched WER evaluation.

## Quick start

On a CPU workstation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
python -m fcrnnt_jax.cli devices
python -m fcrnnt_jax.cli loss-smoke
python -m fcrnnt_jax.cli model-smoke
python -m fcrnnt_jax.cli train-smoke --steps 100 --workdir ./artifacts/train-smoke
python -m fcrnnt_jax.cli checkpoint-smoke --directory ./artifacts/checkpoint-smoke
```

For the already provisioned Cloud TPU v6e-1, follow [deploy.md](deploy.md). It includes source transfer, TPU-specific JAX installation, gates in the required order, evidence collection, troubleshooting, and verified cleanup commands.

## What is implemented

- classic RNN-T forward/backward dynamic programming with explicit lengths and blank semantics;
- compact FastConformer-style encoder, LSTM predictor, and joint network with tiny and 1.1B-shaped presets;
- a complete JIT-able training step and finite-gradient checks;
- Orbax save/restore of the full training PyTree;
- fail-closed tensor conversion rules and conversion reports;
- synthetic smokes and an external loss-fixture path for NeMo parity evidence.

## What remains a compatibility gate

- freeze the exact NeMo version, checkpoint, tokenizer, frontend, and decoder policy;
- generate NeMo loss/gradient and layer-output fixtures;
- finish and review the checkpoint-specific 1.1B tensor mapping, including all mutable state;
- match frontend, masking, subsampling, relative-attention, predictor, and joint semantics tensor by tensor;
- compare greedy tokens and baseline WER before training;
- run matched short NeMo/GPU and JAX/TPU training and evaluate both with the same decoder.

The `parakeet-1.1b` preset is therefore a **shape and fitment target**, not a declaration of checkpoint equivalence.

## Commands

```text
python -m fcrnnt_jax.cli devices
python -m fcrnnt_jax.cli loss-smoke [--fixture fixture.npz]
python -m fcrnnt_jax.cli model-smoke [--preset tiny|parakeet-1.1b]
python -m fcrnnt_jax.cli train-smoke [--steps N] [--workdir PATH]
python -m fcrnnt_jax.cli benchmark [--preset tiny|parakeet-1.1b] [--steps N] [--warmup N]
python -m fcrnnt_jax.cli checkpoint-smoke [--directory PATH]
```

Every command emits a final machine-readable JSON object. Retain that output together with `pip freeze`, device metadata, fixture hashes, and the code revision.

## Design boundary

This is an ASR runtime PoC, not a JAX rewrite of the NeMo toolkit. A second FastConformer RNN-T model should require a new normalized config and explicit tensor mapping, while reusing the loss, encoder primitives, trainer, and checkpoint machinery.
