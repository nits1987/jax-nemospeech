# FastConformer RNN-T on JAX

`fcrnnt-jax` is an engineering qualification PoC for running the trainable core of a Parakeet-style FastConformer + RNN-T model on Cloud TPU. It contains a pure-JAX classic RNN-T loss, a configuration-driven Flax model, an Optax training step, Orbax checkpointing, strict weight-conversion primitives, and repeatable CPU/TPU smoke commands.

This repository deliberately separates **runtime proof** from **model-compatibility proof**. The tiny synthetic path can establish that model, loss, gradients, optimizer, and checkpoints execute on XLA. It does not by itself prove that NVIDIA's public Parakeet checkpoint was converted exactly or that WER is unchanged. Those claims require pinned NeMo fixtures, complete tensor mapping, decoder parity, and matched WER evaluation.

## Start on the provisioned v6e

The manual target is one Compute Engine TPU v6e chip:

- machine type: `ct6e-standard-1t`;
- OS image project: `ubuntu-os-accelerator-images`;
- OS image family: `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e`.

The family name is shared across supported TPU generations; `ct6e-standard-1t` is what selects one v6e chip. This is a direct Compute Engine workflow, not a GKE deployment and not the legacy Cloud TPU API.

If the instance is already running, clone a pinned revision directly on it:

```bash
git clone YOUR_REPOSITORY_URL "${HOME}/fcrnnt-jax"
cd "${HOME}/fcrnnt-jax"
git checkout YOUR_COMMIT_TAG_OR_BRANCH
git rev-parse HEAD
```

If Git is unavailable, copy the complete `fcrnnt-jax` directory to `${HOME}/fcrnnt-jax`. Then follow [deploy.md](deploy.md): run the workstation preflight in Section 1, skip the source-transfer section when the code is already present, and execute the VM commands from Section 3 onward. The runbook pins the current TPU qualification environment, verifies the managed image and TPU backend, runs each correctness gate, preserves evidence, and provides guarded cleanup.

## Local CPU quick start

Python 3.12 is recommended so local validation uses the same Python generation as the pinned TPU lane:

On a CPU workstation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m fcrnnt_jax.cli devices
python -m fcrnnt_jax.cli loss-smoke
python -m fcrnnt_jax.cli model-smoke
python -m fcrnnt_jax.cli train-smoke --steps 100 --workdir ./artifacts/train-smoke
python -m fcrnnt_jax.cli checkpoint-smoke --directory ./artifacts/checkpoint-smoke
```

On a CPU workstation, `devices` is only a local installation smoke; the TPU pass requires backend `tpu`, exactly one device, and the BF16 device computation documented in [deploy.md](deploy.md).

## Real-audio frontend smoke

NVIDIA's Parakeet RNNT 1.1B model card uses LibriSpeech utterance
`2086-149220-0033` as its example. Download the same small WAV and verify its
identity before running it through the JAX frontend:

```bash
curl -fL --retry 3 \
  -o fixtures/audio/2086-149220-0033.wav \
  https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav

python -m fcrnnt_jax.cli audio-smoke \
  --audio fixtures/audio/2086-149220-0033.wav \
  --reference-transcript-file fixtures/audio/2086-149220-0033.txt \
  --expected-audio-sha256 5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150 \
  --output artifacts/parakeet-audio-frontend.npz
```

A pass proves checksum-verified PCM16 mono/16-kHz ingestion and finite 80-bin
log-mel extraction on the selected JAX backend. The expected feature shape is
`[1, 742, 80]`. The transcript is ground truth, not model output: this command
does not claim transcription or WER parity. Those require converted pretrained
weights, the matching tokenizer, and an RNN-T decoder.

## What is implemented

- classic RNN-T forward/backward dynamic programming with explicit lengths and blank semantics;
- compact FastConformer-style encoder, LSTM predictor, and joint network with tiny and 1.1B-shaped presets;
- a complete JIT-able training step and finite-gradient checks;
- Orbax save/restore of the full training PyTree;
- fail-closed tensor conversion rules and conversion reports;
- strict real-WAV loading and a checksum-aware JAX audio-frontend smoke;
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
python -m fcrnnt_jax.cli audio-smoke --audio FILE.wav --reference-transcript-file FILE.txt [--output evidence.npz]
python -m fcrnnt_jax.cli train-smoke [--steps N] [--workdir PATH]
python -m fcrnnt_jax.cli benchmark [--preset tiny|parakeet-1.1b] [--steps N] [--warmup N]
python -m fcrnnt_jax.cli checkpoint-smoke [--directory PATH]
```

Every command emits a final machine-readable JSON object. Retain that output together with `pip freeze`, device metadata, fixture hashes, and the code revision.

## Repository map

- `src/fcrnnt_jax/rnnt_loss.py`: materialized and streamed classic RNN-T losses;
- `src/fcrnnt_jax/model.py`: FastConformer-style encoder, LSTM predictor, and joint network;
- `src/fcrnnt_jax/training.py`: train state, optimizer, synthetic batches, and JIT training step;
- `src/fcrnnt_jax/checkpoint.py`: Orbax save, restore, and resume support;
- `src/fcrnnt_jax/conversion.py`: fail-closed checkpoint mapping primitives;
- `src/fcrnnt_jax/audio.py`: audio/frontend utilities used by the PoC;
- `tests/`: CPU correctness and integration tests;
- `fixtures/`: format and instructions for external NeMo parity fixtures;
- `configs/`: 1.1B shape target, parity tolerances, and source-lock template;
- `deploy.md`: copy-and-run Compute Engine v6e qualification runbook.

## A successful run proves

- the selected JAX/libtpu environment sees one v6e device;
- RNN-T loss values and gradients pass the selected reference fixture;
- tiny and 1.1B-shaped model graphs compile and produce finite outputs;
- a pinned real speech WAV decodes and produces finite JAX frontend features;
- the tiny end-to-end training loop updates parameters and reduces loss;
- the representative full training step fits and runs, or records a reproducible OOM boundary;
- full training state can be checkpointed, restored, and resumed equivalently.

It still does not prove pretrained Parakeet checkpoint parity or unchanged WER. Those remain explicit gates for the subsequent compatibility phase.

## Design boundary

This is an ASR runtime PoC, not a JAX rewrite of the NeMo toolkit. A second FastConformer RNN-T model should require a new normalized config and explicit tensor mapping, while reusing the loss, encoder primitives, trainer, and checkpoint machinery.
