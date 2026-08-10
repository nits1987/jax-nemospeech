# Parakeet FastConformer + RNN-T on JAX/v6e: Fitment PoC Execution Plan

**Status:** execution plan v0.2  
**Purpose:** decide whether to fund the full ASR-on-JAX training pipeline  
**First model:** pinned `nvidia/parakeet-rnnt-1.1b`  
**Hardware:** one Cloud TPU `v6e-1` for qualification; CPU/GPU for reference generation  
**Reference oracle:** pinned NVIDIA NeMo model, recipe, container, tokenizer, and decoder  
**Target runtime:** JAX + Flax NNX + Optax + Orbax

## 1. Decision this PoC must support

The business hypothesis is:

> A reusable JAX ASR training runtime can load a supported FastConformer + RNN-T checkpoint, reproduce NeMo behavior, continue training on TPU without material WER regression, and be reused by another model with the same architecture family.

This PoC is successful only if it proves all of the following:

1. The converted model reproduces NeMo forward outputs and pre-training WER.
2. A classic RNN-T loss produces matching values and logit gradients.
3. A realistic full training step fits and runs on v6e; encoder-only timing is not sufficient.
4. A tiny fixed set overfits and a complete checkpoint restores correctly.
5. A matched short NeMo/GPU and JAX/TPU run stays inside a pre-agreed WER margin.
6. A second FastConformer + RNN-T checkpoint works through configuration and mapping changes, without rewriting shared model or training mathematics.

This is an **ASR compatibility PoC**, not a JAX rewrite of the full NeMo toolkit.

## 2. Decisions already made

- Parakeet uses **classic RNN-T**, not HAT. Blank token semantics, reduction, optional FastEmit/clamp settings, and terminal-blank behavior must come from the pinned NeMo recipe.
- Use the [Mddct JAX RNN-T repository](https://github.com/Mddct/jax-rnnt-loss) only as an algorithmic reference. It has no repository license or test/release package.
- Use the Apache-2.0-licensed [AXLearn transducer implementation](https://github.com/apple/axlearn/blob/main/axlearn/common/transducer.py) and its [tests](https://github.com/apple/axlearn/blob/main/axlearn/common/transducer_test.py) as the provenance base for a small standalone loss module.
- Run the model in bf16 where parity permits it, but perform log-softmax and the RNN-T forward/backward recurrence in fp32.
- NeMo remains the behavioral oracle. Hugging Face safetensors may be the import/export format, but do not silently replace NeMo semantics with library defaults.
- `v6e-1` is a fitment device. If the full step cannot fit unsharded, record that result and run a separately approved v6e-4/FSDP spike; do not weaken the memory test.

## 3. Work lanes

| Lane | Runs where | Owner | Output |
|---|---|---|---|
| TPU readiness | Cloud TPU v6e-1 | Operator | Device/version report and matmul proof |
| NeMo oracle | Pinned CPU/GPU container | ASR engineer | Source lock, golden tensors, baseline WER, loss/gradient fixtures |
| JAX loss | CPU first, then v6e-1 | JAX engineer | Licensed classic RNN-T loss, tests, real-shape HBM/throughput report |
| JAX model/converter | CPU first, then v6e-1 | JAX + ASR engineer | FastConformer/predictor/joint implementation and complete mapping report |
| Integrated trainer | v6e-1 | JAX engineer | Full train step, tiny overfit, Orbax restore, performance evidence |
| WER and reuse | Same reference evaluator plus v6e-1 | ASR engineer | Non-regression report and second-model compatibility report |

The TPU readiness check can happen immediately. Most implementation and parity debugging must happen on CPU/GPU so the TPU is not left billing while code is written.

## 4. Required repository outputs

The implementation work should converge on this minimal shape:

```text
poc/
  source-lock.json
  requirements-reference.lock
  requirements-tpu.lock
  configs/
    parakeet-rnnt-1.1b.yaml
    parakeet-rnnt-0.6b.yaml
    parity-tolerances.yaml
    wer-gates.yaml
  reference/
    dump_nemo_reference.py
    run_nemo_loss_fixture.py
    run_nemo_wer.py
  src/nemo_asr_jax/
    audio.py
    fastconformer.py
    predictor.py
    joint.py
    conversion.py
    rnnt_reference.py
    rnnt_xla.py
    train_step.py
    checkpoint.py
    export.py
  tests/
    test_conversion.py
    test_forward_parity.py
    test_rnnt_loss.py
    test_train_step.py
    test_checkpoint.py
  fixtures/
    parity/
    rnnt/
  reports/
    tpu-readiness.json
    conversion-report.json
    parity-report.json
    rnnt-report.json
    train-step-report.json
    wer-report.json
    performance-report.json
    reuse-report.json
    verdict.md
```

Every report must record the code commit, model/checkpoint hashes, dependency versions, device type, input fixture hashes, precision, shapes, and pass/fail thresholds.

## 5. Phase P0: v6e-1 readiness and cost control

**Goal:** prove the machine is usable by JAX, capture its environment, then release it until TPU code is ready.

### P0.1 Record the resource

Run locally or in Cloud Shell, replacing the placeholders with the machine being created:

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export ZONE="YOUR_US_ZONE"
export TPU_NAME="YOUR_V6E_1_NAME"

gcloud compute tpus tpu-vm describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

Record whether the machine was created directly or by a queued resource. Cleanup differs for those two modes.

### P0.2 Create the JAX environment on the TPU VM

```bash
python3 -m venv "${HOME}/venvs/parakeet-jax-poc"
source "${HOME}/venvs/parakeet-jax-poc/bin/activate"
python -m pip install --upgrade pip
python -m pip install "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python -m pip install flax optax orbax-checkpoint

python - <<'PY' | tee "${HOME}/parakeet-jax-poc-readiness.txt"
import json
import time
import jax
import jax.numpy as jnp

print("devices:", jax.devices())
print("backend:", jax.default_backend())

@jax.jit
def matmul(x):
    return x @ x

x = jnp.ones((4096, 4096), dtype=jnp.bfloat16)
start = time.time()
y = matmul(x)
y.block_until_ready()
print("first_call_seconds:", time.time() - start)
print("finite:", bool(jnp.isfinite(y).all()))

report = {
    "backend": jax.default_backend(),
    "device_count": jax.device_count(),
    "devices": [str(device) for device in jax.devices()],
    "jax_version": jax.__version__,
}
print("TPU_READY=" + json.dumps(report, sort_keys=True))
PY

python -m pip freeze > "${HOME}/parakeet-jax-poc-requirements.txt"
```

The official installation reference is [JAX on a Cloud TPU VM](https://cloud.google.com/tpu/docs/run-calculation-jax).

### P0.3 Release idle TPU capacity

Exit the VM and copy both evidence files locally before releasing it:

```bash
gcloud compute tpus tpu-vm scp \
  "${TPU_NAME}:~/parakeet-jax-poc-readiness.txt" \
  "./${TPU_NAME}-readiness.txt" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"

gcloud compute tpus tpu-vm scp \
  "${TPU_NAME}:~/parakeet-jax-poc-requirements.txt" \
  "./${TPU_NAME}-requirements.txt" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

Delete the TPU if no compiled JAX model/loss test will run immediately. For a directly created TPU VM:

```bash
gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --quiet
```

If it came from a queued resource, delete the TPU first, wait until the request becomes `SUSPENDED`, and then delete the queued-resource request. Do not delete an unknown resource name; first confirm both names with `describe`/`list`.

**P0 gate**

- Exactly one TPU device is visible and the default backend is `tpu`.
- The compiled matmul returns finite output.
- JAX/libtpu/Flax/Optax/Orbax versions and resource metadata are retained.
- The TPU is not left idle while P1-P3 implementation work proceeds.

## 6. Phase P1: freeze the source and generate the NeMo oracle

**Goal:** create immutable reference evidence before implementing JAX behavior.

### Work

1. Pin the exact model repository commit and hash the `.nemo`, safetensors, config, tokenizer, and processor files.
2. Pin the NeMo version, CUDA/PyTorch versions, container digest, decoder configuration, and current GPU training recipe.
3. Pin LibriSpeech manifests and audio hashes for:
   - four parity utterances with short, odd, and long lengths;
   - one padded mixed-length parity batch;
   - twenty `train-clean-100` overfit utterances;
   - a representative throughput shard;
   - the complete `dev-clean` WER set.
4. Run NeMo in deterministic eval/fp32 mode with dropout and LayerDrop disabled.
5. Inventory every parameter and mutable buffer: name, shape, dtype, count, and hash.
6. Dump reference outputs for:
   - features, feature lengths, masks, and subsampling outputs;
   - FastConformer blocks 0, 1, middle, and final, plus final encoder output;
   - predictor embeddings, LSTM states, and output;
   - encoder/predictor projections and selected joint logits;
   - greedy token IDs and normalized transcripts;
   - classic RNN-T loss and logit gradients on small fixed/random lattices.
7. Run and retain the NeMo baseline WER on `dev-clean` using the exact decoder policy that will be used for all comparisons.
8. Approve component tolerances and the WER non-inferiority margin before viewing JAX results.

**P1 gate**

- `source-lock.json`, hashes, resolved config/recipe, fixtures, and baseline WER are complete.
- Two deterministic oracle runs reproduce fixture hashes.
- The weight inventory includes BatchNorm state and every predictor/joint parameter, not only the encoder.

## 7. Phase P2: build and qualify the classic JAX RNN-T loss

**Goal:** adapt an existing licensed JAX dynamic program to Parakeet/NeMo semantics and prove it on CPU before TPU optimization.

### P2.1 Establish licensed provenance

1. Pin an AXLearn commit containing the transducer implementation and tests.
2. Extract only the required mathematical primitives into `rnnt_xla.py` while retaining Apache-2.0 copyright/license/notice requirements.
3. Document every local modification. Do not copy from the unlicensed Mddct fork.

### P2.2 Implement the Parakeet contract

- Add classic RNN-T `log_softmax`; do not use the HAT helper.
- Use blank ID and BOS/padding behavior from the pinned checkpoint; the public candidate currently uses blank ID `1024`, but the source lock wins.
- Accept static padded tensors plus explicit acoustic and label lengths.
- Match NeMo reduction and any FastEmit/clamp/pruning options used by the frozen recipe.
- Cast joint logits/log probabilities and the forward/backward recurrence to fp32.
- Surface invalid lengths, labels, and padding rather than discarding `checkify` errors.
- Keep the custom VJP and retain only blank/target probabilities, not the full `[B,T,U,V]` tensor.
- Make diagnostic matrices optional so the production train step does not return unnecessary `B x T x U` buffers.

### P2.3 Correctness matrix

Test all of the following against the NeMo CPU/reference path and finite differences where practical:

- `T=1`, `U=0`, short and unequal `T/U`;
- mixed lengths and padding invariance within one batch;
- blank in the configured first/last position;
- maximum target ID and rejection of blank/invalid target IDs;
- terminal blank behavior;
- configured reduction and optional loss settings;
- non-finite inputs and invalid lengths;
- randomized fp32 lattices across a fixed seed matrix;
- loss values and gradients with respect to logits.

Initial candidate thresholds, frozen in `parity-tolerances.yaml` before comparison:

- fp32 loss: `rtol <= 1e-5`, `atol <= 1e-5` on small lattices;
- logit-gradient cosine similarity: `>= 0.99999`;
- padding changes no valid-example loss beyond the same tolerance;
- all valid loss and gradients are finite.

### P2.4 Real-shape TPU spike

After CPU correctness is green, recreate v6e-1 and compile representative static `T/U/V` buckets derived from P1 data. Measure:

- compilation time;
- forward and backward HBM;
- steady-state loss time over at least 100 post-compile calls;
- whether the frame-wise `lax.map` serializes too much work;
- whether chunked joint evaluation improves throughput without changing results.

**P2 gate**

- Loss and gradients pass the full reference matrix.
- Classic RNN-T rather than HAT semantics are proven.
- The largest representative loss bucket compiles on v6e-1 without a full vocabulary lattice or non-finite results.
- A measured implementation is selected: frame-wise, chunked, or a documented alternative.

## 8. Phase P3: port the model and convert the checkpoint

**Goal:** reproduce the complete base checkpoint in a config-driven JAX model.

### Implementation order

1. Feature/length/mask contract, initially tested using saved NeMo features to isolate the encoder.
2. Predictor LSTM and joint network, including PyTorch LSTM gate order and bias semantics.
3. Exact convolutional subsampling and length arithmetic from the pinned reference.
4. Normalization, FFN, relative-position attention, and convolution modules.
5. One FastConformer block, then the full configured stack.
6. Greedy RNN-T decoder for fixed-token parity.
7. Declarative NeMo/Hugging Face to JAX mapping and inverse export mapping.

### Required conversion behavior

- Explicitly map dense, convolution, depthwise convolution, Q/K/V, LSTM, normalization, embedding, and BatchNorm-running-state layouts.
- Fail on any missing, duplicate, extra, or shape-mismatched model tensor.
- Emit `conversion-report.json` with 100% mapping coverage, parameter counts, shapes, dtypes, transforms, and hashes.
- Never silently initialize a missing pretrained tensor.

### Parity order

1. CPU fp32 component parity using identical saved features, lengths, masks, and parameters.
2. Full-model fp32 parity and matching output lengths.
3. Selected bf16 comparisons to establish TPU tolerances.
4. Fixed-audio greedy token parity.
5. Export the untrained converted model back to the supported NeMo/Hugging Face runtime and rerun `dev-clean` WER.

**P3 gate**

- All component tolerances and exact length/mask checks pass.
- Every parameter/buffer is mapped.
- Golden greedy tokens match, except approved numerical ties.
- Pre-training converted/exported WER remains inside the frozen non-inferiority margin.

## 9. Phase P4: integrate a complete training step

**Goal:** prove the complete gradient and persistence path on realistic shapes.

### Work

1. Implement the exact Optax optimizer, schedule, weight-decay exclusions, clipping, accumulation, stochastic-layer RNG, and precision policy from P1.
2. Use static audio/text buckets and masks to avoid shape-triggered recompilation.
3. Apply encoder rematerialization before considering sharding.
4. Compare one identical batch in NeMo and JAX:
   - scalar/per-example loss;
   - gradient norms and selected gradient tensors;
   - parameter-update norms and direction;
   - mutable BatchNorm state updates where applicable.
5. Compile the full `encoder -> predictor -> joint -> loss -> backward -> optimizer` step on v6e-1 using a representative bucket. Record peak HBM and the XLA memory report.
6. Overfit the fixed twenty-record training set for up to 500 steps.
7. Save parameters, mutable state, optimizer, step, RNG, data cursor, and config/source hashes using Orbax.
8. Restore and verify that the next step matches an uninterrupted control run within the frozen tolerance.

If the full step OOMs, capture the failing shape and compiler memory evidence. This is a **conditional result**, not permission to benchmark only the encoder. The next experiment would be v6e-4 with a `data x fsdp` mesh.

**P4 gate**

- A representative full step fits v6e-1, or a documented v6e-4/sharding requirement is produced.
- Loss falls by at least 90% on the tiny fixed set without NaN/Inf.
- All intended parameter groups receive finite updates.
- Orbax save/restore produces the same next-step behavior as the uninterrupted control.

## 10. Phase P5: matched training and WER non-regression

**Goal:** distinguish mathematical correctness from equivalent learning behavior.

### Work

1. Start NeMo/GPU and JAX/TPU from the same source checkpoint.
2. Use the same immutable training subset, order, bucket policy, tokenizer, augmentation policy, optimizer values, scheduler, number of examples, and total steps.
3. Run a short but non-overfit training comparison; the initial proposal is 500-1,000 optimizer steps, finalized after measured throughput.
4. Export the JAX checkpoint back to the supported reference format.
5. Evaluate the NeMo-trained and JAX-trained checkpoints using the **same** pinned evaluator and decoder on `dev-clean`.
6. Report loss curves, WER, substitutions/deletions/insertions, paired utterance differences, and the exact evaluation count.

Default planning gate, to be approved in `wer-gates.yaml` before execution:

- base conversion: no unexplained fixed-token differences and no more than 1% relative WER regression;
- matched short training: JAX WER is no worse than NeMo by more than 1% relative, with paired utterance analysis for any difference;
- no claim about Indic quality until the client checkpoint/data evaluation is run later.

**P5 gate**

- Both base-conversion and post-training WER gates pass.
- Any numerical divergence is bounded and does not produce a material WER regression.

## 11. Phase P6: full-step efficiency and economics

**Goal:** determine whether the correct pipeline is worth scaling.

### Work

1. Benchmark an in-memory/precomputed-feature path first to isolate TPU compute.
2. Run at least 200 post-compilation full training steps for each representative static bucket.
3. Then benchmark the end-to-end sharded GCS input path with prefetch.
4. Record:
   - compile time separately from steady state;
   - p50/p95 full-step time;
   - peak HBM and host memory;
   - audio-seconds and examples processed per chip-second;
   - input-wait percentage;
   - checkpoint time;
   - optional MFU using an explicitly documented FLOP estimate.
5. Run a matched short GPU measurement using the same examples and effective batch.
6. Calculate cost per processed audio-hour using actual TPU/GPU prices supplied for the comparison.

Do not use a universal `1 GB/s` input target or encoder-forward MFU as the go/no-go gate. The input path passes when it supplies at least 1.2x measured model consumption and contributes less than 10% input wait. The primary comparison is full-step audio throughput and cost.

**P6 gate**

- Full-step throughput is stable without unexplained recompiles.
- Input does not bottleneck the measured model step.
- TPU/GPU throughput and cost evidence support either scale, optimize, or stop.

## 12. Phase P7: prove framework reuse

**Goal:** validate that this is a reusable FastConformer + RNN-T pipeline, not a hard-coded Parakeet 1.1B program.

Use `nvidia/parakeet-rnnt-0.6b` as the first low-cost reuse candidate, pinned independently.

### Work

1. Add only a normalized model config and model-specific tensor mapping.
2. Reuse the same audio, FastConformer, predictor, joint, loss, trainer, checkpoint, and reporting code.
3. Run complete tensor mapping, selected forward parity, fixed-token parity, RNN-T loss integration, one full TPU step, and a short overfit.
4. Record every code change required for the second checkpoint.

**P7 gate**

- The second model passes without modifying shared mathematical modules.
- New work is limited to approved configuration/mapping differences.
- Any architecture-specific exception is added to the support matrix rather than hidden in shared defaults.

Passing 0.6B proves reuse across sizes in the Parakeet family. A later second FastConformer RNN-T family is still required before claiming broad cross-family compatibility. CTC and TDT reuse the encoder/trainer but require separately qualified heads, losses, and decoders.

## 13. Go/no-go decision

| Result | Decision |
|---|---|
| P1-P7 green | Fund the v6e-4 two-hour CPT pilot and ASR-on-JAX R1 engineering |
| Correctness/WER green, v6e-1 full step OOM | Run a bounded v6e-4 FSDP qualification before deciding economics |
| Correctness green, throughput/cost red | Optimize the joint/loss/input path against a timebox, then reassess |
| Loss/gradient parity red | Do not start full model training; resolve RNN-T semantics first |
| Forward or base-WER parity red | Do not train; fix model conversion/frontend/masking first |
| Matched post-training WER red | Do not claim training compatibility, even if loss decreases |
| Reuse gate red | Treat the output as a Parakeet-specific implementation, not a reusable framework |

The final `verdict.md` must report each gate with its measured number and artifact link. “It ran on TPU” is not a fitment verdict.

## 14. Indicative execution schedule

| Engineering days | Primary work | Parallel work |
|---|---|---|
| 0-1 | P0 TPU readiness and source lock | Release TPU after smoke |
| 1-3 | P1 NeMo fixtures and baseline WER | Loss extraction, license/provenance, CPU tests |
| 3-6 | P2 loss parity and real-shape spike | P3 model primitives and converter |
| 5-9 | P3 full model parity and base WER | Trainer/checkpoint skeleton |
| 9-11 | P4 full step, overfit, Orbax | WER/export automation |
| 11-13 | P5 matched training/WER and P6 performance | Defects and evidence review |
| 13-15 | P7 second-model reuse and final verdict | Documentation |

Plan approximately **2-3 calendar weeks with two experienced JAX/ASR engineers**. A single engineer or a custom-loss performance rewrite extends the calendar. TPU capacity wait time is external.

## 15. Explicitly out of scope

- Full client/Indic CPT and its production WER decision
- Multi-host execution
- GKE and production orchestration
- Native JAX serving and beam search
- Broad NeMo Python/API compatibility
- CTC, TDT, HAT, Canary/AED, diarization, TTS, or non-ASR collections
- Production-scale data migration

## 16. Immediate handoff

### Operator now

1. Finish creating the US `v6e-1` TPU VM.
2. Run P0.1-P0.2 and retain the `TPU_READY=...` output plus `pip freeze`.
3. Record project, zone, TPU name, direct-versus-queued provisioning, and runtime version.
4. Release the TPU after the smoke unless a compiled test is ready immediately.

### Engineering now

1. Create the `poc/` package skeleton and source-lock schema.
2. Pin the public checkpoint and NeMo reference environment.
3. Extract the licensed AXLearn DP and its tests.
4. Implement classic RNN-T semantics and the NeMo loss/gradient fixture harness.
5. Begin FastConformer conversion and loss work in parallel after P1 fixtures are frozen.

## 17. Primary references

- [TPU v6e configurations](https://cloud.google.com/tpu/docs/v6e)
- [Run JAX on a Cloud TPU VM](https://cloud.google.com/tpu/docs/run-calculation-jax)
- [Manage queued TPU resources](https://cloud.google.com/tpu/docs/queued-resources)
- [NVIDIA NeMo ASR models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [Parakeet RNN-T 1.1B model files](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
- [AXLearn transducer implementation](https://github.com/apple/axlearn/blob/main/axlearn/common/transducer.py)
- [AXLearn transducer tests](https://github.com/apple/axlearn/blob/main/axlearn/common/transducer_test.py)
- [AXLearn Apache-2.0 license](https://github.com/apple/axlearn/blob/main/LICENSE)
- [Mddct JAX RNN-T reference](https://github.com/Mddct/jax-rnnt-loss)
