# NeMo ASR on JAX Migration Plan

**Status:** engineering plan for an ASR-only JAX compatibility layer and its first TPU qualification  
**Collection scope:** NeMo Automatic Speech Recognition (ASR) models only  
**First vertical slice:** pinned `nvidia/parakeet-rnnt-1.1b` artifact  
**Target runtime:** JAX + Flax NNX + Optax + Orbax on TPU v6e  
**First-slice assumption:** supervised continued training with transcripts and an RNN-T objective

## 1. Program boundary

This is **not a rewrite of the full NeMo toolkit** and it is **not only a one-off Parakeet port**. The deliverable is a standalone JAX implementation of the reusable NeMo **ASR** model stack, with explicit config/checkpoint compatibility at its boundaries. NeMo/PyTorch remains the behavior oracle; the TPU training runtime is native JAX rather than a second backend hidden inside NeMo's PyTorch/Lightning internals.

| In scope | Out of scope |
|---|---|
| ASR manifests, audio frontend, augmentation, tokenizers, masks, static buckets, and WER evaluation | NeMo Core or a general replacement for Lightning/Hydra across every NeMo collection |
| Conformer/FastConformer encoders, starting with the Parakeet FastConformer contract | TTS, LLM/Megatron, multimodal, vision, and diffusion models |
| CTC, RNN-T, TDT, and hybrid ASR heads, losses, and decoding in a staged rollout | Speaker recognition, diarization, speech classification, and unrelated speech collections |
| NeMo/Hugging Face config and checkpoint import/export for supported ASR families | Drop-in source compatibility with every NeMo Python class, callback, or trainer feature |
| Optax training, JAX sharding, Orbax resume, TPU qualification, and serving export | Porting every legacy or experimental ASR model in the first release |

The compatibility contract is therefore at the **ASR model, data, recipe, and artifact level**. A supported model must load its pinned NeMo/Hugging Face artifact, reproduce agreed reference behavior, train and resume on TPU, and export into a validated serving format.

## 2. ASR support roadmap

The shared layer is designed before the first port, but support is earned model family by model family. “ASR support” must not be interpreted as every model on the NeMo ASR page passing on day one.

| Release | Model families and objectives | Shared capability delivered | Exit evidence |
|---|---|---|---|
| R1 — reference vertical slice | Parakeet RNN-T 1.1B | Audio contract, FastConformer, RNN-T predictor/joint/loss, converter, trainer, Orbax, export | One-epoch two-hour TPU CPT, stop/resume, parity, and clean-runtime export validation |
| R2 — Parakeet CTC | Selected Parakeet CTC variants | Reuse FastConformer; add CTC head/loss/decoder and model mappings | Per-variant conversion, forward/loss parity, tiny overfit, and TPU smoke |
| R3 — FastConformer hybrid | Selected hybrid CTC-RNN-T models | Dual heads, weighted objectives, decode/export selection, and multi-head config translation | Both objectives pass parity and a joint training smoke |
| R4 — Parakeet TDT | Selected Parakeet TDT variants | Add the TDT joint, duration objective, loss, decode behavior, and mappings | TDT loss/gradient parity, tiny overfit, and TPU smoke |
| R5 — reusable Conformer stack | Selected Conformer/FastConformer CTC, RNN-T, hybrid, and cache-aware streaming models | Generalized encoder variants and streaming caches | Published support matrix with a regression pack for every supported architecture |
| R6 — demand-driven ASR expansion | Canary/AED, HAT, multi-talker, Citrinet/QuartzNet/Jasper, or other ASR families selected by business need | New decoder or encoder families without changing the common trainer/artifact contracts | Separate estimate and acceptance suite per architecture family |

Parakeet RNN-T is first because it exercises the common FastConformer path and the highest-risk training primitive, the RNN-T loss. Once that vertical slice passes, Parakeet CTC and TDT reuse most of the encoder, data, conversion, training, checkpoint, and deployment stack.

## 3. ASR compatibility architecture

| Layer | Responsibility | First release |
|---|---|---|
| Reference compatibility | Parse supported NeMo/Hugging Face ASR configs; inventory, import, and export tensors; generate golden fixtures | Parakeet RNN-T mapping and pinned oracle |
| Audio and data | Manifest schema, feature extraction, normalization, augmentation, tokenization, length math, bucketing, prefetch | Exact Parakeet training contract |
| Encoders | Shared subsampling, attention, convolution, normalization, masks, and streaming-state interfaces | FastConformer-XXL used by Parakeet 1.1B |
| Objectives | Head, loss, and decode interfaces for CTC, RNN-T, TDT, hybrid, and later AED | RNN-T predictor, joint, loss, and greedy parity decoder |
| Model adapters | Compose an encoder/objective from a translated config and expose family-specific compatibility rules | `parakeet_rnnt` adapter |
| Training runtime | Optax recipe, bf16/fp32 policy, RNG, metrics, rematerialization, JAX mesh, checkpoint/resume | Four-chip v6e data-parallel CPT |
| Artifact runtime | Orbax schema, provenance, reverse export, independent load/transcription validation | Orbax to supported Parakeet serving artifact |

This layered boundary prevents Parakeet-specific tensor names or defaults from leaking into shared encoders, losses, data code, and training state.

## 4. Where the migration happens

| Stage | Execution environment | Why |
|---|---|---|
| Reference extraction | Pinned NeMo/Transformers container on CPU/GPU | Source of truth for config, tensors, intermediate activations, loss, gradients, and decoded tokens |
| JAX primitives and converter | Local/CI CPU first | Cheap deterministic unit tests; no TPU capacity should be consumed here |
| Full JAX forward parity | CPU, then one inexpensive TPU compile smoke | Isolates framework/layout errors before distributed training |
| RNN-T loss and gradient parity | CPU with small synthetic lattices; GPU reference where needed | Correctness is easier to inspect at tiny static shapes |
| Tiny overfit and recipe parity | CPU/GPU first, then one TPU chip or v6e-4 | Proves the model can learn, not only infer |
| Four-chip training/checkpoint | GKE v6e-4 | Validates JAX mesh, throughput, GCS checkpointing, and resume |
| Export validation | Independent pinned Transformers/NeMo CPU/GPU runtime | Proves the JAX result is consumable without the training stack |

The GKE infrastructure preflight can run in parallel with early migration work. The paid Parakeet TPU training gates must wait until first-slice migration gates **JM0-JM7** below pass.

```text
Pinned reference pack
  -> JAX model primitives
  -> deterministic weight mapping
  -> forward/decode parity
  -> RNN-T loss + gradient parity
  -> matched train step + tiny overfit
  -> v6e-4 training
  -> Orbax stop/resume
  -> reverse export + clean-runtime validation
```

## 5. Frozen first-slice model contract

The public pinned configuration currently describes:

- 80-bin acoustic features and 8x convolutional subsampling with 256 channels;
- encoder hidden size 1024, 42 blocks, 8 attention heads, FFN size 4096, convolution kernel 9;
- RNN-T decoder hidden size 640 with two decoder layers;
- vocabulary size 1025 with blank token ID 1024;
- maximum 10 emitted symbols per encoder step for decoding.

These values are an inventory starting point, not permission to infer missing behavior. The migration must extract from the pinned reference implementation the exact feature normalization, positional encoding, residual order/scales, convolution padding, normalization epsilon, Q/K/V layout, recurrent-cell gate order, joint projection/activation, tokenizer normalization, blank/pad treatment, dropout/layerdrop policy, and length arithmetic.

## 6. Proposed ASR JAX repository shape

This is the target shape across releases. R1 creates the shared contracts and the Parakeet/FastConformer/RNN-T paths; CTC, TDT, streaming, and other family modules are added only in their funded releases.

```text
nemo-asr-jax/
  pyproject.toml
  Dockerfile
  configs/
    models/parakeet-rnnt-1.1b.yaml
    recipes/parakeet-rnnt-1.1b-cpt.yaml
    parity-tolerances.yaml
  src/nemo_asr_jax/
    compat/
      nemo_config.py
      inventory.py
      mapping.py
      import_hf.py
      import_nemo.py
      export_hf.py
    audio/
      features.py
      augmentation.py
      masks.py
    encoders/
      subsampling.py
      attention.py
      conformer_conv.py
      conformer_block.py
      fastconformer.py
    objectives/
      base.py
      ctc.py
      predictor.py
      joint.py
      rnnt.py
      tdt.py
    losses/
      ctc.py
      rnnt_reference.py
      rnnt_xla.py
      tdt.py
    decoding/
      ctc_greedy.py
      rnnt_greedy.py
      tdt_greedy.py
    models/
      registry.py
      parakeet_rnnt.py
    data/
      manifest.py
      tokenizer.py
      batching.py
      input_pipeline.py
    training/
      state.py
      optimizer.py
      train_step.py
      mesh.py
      metrics.py
    checkpointing/
      orbax_manager.py
    cli/
      convert.py
      parity.py
      train.py
      export.py
  reference/
    dump_reference.py
    compare_reference.py
  tests/
    unit/
    conversion/
    parity/
    loss/
    integration/
```

Create modules only as their release is funded; the shared interfaces reserve extension points without shipping placeholder support. A family is reported as supported only after its acceptance suite passes. Use one public CLI contract throughout: `python -m nemo_asr_jax.cli.convert|parity|train|export`.

## 7. ASR platform work packages

These packages make the first vertical slice reusable without asking it to implement every ASR family immediately.

| Work package | Output | Parakeet R1 dependency |
|---|---|---|
| ASR0 — support policy | Versioned matrix of model family, objective, training/inference/export status, pinned source version, and tolerances | Define Parakeet RNN-T 1.1B as the only R1 supported row |
| ASR1 — compatibility schema | Typed normalized config plus declarative tensor-mapping and provenance schema | Translate the pinned Parakeet artifact without embedding NeMo objects in JAX execution |
| ASR2 — common audio/data contract | Shared features, tokenization, augmentation, masks, buckets, manifest, and WER interfaces | Reproduce the Parakeet recipe exactly |
| ASR3 — encoder/objective APIs | Stable composition boundaries for encoders, heads, losses, decode state, and streaming state | Implement FastConformer + RNN-T behind those boundaries |
| ASR4 — trainer/checkpoint contract | Objective-independent train state, Optax recipe, metrics, sharding, and complete Orbax resume | Run Parakeet CPT and later reuse it for CTC/TDT |
| ASR5 — conformance suite | Golden fixtures, conversion coverage, forward/loss/gradient tests, tiny overfit, export validation | Supply all Parakeet acceptance evidence |
| ASR6 — release qualification | Per-model compatibility card and GKE TPU qualification report | Publish R1 only after JM0-JM10 pass |

## 8. Parakeet RNN-T 1.1B first vertical slice

The following steps implement and qualify R1. They are the first delivery through the shared ASR interfaces above, not a separate Parakeet-only codebase.

### JM0 — Freeze objective, source, and recipe

**Work**

1. Confirm supervised RNN-T CPT versus encoder-only self-supervised CPT. This plan continues only with supervised RNN-T.
2. Pin the model repository commit, every source file hash, NeMo/Transformers version, PyTorch/CUDA image, tokenizer, and client checkpoint.
3. Export the exact current GPU training recipe: optimizer, scheduler, weight-decay exclusions, clipping, frozen layers, batch construction, accumulation, precision, SpecAugment, maximum audio/text lengths, seed, and RNN-T loss variant/reduction/FastEmit or pruning options.
4. Freeze 20 tiny-overfit records, approximately two hours of pilot train data, held-out data, and a fixed inference sample.

**Exit gate**

- `source-lock.json`, `SHA256SUMS`, resolved reference config, tokenizer files, resolved GPU recipe, and immutable data manifests are reviewed.
- No architecture or training value remains as an undocumented default.

### JM1 — Build the reference/golden-output harness

**Work**

1. Load the pinned artifact in the supported PyTorch Transformers implementation and NeMo. For the public base model they should cross-check each other; for a client-modified checkpoint/recipe, the pinned client NeMo run remains authoritative and any Hugging Face difference must be documented.
2. Inventory every tensor: name, shape, dtype, parameter count, and hash.
3. With dropout/layerdrop disabled and fixed lengths/masks, dump golden outputs for:
   - feature extraction and feature lengths;
   - subsampling output and lengths;
   - encoder blocks 0, 1, 20, and 41 plus final encoder output;
   - predictor embeddings, recurrent states, and output;
   - encoder/predictor projections and selected joint logits;
   - greedy decoder token IDs and normalized text.
4. For small synthetic lattices, dump RNN-T loss and gradients with respect to logits.
5. Store golden inputs/outputs in a versioned, compact test pack. Do not dump multi-gigabyte full-run activations.
6. Approve per-component fp32 and bf16 comparison thresholds in `parity-tolerances.yaml` before looking at JAX results.

**Exit gate**

- The reference harness is repeatable in a clean pinned container.
- Two independent runs produce identical hashes for deterministic artifacts.
- Golden pack covers forward, decoder, loss, and gradient behavior.

### JM2 — Implement JAX model primitives bottom-up

Use Flax NNX for the new implementation and explicit JAX arrays/masks. Keep every module independently callable so reference hooks can compare it.

**Implementation order**

1. Feature/tokenizer/length contract and static padding masks.
2. Token embedding, recurrent prediction network, joint network, and blank conventions. These expose LSTM gate and blank-ID errors cheaply.
3. Convolutional subsampling and exact output-length arithmetic.
4. Normalization, feed-forward module, relative/positional attention, and convolution module.
5. One FastConformer block, then a scanned/stacked 42-block encoder and encoder projection.
6. Full end-to-end logits.
7. Greedy RNN-T decoder for parity only.

**Testing rule**

Each primitive gets shape, mask, padding, dtype, gradient, and golden-output tests before it is composed into the next level. Variable-length examples must include zero padding, odd lengths, and maximum bucket lengths.

**Exit gate**

- Randomly initialized JAX modules compile and run on CPU.
- Every component has unit tests, stable parameter paths, and no data-dependent shapes inside `jit`.
- A full randomly initialized forward pass returns correct output/length shapes.

### JM3 — Build deterministic weight import and round-trip mapping

Use the Hugging Face safetensors layout as the primary public import surface; support `.nemo` only where the client checkpoint requires it.

**Work**

1. Create a declarative mapping table with source name, JAX path, source/destination shapes, transform, and dtype.
2. Implement and test explicit transforms for:
   - dense kernel transpose;
   - convolution and depthwise-convolution axis order;
   - packed versus separate Q/K/V projections;
   - recurrent input/recurrent kernels, biases, and exact LSTM gate order;
   - normalization parameters and non-parameter state, including convolution-module BatchNorm running statistics where present;
   - stacked/repeated encoder layer numbering.
3. Fail conversion on every unmapped, duplicate, missing, or shape-mismatched tensor. Never silently initialize a missing pretrained parameter.
4. Emit `mapping-report.json` with parameter counts and hashes.
5. Implement the inverse mapping early enough to prove import -> export preserves source-layout tensors.

Mandatory layout checks include:

| Source representation | JAX/Flax check |
|---|---|
| Dense `[out, in]` | transpose to `[in, out]` |
| Conv1D/Conv2D `[out, in, spatial...]` | move spatial axes first and output channel last |
| Depthwise convolution | preserve feature groups; do not treat as a regular convolution |
| Packed/separate Q/K/V | explicitly verify head and projection layout before transpose |
| LSTM input/recurrent kernels and two biases | preserve the pinned source gate order and bias semantics; do not rely on an implicit Flax cell convention |
| BatchNorm running mean/variance | map mutable statistics and verify framework momentum semantics |
| Embedding/LayerNorm vectors | normally unchanged, but still verify names, shapes, and dtype |

**Exit gate**

- 100% of expected tensors are mapped or explicitly approved as non-model metadata.
- Total parameter count and each leaf shape match.
- Import -> inverse export round-trip reproduces source tensors within dtype-preserving tolerance.

### JM4 — Establish layer-by-layer and end-to-end inference parity

**Work**

1. Compare JAX fp32 against the pinned reference with identical features, masks, lengths, parameters, and disabled stochastic layers.
2. Debug from the first divergent component; do not compensate downstream with ad hoc scaling.
3. Run the golden set through the full encoder, predictor, joint, and greedy decoder.
4. Repeat selected comparisons in bf16 to set realistic TPU tolerances.

**Exit gate**

- All agreed per-component thresholds pass.
- Output lengths and masks match exactly.
- Greedy token IDs match on the fixed golden set, or every difference has an approved numerical tie explanation.
- The public base checkpoint runs through `python -m nemo_asr_jax.cli.parity` with a machine-readable pass report.

### JM5 — Implement the supervised RNN-T loss

This is the critical-path item for supervised CPT. NeMo's production RNN-T training kernel is CUDA-oriented; it is not inherited by merely porting model layers.

**Work**

1. Implement a simple fp32 JAX reference dynamic program for tiny tensors.
2. Validate it against the pinned reference over exhaustive/random small `T` and `U`, including empty/short labels, maximum lengths, blank handling, and padded batches.
3. Validate analytic gradients against the reference and finite differences.
4. Implement an XLA-efficient version with static padded shapes and length masks using JAX control-flow primitives.
5. Avoid materializing the production-scale `[batch, time, label, vocabulary]` joint tensor. Spike chunked/sequential joint evaluation or a custom VJP and choose based on measured memory/step time, not elegance.
6. Accumulate log probabilities and forward/backward recurrences in fp32; assert finite loss/gradients.

**Exit gate**

- Loss and logit gradients pass the approved reference tolerances across the test matrix.
- Padding does not change valid-example loss.
- The optimized loss passes the largest practical pre-TPU test and its compiled program does not require the prohibited full joint tensor. Actual v6e HBM fit is a JM8 gate.
- The implementation has a benchmark and a regression test, not only an example notebook.

If JM0 selects self-supervised CPT, replace JM5 with the exact BEST-RQ/masked-prediction objective, quantizer/target pipeline, and reference tests. Do not keep the RNN-T predictor/joint in the training critical path unnecessarily.

### JM6 — Match the data and training recipe

**Work**

1. Reproduce feature extraction, text normalization, tokenizer IDs, blank insertion policy, length filtering, SpecAugment, and batching from the frozen recipe.
2. Bucket audio/text to a small set of static shapes; mask padding in attention and loss.
3. Add deterministic shuffling, resumable data position, parallel GCS reads, host prefetch, and device prefetch.
4. Implement the exact optimizer/schedule, weight-decay mask, clipping, gradient accumulation, mixed precision, RNG handling, and metrics.
5. Start with replicated parameters/optimizer on four chips because the 1.1B model is expected to fit with rematerialization and a conservative microbatch. Add state/FSDP sharding only if measured memory requires it.
6. Rematerialize encoder blocks and keep numerically sensitive normalization, softmax, and RNN-T recurrence in fp32.

**Exit gate**

- JAX and reference pipelines produce identical features/token IDs/lengths for the golden records.
- One train step on the same batch produces matching loss and directionally matching parameter gradients/updates.
- A fixed 20-sample dataset overfits materially toward zero loss without NaN/Inf.
- Repeated runs with the same seed reproduce the documented tolerance.

### JM7 — Freeze checkpoint and deployment contracts before paid pilot work

**Work**

1. Store parameters, optimizer state, global step, RNG state, loss-scale state if any, data position, resolved config, source/mapping hashes, and mesh metadata in Orbax.
2. Write to a run-specific GCS prefix with atomic/finalized checkpoint discovery.
3. Wait for asynchronous save completion before a successful process exit.
4. Restore on CPU where possible and on the original four-device mesh; validate the next step matches an uninterrupted run.
5. Implement SIGTERM-triggered checkpoint finalization and test it against the container locally; the Kubernetes grace-period test occurs in JM8.
6. Freeze the CLI shown below, build the application image, pin it by digest, and make the converted-model URI an explicit input to `train`:

   ```text
   python -m nemo_asr_jax.cli.convert
   python -m nemo_asr_jax.cli.parity
   python -m nemo_asr_jax.cli.train --mode=forward-smoke|tiny-overfit|train ...
   python -m nemo_asr_jax.cli.export ...
   ```

**Exit gate**

- Save -> restore -> next-step state matches the uninterrupted control run.
- An incomplete checkpoint is ignored.
- A deliberate stop/relaunch resumes from the latest finalized step, including optimizer and data position.
- The digest-pinned image exposes the frozen CLI, and `train` rejects a missing or unvalidated converted-model artifact.

### JM8 — Compile and scale on TPU v6e

**Work**

1. Run a one-device compile/forward/train-step smoke if useful, then the GKE v6e-4 forward-smoke gate.
2. Use a four-device mesh with batch data parallelism and gradient all-reduce.
3. Run tiny overfit, then 200-500 repeated steps to separate first-compile time from steady-state time.
4. Measure HBM, host memory, input wait, compile time, step time, examples/audio-hours per second, and numerical health.
5. If memory does not fit, reduce microbatch/rematerialize first; introduce parameter/optimizer sharding only with a new parity and checkpoint test matrix.

**Exit gate**

- All four chips perform training work.
- The largest pilot bucket compiles on v6e without OOM and does not materialize a prohibitive full joint tensor.
- Peak HBM remains below the agreed safety threshold (initial target: 85%) or an approved sharding change is introduced with renewed parity/checkpoint tests.
- Loss behavior tracks the matched short GPU run within the agreed tolerance.
- Steady-state steps are stable, input wait is bounded, and no unexplained recompiles occur.
- Orbax kill/resume passes on GKE.

### JM9 — Run the two-hour epoch and reverse export

**Work**

1. Run the requested one-epoch two-hour smoke with the exact consumed sample/audio count logged.
2. Export the chosen final Orbax step through the inverse mapping to a versioned Hugging Face safetensors layout.
3. Emit source run/step, dtype/mapping report, hashes, config, and tokenizer.
4. In a separate pinned supported runtime, load the export and compare fixed-audio tokens/text with the pre-export JAX model and golden reference.

**Exit gate**

- Final checkpoint, metrics, provenance, and export are complete.
- A clean runtime loads the export without the JAX training environment.
- Fixed-audio validation passes and the held-out pilot WER is reported.

### JM10 — Publish R1 compatibility card and operational handoff

JM7 freezes the image and deployment interface before JM8 uses GKE. After JM9, collect the conformance, TPU, resume, export, provenance, and known-limitation evidence into a versioned Parakeet RNN-T 1.1B support card. Link the exact application image digest and the matching [manual GKE v6e runbook](gke-v6e-jax-cpt-runbook.md). The runbook owns infrastructure, Workload Identity, TPU scheduling, Job lifecycle, cost control, and teardown; it does not own model migration logic.

**Exit gate**

- The R1 support matrix row names exact source and application versions and makes no claim about unqualified ASR families.
- A new operator can execute the runbook using the published artifact URIs and reproduce the evidence package.

## 9. First-slice acceptance matrix

| Capability | Required evidence | Blocks |
|---|---|---|
| Source lock | hashes, pinned images/versions, resolved config/recipe | all implementation |
| Model structure | 100% tensor inventory and parameter count | weight conversion |
| Weight conversion | mapping report and round-trip | forward parity |
| Forward parity | component report and exact shapes/lengths | training |
| Decoder parity | fixed token sequences | export/serving validation |
| RNN-T loss | loss and gradient test matrix | supervised training |
| Train step | same-batch loss/update comparison | tiny overfit |
| Tiny overfit | falling/near-zero loss and finite gradients | TPU pilot |
| Orbax | uninterrupted versus resumed next-step comparison | reliability claim |
| Four-chip TPU | stable measured steps on all chips | two-hour run |
| Reverse export | clean-runtime load and fixed-audio match | serving handoff |

No stage is accepted from a screenshot or a single successful command. Each gate produces a machine-readable artifact retained with the code commit and source hashes.

## 10. Engineering sequencing and staffing

Indicative calendar sequence with two experienced JAX/ASR engineers:

| Calendar block | Primary work | Parallel work |
|---|---|---|
| Weeks 1-2 | JM0-JM1 source lock and golden harness | ASR0-ASR3 contracts plus GKE/identity preflight |
| Weeks 2-4 | JM2-JM4 model, conversion, and forward parity | JM5 reference RNN-T loss plus data validation |
| Weeks 4-6 | JM5 production loss and memory work | JM6 trainer/input pipeline and export skeleton |
| Weeks 5-7 | JM6-JM7 tiny overfit and Orbax | Container, serving validator, and GKE manifests |
| Weeks 7-8 | JM8 four-chip TPU, resume, and performance fixes | Evidence automation and defects |
| Weeks 8-9/buffer | JM9-JM10 two-hour run, export, review, and handoff | Documentation and go/no-go package |

For budgeting, separate the reusable ASR foundation from wider model coverage:

| Investment | Indicative engineering effort | What the business can claim |
|---|---|---|
| ASR contract and repository foundation (ASR0-ASR3 skeleton) | 2-3 engineer-weeks, overlapping the first slice | An explicit ASR-only architecture, support policy, and conformance contract |
| Parakeet RNN-T 1.1B first production-quality slice | 10-14 engineer-weeks, including foundation work | Repeatable Parakeet CPT on v6e with conversion, parity, resume, and export evidence |
| R2-R3 Parakeet/FastConformer CTC plus hybrid support | Additional 4-7 engineer-weeks | Evidence that the shared layer generalizes beyond one objective |
| R4 Parakeet TDT | Additional 4-7 engineer-weeks, subject to the selected checkpoint and loss/decode contract | A second transducer objective on the shared FastConformer runtime |
| Broad coverage of the NeMo ASR model catalog | Roughly 9-15 engineer-months, released in selected families | An allowlisted ASR compatibility product, not full NeMo compatibility |

With two experienced JAX/ASR engineers, plan approximately **7-9 calendar weeks** for the Parakeet RNN-T first slice because conversion, loss, training, and infrastructure can overlap but parity gates remain sequential. Keep 2-3 engineer-weeks of contingency for custom-gradient or production-memory work in the RNN-T loss. Encoder-only self-supervised CPT is materially smaller and needs its own objective-specific estimate.

## 11. Business decision after the pilot

The pilot produces evidence for one of three ASR decisions:

1. **Proceed with targeted Parakeet JAX CPT:** parity, reliability, quality, and economics pass.
2. **Use JAX only for encoder/self-supervised CPT:** RNN-T training cost/risk is not justified, but encoder pretraining is viable.
3. **Expand the NeMo ASR-on-JAX support matrix:** qualify Parakeet CTC, selected hybrid models, Parakeet TDT, and then selected Conformer/FastConformer families through the same interfaces and gates.

The third decision is not an automatic extension of the first. It needs committed ASR model families, a compatibility policy, dedicated owners, and a multi-release roadmap. It still does not authorize work outside ASR.

## 12. Primary references

- [NVIDIA NeMo ASR supported-model documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)
- [Pinned Parakeet model files](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/tree/2acc4c61eede2f52ddefe74935de1930a9064d4a)
- [Parakeet RNN-T configuration](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/blob/2acc4c61eede2f52ddefe74935de1930a9064d4a/config.json)
- [Flax NNX documentation](https://flax.readthedocs.io/en/stable/)
- [Orbax checkpointing](https://orbax.readthedocs.io/en/stable/guides/checkpoint/orbax_checkpoint_101.html)
- [GKE TPU planning](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)
