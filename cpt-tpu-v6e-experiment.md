# Continued Pretraining on TPU v6e — Experiment Design

**Status:** draft v0.1 — several inputs still open, see §1
**Model:** `parakeet-rnnt-1.1b` + client CPT checkpoint (FastConformer-XXL encoder, RNNT decoder)
**Hardware:** TPU v6e (Trillium)
**Scope:** training only. The inference/serving migration is a separate track.
**JAX migration:** see [NeMo ASR on JAX migration plan](nemo-asr-jax-migration-plan.md); Parakeet RNN-T 1.1B is its first vertical slice.  
**Manual deployment:** see [GKE TPU v6e JAX CPT runbook](gke-v6e-jax-cpt-runbook.md). It deliberately starts with a lower-cost single-host v6e-4 smoke test; the measured pilot can still move to v6e-8.

---

## 1. Open inputs — fill before locking the plan

Everything downstream depends on these. The first one is load-bearing enough to change the entire framework decision.

| # | Input | Value | Impact if unknown |
|---|---|---|---|
| 1 | **Is CPT self-supervised (BEST-RQ / wav2vec-style) or supervised transducer training?** | TBD | Decides whether an XLA RNNT loss must be written at all |
| 2 | Dataset size — hours of audio, TB on disk | TBD | Sets cluster size and run duration |
| 3 | Planned epochs / total steps | TBD | Same |
| 4 | Where the data lives today | TBD | Sets data-migration effort |
| 5 | Tokenizer: already expanded for Indic, or expanding in this run? | TBD | Changes vocab size, joint tensor memory, downstream decode cost |
| 6 | Full fine-tune, or encoder partly frozen? | TBD | Changes optimizer memory by up to 4× |
| 7 | Current GPU recipe: batch size, grad accumulation, loss variant (pruned? FastEmit?) | TBD | Reference point for the port |
| 8 | Current run duration and GPU count | TBD | Baseline for the cost comparison |
| 9 | Language mix and target hours per language | TBD | Eval set design |
| 10 | v6e slice sizes available under the discount | TBD | Caps the sizing options |

---

## 2. Objective

Run continued pretraining of the client's ASR model on TPU v6e, producing a checkpoint of equal or better quality than the existing GPU recipe would produce, at lower cost and with materially fewer run failures.

The driver is **not** raw speed. It is that the client's current GPU providers are unstable, and multi-day training runs are the workload least tolerant of that.

**Explicit non-goals for this experiment:**

- The inference/serving migration (tracked separately)
- Any change to the model architecture
- Any change to the training data or recipe beyond what porting requires

If the port changes training dynamics, that's a bug, not a feature.

---

## 3. Success criteria

**Quality — the hard gate**

- Resulting checkpoint's WER on the Indic eval set is within 1% relative of what the GPU recipe produces from the same data and hyperparameters
- English regression set does not degrade beyond the GPU recipe's own drift
- Loss curve over the first N steps tracks the GPU run within noise (see §8, phase 2)

**Throughput**

- Model FLOP utilisation ≥ 25% *(conformers run lower than transformers; treat this as a floor, and calibrate the target after phase 2)*
- Audio-hours processed per chip-hour ≥ TBD, set once phase 2 measures the real number

**Reliability**

- A run survives a deliberate mid-run kill and resumes from checkpoint with no quality impact
- No manual intervention required for a full-length run

**Cost**

- Total chip-hours × discounted rate is below the GPU equivalent for the same job

---

## 4. Background

`parakeet-rnnt-1.1b` is a FastConformer-XXL encoder (~1.0B params) with an RNNT decoder (~0.1B). Audio is downsampled 8× to one frame per 80ms. The client has already run one CPT pass adding Indic languages; this experiment moves that process onto TPU.

Training is a far better fit for TPU than the streaming inference case:

- No latency constraint — throughput only
- Batches are naturally large, so the matrix unit stays fed
- No autoregressive decode loop, so no data-dependent control flow
- Long runs benefit from reserved capacity, which is the problem being solved

Google trained a comparable 1B RNN-T (the USM model, arXiv 2406.02887) on TPUs, so the workload class is proven. The obstacle is framework, not hardware.

---

## 5. The fork: which kind of CPT

### Path A — self-supervised pretraining (BEST-RQ, wav2vec2-style)

Encoder-only, masked-prediction or contrastive objective. **No RNNT loss involved.** Fixed shapes throughout, no lattice, no variable-length label dimension.

This is the most TPU-friendly training workload in speech — BEST-RQ was designed at Google specifically for TPUs. If this is the path, most of §6.1's memory pressure disappears and the project is considerably simpler.

### Path B — supervised transducer training

Requires the RNNT loss. **NeMo's implementation is CUDA-only** — it depends on a Numba CUDA kernel, and the CPU fallback is a reference implementation, not something to train on. There is no TPU path.

Someone must write an XLA-compatible forward-backward over the transducer lattice, with static shapes and masking for variable T and U. This is the single largest piece of new work in the project, and it is on the critical path.

> **Resolve input #1 before anything else.** The two paths have different frameworks, different memory profiles, and different risk.

---

## 6. Framework decision

### The options

**Option 1 — NeMo + PyTorch/XLA.** Keeps the existing recipe, configs, and data pipeline. But under path B you still have to write the transducer loss, and NeMo carries a lot of CUDA-shaped assumptions in augmentation and data loading that will surface one at a time.

**Option 2 — JAX/Flax reimplementation.** Port the FastConformer encoder and RNNT head to Flax, load the existing weights. More work up front; much cleaner steady state. Prior art exists for transducer losses in JAX, and TPU-side transducer training has been published since 2018.

### Recommendation: Option 2, conditional on path B

The reasoning is narrow rather than ideological. **Under path B you are writing an XLA transducer loss either way.** If that's unavoidable, it is easier to write in JAX, and there is existing work to reference. Under path A, where no such loss is needed, Option 1 becomes much more attractive and should be reconsidered.

**Strategic note worth raising with the client:** if training moves to JAX on TPU, the model ends up in a TPU-native framework, and the inference migration's hardest prerequisite is already paid for. The serving work that was scoped as a possible quarter of effort shrinks considerably. Sequencing training first is defensible on those grounds alone.

**The cost of Option 2** is that the Flax implementation must be numerically equivalent to NeMo's, or the existing checkpoint won't load and the WER baseline is meaningless. See §8 phase 1 — that parity work is the critical path, not the training itself.

---

## 7. Technical design

### 7.1 Memory budget (path B, full fine-tune, per chip)

```
bf16 params                       2.2 GB
fp32 master weights               4.4 GB
Adam m + v                        8.8 GB
gradients (bf16)                  2.2 GB
                                 -------
optimizer + params               17.6 GB

encoder activations, batch 8      ~3 GB    (with gradient checkpointing)
RNNT joint [B,T,U,V] + gradient   ~2.5 GB  (B=8, T=375, U=200, V=1024)
                                 -------
total                            ~23 GB    against 32 GiB per chip
```

**The model fits on one chip.** That is the most important line in this document, because it means **no tensor parallelism and no pipeline parallelism** — the two things that make distributed training genuinely difficult.

The joint tensor scales linearly with batch at roughly 300 MB per utterance, and it — not the encoder — caps per-chip batch size. If headroom is needed: chunk the loss over T, or use a pruned transducer loss. Check what the current GPU recipe already does here (input #7); they will have hit the same wall.

*Recompute this table once input #5 lands — a larger vocabulary scales the joint tensor proportionally.*

### 7.2 Sharding

Start the v6e-4 pilot with a four-chip data-parallel mesh, replicated parameters/optimizer state, encoder rematerialization, and gradient all-reduce. The preliminary budget in §7.1 suggests this should fit and it keeps parity and checkpoint debugging simple.

Measure real peak HBM on the largest pilot bucket. If it exceeds the 85% safety threshold or forces an uneconomic microbatch, move to a `2 x 2` data-by-FSDP mesh and shard Adam state plus the largest kernels. That change must repeat parity, train-step, and Orbax restore tests because sharding changes the compiled and checkpointed state layout.

Do not introduce tensor or pipeline parallelism for this single-host pilot unless measurement proves data parallelism plus selective FSDP insufficient.

### 7.3 Cluster sizing

Rough training-FLOP arithmetic:

```
6 FLOPs per parameter per token (forward + backward)
12.5 frames per second of audio

encoder                    1.0e9 × 6 × 45,000  ≈  270 TFLOP per audio-hour
joint + attention overhead                     ≈   ~30 TFLOP
                                                  ----------
                                                  ~300 TFLOP per audio-hour
```

At 30% MFU on v6e's 918 TFLOPs → ~275 TFLOPs effective → **roughly 1 audio-hour per second per chip**, or ~3,600 audio-hours per chip-hour.

```
chip-hours  ≈  (dataset hours × epochs) ÷ 3,600
```

| Dataset | Epochs | Audio-hours | v6e-8 | v6e-32 |
|---|---|---|---|---|
| 5,000 | 5 | 25k | ~1 hr | ~15 min |
| 20,000 | 10 | 200k | ~7 hrs | ~1.7 hrs |
| 50,000 | 20 | 1M | ~35 hrs | ~9 hrs |

**These are order-of-magnitude estimates.** The MFU assumption is the soft spot and must be replaced with a measurement from phase 2 before any slice size is committed. Even trebled for real-world inefficiency, a v6e-8 covers most realistic CPT runs inside a day.

### 7.4 Slice progression

| Slice | Hosts | Purpose |
|---|---|---|
| v6e-1 | single | development, loss correctness, layer-by-layer parity |
| v6e-8 | single | pilot runs, MFU measurement, most production CPT |
| v6e-32+ | multi | only if measurement demands it |

**Stay single-host as long as possible.** v6e-1 through v6e-8 is one process and one debugger. From v6e-16 up you inherit synchronised startup, distributed checkpointing, and failure modes unrelated to the model. Per §7.3, you may never need to cross that line.

### 7.5 Data pipeline

At ~1 audio-hour per second per chip, a v6e-8 consumes **eight audio-hours every second** — roughly 920 MB/s of sustained raw-audio read.

An underfed input pipeline is the most common cause of poor TPU utilisation, and at this rate it is not hypothetical. Requirements:

- Data staged in GCS in a **sharded sequential format** — ArrayRecord, WebDataset, or TFRecord. Individual WAV files will not work; per-object latency destroys throughput.
- **Precompute mel features** rather than decoding audio on the fly. Halves bytes read and removes STFT from the host critical path.
- Aggressive parallel reads and deep prefetch, using the host's ~180 vCPUs.

**Measure input throughput in isolation before the model exists.** If the pipeline cannot sustain ~1 GB/s, chip count is irrelevant.

### 7.6 Checkpointing

Non-negotiable given that run stability is the entire rationale.

- Checkpoint to GCS every N steps (Orbax if JAX), N tuned so that at most ~30 minutes of work is ever at risk
- Include optimizer state, not just weights
- Test resume explicitly as a phase 2 exit criterion — an untested resume path is not a resume path

### 7.7 Precision

- bf16 compute with fp32 master weights and fp32 optimizer state — standard mixed precision
- fp32 for LayerNorm statistics, softmax, and the joint output projection
- The transducer loss forward-backward should accumulate in fp32; log-probabilities over a long lattice are precision-sensitive

Confirm what precision the GPU recipe currently trains in (input #7). If it is already bf16 autocast, there is no precision change to justify at all.

---

## 8. Phases

### Phase 0 — Preparation

Resolve inputs §1. Stage data to GCS in a sharded format and benchmark read throughput standalone. Dump the model config from the `.nemo`. Freeze the eval sets and confirm the WER harness reproduces known numbers.

*Exit:* input pipeline sustains target throughput; eval harness reproduces the GPU baseline WER.

### Phase 1 — Correctness on a single chip

Implement (or port) the model. Establish numerical parity against NeMo layer by layer using golden activation dumps. Under path B, verify the transducer loss against `warprnnt_numba`'s CPU reference on small tensors — both loss value and gradients.

Then overfit a tiny batch to near-zero loss. This catches most implementation errors cheaply.

*Exit:* per-layer activations match NeMo within tolerance; loss and gradients match the reference implementation; a 20-sample batch overfits.

### Phase 2 — Pilot on v6e-8

Short run, a few hundred steps, against an identical short run on GPU with the same data order and seed.

Measure: MFU, audio-hours per chip-hour, step time breakdown, input pipeline utilisation, memory headroom. Test checkpoint-and-resume by killing the job mid-run.

*Exit:* loss curve tracks the GPU run within noise; MFU ≥ 25%; resume verified.

### Phase 3 — Full CPT run

Slice size chosen from phase 2's measured throughput, not from §7.3's estimates.

*Exit:* run completes without manual intervention.

### Phase 4 — Validation

Full WER evaluation on Indic and English sets. Compare against both the pre-CPT checkpoint and, if available, a GPU-trained equivalent.

*Exit:* success criteria in §3 met.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| XLA transducer loss is harder than estimated (path B) | Verify against CPU reference early, in phase 1, before any scale-up |
| Flax port not numerically equivalent to NeMo | Golden activation dumps, layer-by-layer; treated as critical path |
| Input pipeline bottlenecks the chips | Benchmarked standalone in phase 0, before the model exists |
| MFU well below 25% | Phase 2 measures it; slice sizing is deferred until then |
| Joint tensor memory blows up with an expanded vocabulary | Recompute §7.1 once input #5 lands; chunked or pruned loss in reserve |
| Multi-host complexity if v6e-8 proves insufficient | Sizing suggests it won't be; keep single-host as the default |
| Training dynamics differ subtly (dropout, augmentation, data order) | Phase 2 compares loss curves against GPU with matched seed and data order |

---

## 10. Metrics to log from day one

| Metric | Why |
|---|---|
| Step time, and its breakdown | Finds the bottleneck |
| MFU | Sizing, and the honest efficiency number |
| Audio-hours per chip-hour | The cost comparison unit |
| Input pipeline utilisation / host idle time | Detects data starvation |
| Peak HBM per chip | Headroom for batch increases |
| Loss curve, versus GPU reference | Correctness of the port |
| Gradient norm | Catches divergence early |
| Checkpoint write duration | Hidden throughput cost |

---

## 11. Things to revisit as this doc matures

- Replace every FLOP and MFU estimate in §7.3 with phase 2 measurements
- Recompute §7.1 once the tokenizer question resolves
- Add the specific eval set composition once language mix is known
- Add hyperparameters once the current GPU recipe is documented
- Decide whether tokenizer expansion is bundled into this run — if another CPT pass is happening anyway, it is far cheaper to do now than later, and it would substantially reduce Indic decode cost downstream
