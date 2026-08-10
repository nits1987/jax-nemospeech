# Code review: `fcrnnt-jax`

**Reviewed against:** [poc-steps.md](../poc-steps.md) (execution plan v0.2)
**Date:** 2026-08-10
**Baseline at review time:** `pytest -q` → **54 passed** (~105 s, CPU)
**Environment:** jax 0.11.0, flax 0.12.8, optax 0.2.8, numpy 2.4.6

## Scope and method

Read all eight modules under `src/fcrnnt_jax/` plus the test suite, configs, and notices. Claims below
were checked by execution, not by inspection alone — each finding states its evidence. Where I could
not verify something without the pinned NeMo oracle, it is filed as a *risk to settle in P1*, not as a
defect.

> **Note on concurrent edits.** `audio.py` (151→225 lines), `cli.py` (711→844) and `tests/test_audio.py`
> were modified partway through this review, adding `load_pcm16_wav` and the `audio-smoke` command.
> Findings below are against the current files and line numbers were re-verified afterwards.
> `log_mel_spectrogram` itself was not changed by that edit.

## Verdict

The mathematical core is sound. I built an independent FP64 NumPy RNN-T implementation (separate alpha
and beta recursions, analytic edge posteriors, no shared code) and cross-checked it against
`rnnt_loss.py`:

| Reduction | Loss max abs error | Logit-gradient max abs error |
|---|---|---|
| `none` | 9.99e-07 | 9.15e-07 |
| `sum` | 9.60e-07 | 9.15e-07 |
| `mean_batch` | 4.80e-07 | 4.58e-07 |
| `mean` | 2.30e-07 | 2.29e-07 |
| `mean_volume` | 3.20e-07 | 3.05e-07 |

Both value and gradient agree to ~1e-6 on a mixed-length padded batch. **No correctness defect was
found in the loss, the DP, or its custom VJP.** That is the single most important thing this PoC needed
to get right, and it is right.

Findings are therefore about efficiency, diagnostics, and frontend parity risk — not about broken math.

| # | Severity | Area | Finding |
|---|---|---|---|
| F1 | **Medium** | `training.py` | Optimizer transformation computed twice on every step |
| F2 | Low | `conversion.py` | Prefix-collision check is incomplete; caught late, with the wrong error type |
| F3 | **Medium** | `audio.py` | Symmetric Hann window; torch/NeMo default is periodic |
| F4 | **Medium** | `audio.py` | Non-centred STFT; torch/NeMo default is `center=True` |
| F5 | Low | `audio.py` | Variance is biased (`/n`); torch `.std()` default is unbiased (`/(n-1)`) |
| F6 | Low | `audio.py` | Log floor clamps rather than adding an epsilon |
| F7 | Low | `rnnt_loss.py` | Two different negative-infinity sentinels in one recurrence |
| F8 | Low | `model.py` | LayerDrop runs the block before discarding it |
| F9 | Low | `cli.py` / tests | The NumPy loss oracle is duplicated in two places |
| F10 | Info | design | Summing LSTM biases on import makes export non-invertible |

---

## F1 — Optimizer transformation computed twice per step (Medium)

**Location:** [training.py:233](src/fcrnnt_jax/training.py#L233) and [training.py:242](src/fcrnnt_jax/training.py#L242)

```python
updates, _ = state.tx.update(gradients, state.opt_state, state.params)   # line 233
...
state = state.apply_gradients(grads=gradients, ...)                      # line 242
```

**Evidence.** Flax's `TrainState.apply_gradients` calls `self.tx.update(grads_with_opt, self.opt_state,
params_with_opt)` internally — confirmed by reading the installed `flax/training/train_state.py`. So
`tx.update` runs twice per step. Both calls read the same `state.opt_state`, so they return identical
updates and **the trained result is correct**. The second computation exists only to populate the
`update_norm` metric.

**Why it matters.** The chain is `clip_by_global_norm` + `adamw`, so each call is a full traversal of
the parameter tree with AdamW moment arithmetic. At 1.1B parameters that is a substantial per-step tax
in both compute and optimizer-state memory traffic — and P4 and P6 exist specifically to measure
per-step time and peak HBM. The measurement is inflated by the instrumentation.

**Fix.** Compute once and apply explicitly. Note that `apply_gradients` also increments `step`, so that
must be carried over:

```python
updates, new_opt_state = state.tx.update(gradients, state.opt_state, state.params)
metrics = {..., "update_norm": _tree_l2_norm(updates)}
state = state.replace(
    step=state.step + 1,
    params=optax.apply_updates(state.params, updates),
    opt_state=new_opt_state,
    batch_stats=batch_stats,
    rng=next_rng,
    data_cursor=state.data_cursor + batch["features"].shape[0],
)
```

`tests/test_training.py` already asserts `step == 1` and `data_cursor == 1` after one step, so it will
catch a mistake in this refactor.

---

## F2 — Prefix-collision check is incomplete (Low)

**Location:** [conversion.py:772-775](src/fcrnnt_jax/conversion.py#L772-L775)

```python
sorted_targets = sorted(seen_targets)
for index, target in enumerate(sorted_targets[:-1]):
    following = sorted_targets[index + 1]
    if following.startswith(target + "."):
```

This compares only **adjacent** entries in the sorted list. A third target can sort between a colliding
pair and hide it. Verified counterexample — targets `a`, `a!x`, `a.b` sort as `['a', 'a!x', 'a.b']`
because `!` (0x21) < `.` (0x2e), so `a` and `a.b` are never compared.

**Correction to an earlier draft of this review: this is not a soundness hole.** I tested it, and the
collision is still caught downstream by `unflatten_pytree`
([conversion.py:119](src/fcrnnt_jax/conversion.py#L119)), which raises
`ValueError: path collision while inserting 'a.b'`. Bad mappings do **not** get through the converter.

What is actually wrong is narrower, and worth fixing anyway because this module's whole value
proposition is failing closed *legibly*:

- the error surfaces as a bare `ValueError` rather than the typed `MappingValidationError`;
- it bypasses the error-aggregation in `_validate_rules`, so the operator sees one problem at a time
  instead of the full list;
- it fires **after** every transform has already been applied, rather than during validation.

**Fix.** For each target, test each of its ancestor prefixes against the target set — exact, and
`O(n · depth)`:

```python
for target in seen_targets:
    parts = target.split(".")
    for depth in range(1, len(parts)):
        ancestor = ".".join(parts[:depth])
        if ancestor in seen_targets:
            errors.append(f"prefix-colliding targets {ancestor!r} and {target!r}")
```

---

## F3–F6 — Frontend parity risks (settle against P1 fixtures, do not guess)

`audio.py` documents NeMo as the frontend oracle, so these are not bugs against a stated contract.
They are listed individually because each is independently checkable once P1 fixtures exist, and each
would otherwise surface as an encoder forward-parity failure — the most expensive possible place to
discover a frontend difference.

**F3 — Symmetric vs periodic Hann window.** [audio.py:203](src/fcrnnt_jax/audio.py#L203).
Verified: `jnp.hanning(8)` equals `np.hanning(8)` (symmetric, both endpoints exactly 0) and differs
from the periodic window at every interior point:

```
jnp.hanning(8) : [0. 0.1883 0.6113 0.9505 0.9505 0.6113 0.1883 0.]
periodic hann  : [0. 0.1464 0.5    0.8536 1.     0.8536 0.5    0.1464]
```

`torch.hann_window` defaults to `periodic=True`. Confirm which the pinned NeMo preprocessor uses.

**F4 — Non-centred STFT.** [audio.py:122](src/fcrnnt_jax/audio.py#L122),
[audio.py:197](src/fcrnnt_jax/audio.py#L197). Frames start at sample 0 with no padding.
`torch.stft` defaults to `center=True` with reflect padding. This changes both the **frame count** and
the **time alignment** of every frame, so it would also perturb subsampled encoder lengths — likely the
highest-impact of these four.

**F5 — Biased variance.** [audio.py:215-222](src/fcrnnt_jax/audio.py#L215-L222). Divides by `count`;
`torch.Tensor.std()` defaults to unbiased `count - 1`. Relative difference is largest on short
utterances, which the parity set deliberately includes.

**F6 — Log floor semantics.** [audio.py:208](src/fcrnnt_jax/audio.py#L208).
`log(max(mel, 1e-10))` clamps, producing exact `log(1e-10)` plateaus with zero gradient in silence;
an additive `log(mel + eps)` guard does not. Different behaviour on silent frames.

---

## F7 — Mixed negative-infinity sentinels (Low)

**Location:** [rnnt_loss.py:159](src/fcrnnt_jax/rnnt_loss.py#L159)

The module defines `_NEG_INF = -1.0e30` and uses it consistently for lattice padding — except inside
`_prefix_log_probs`, where the label term pads with true `-jnp.inf`:

```python
via_label = jnp.pad(label + carry[:-1], (1, 0), constant_values=-jnp.inf)
```

This is currently safe: the backward pass is analytic (`custom_vjp`), so no gradient flows through the
`-inf`, and the padding-invariance and empty-target tests pass. It is flagged only because sentinel
discipline is load-bearing in exactly this file, and a future change to the VJP or to `_suffix_log_probs`
could turn `-inf - -inf` into a NaN. Use `_NEG_INF` for consistency, or add a comment explaining why
this one site deliberately differs.

---

## F8 — LayerDrop evaluates the block it discards (Low, already documented)

**Location:** [model.py:360-365](src/fcrnnt_jax/model.py#L360-L365)

The block is fully computed, then `jnp.where(keep, x, block_input)` selects. Consequences: a "dropped"
block still updates its BatchNorm running statistics, and LayerDrop yields no compute saving. This is
already listed honestly in `PARITY_GAPS`; repeated here because it blocks the P4 gate on
*mutable-state update parity*, so it needs disabling for those runs rather than merely noting.

---

## F9 — The loss oracle is duplicated (Low)

`_numpy_single_rnnt_loss` / `_numpy_batch_loss` at
[cli.py:108](src/fcrnnt_jax/cli.py#L108) and [cli.py:137](src/fcrnnt_jax/cli.py#L137) implement the
same alpha recursion as `_np_rnnt_loss` at
[tests/test_rnnt_loss.py:22](tests/test_rnnt_loss.py#L22). Two copies of the reference, neither
importable, and §4 of the plan names a `rnnt_reference.py` module that does not exist. They can drift
independently, and a drifted oracle is worse than no oracle. Consolidate into
`src/fcrnnt_jax/rnnt_reference.py` and import from both call sites.

---

## F10 — Summing LSTM biases forecloses export (Info)

`PARITY_GAPS` states that NeMo's separate input and recurrent LSTM biases are **summed** on import, and
`_pytorch_lstm_bias` implements that. Summation is lossy, so that tensor can never be exported back.
P3 parity step 5 and P5 step 4 both require exporting to the reference runtime for WER scoring.

Decide before P5 depends on it: either store both biases in the JAX tree, or accept and document that
the LSTM bias does not round-trip. This is a design consequence, not a defect.

---

## What is done well

Worth recording, because these are the parts that should not be traded away under schedule pressure:

- **The loss is correct and independently verifiable** — see the cross-check table above.
- **Streamed joint (`rnnt_loss_from_joint`)** keeps persistent edge tensors at `O(B·T·U)` instead of
  materializing `O(B·T·U·V)` logits, with `jax.checkpoint` rematerializing each frame in backward.
  This is the design decision that makes a 1025-vocabulary lattice fit at all.
- **Analytic `custom_vjp`** avoids retaining every diagonal scan intermediate.
- **Fail-closed conversion** — duplicate sources, duplicate targets, missing/extra tensors, and shape
  mismatches are all errors, with SHA-256 per tensor and an order-independent set digest.
- **Dynamic-length validation is jit-safe** — invalid lengths yield NaN via a clamp-and-sentinel path
  rather than a host callback or an out-of-bounds program.
- **Intellectual honesty about what is not proven.** `PARITY_GAPS`, `compatibility_claim:
  "shape-fitment-only"`, `"synthetic-shape-fitment-not-checkpoint-or-wer-parity"`, and the new
  `audio-smoke` reporting `asr_decode_executed: false` with an explicit blocker. A PoC whose reports
  cannot overstate themselves is worth more than one that runs faster.

## Coverage gaps against poc-steps.md §4

Not defects — unwritten code, each blocking a named gate.

| Missing | Gate blocked |
|---|---|
| Greedy RNN-T decoder | P3 gate "Golden greedy tokens match" (P3.6). Nothing decodes today; `audio-smoke` names this as its blocker. |
| `rnnt_reference.py` | §4 file list — see F9. |
| `export.py` | P3 parity step 5, P5 step 4 (export back, rerun WER). |
| `reference/` NeMo scripts | P1 fixture generation. Needs a pinned NeMo/PyTorch environment. |

## How to reproduce the checks in this review

```bash
cd fcrnnt-jax && python -m pytest -q          # 54 passed
```

- **F1:** `python -c "import inspect,flax.training.train_state as t; print(inspect.getsource(t.TrainState.apply_gradients))"`
  — shows the internal `self.tx.update(...)`.
- **F2:** build a `MappingSpec` with targets `a`, `a!x`, `a.b` and call `convert_tensors`; observe it
  raises from `unflatten_pytree` (line 119), not from `_validate_rules`.
- **F3:** compare `jnp.hanning(8)` against `0.5 - 0.5*cos(2*pi*arange(8)/8)`.
- **Loss cross-check:** requires the FP64 NumPy reference from F9 (prototyped during review, not
  committed); the table above is its output across all five reductions.
