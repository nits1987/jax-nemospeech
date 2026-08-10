# Manual Compute Engine v6e-1 Deployment and PoC Run Guide

This runbook deploys the `fcrnnt_jax` PoC to an **already-provisioned, single-chip Compute Engine TPU v6e VM in a US zone**. The expected machine type is `ct6e-standard-1t`, and the required Google-managed OS image family is `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e` in the `ubuntu-os-accelerator-images` project.

This image uses the Compute Engine TPU resource model. Therefore every lifecycle command in this guide uses `gcloud compute instances`, `gcloud compute ssh`, or `gcloud compute scp`. Do not substitute legacy `gcloud compute tpus tpu-vm` or queued-resource commands. Run the workstation/Cloud Shell commands from a machine with `gcloud` installed, then run the marked VM commands over SSH.

This is not a GKE node-pool runbook. A future GKE deployment should use a version-pinned TPU/JAX container and GKE's TPU node configuration; the Compute Engine OS image-family flag is not a container image and should not be copied into Kubernetes manifests.

The required validation order is:

1. identify the exact Compute Engine TPU instance, machine type, and resolved OS image;
2. transfer the source and install an isolated environment;
3. run CPU tests, then prove JAX sees the TPU;
4. run loss, model, training, benchmark, and checkpoint gates in order;
5. copy all evidence off the ephemeral VM;
6. delete the exact Compute Engine TPU instance.

Do not continue after a failed correctness gate. A first JAX call can spend several minutes compiling; benchmark only post-compilation steps.

## 1. Set operator variables

Run in Cloud Shell or a local Bash shell. Use the actual values from provisioning. `ZONE` must be the US zone containing the VM, not only a region.

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export ZONE="YOUR_US_ZONE"
export TPU_NAME="YOUR_COMPUTE_ENGINE_TPU_INSTANCE_NAME"

export MACHINE_TYPE="ct6e-standard-1t"
export IMAGE_PROJECT="ubuntu-os-accelerator-images"
export IMAGE_FAMILY="ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e"

# Local source folder and local evidence destination.
export LOCAL_SOURCE_DIR="${PWD}/fcrnnt-jax"
export LOCAL_EVIDENCE_DIR="${PWD}/fcrnnt-jax-evidence"

gcloud config set project "${PROJECT_ID}"
gcloud auth list --filter=status:ACTIVE
mkdir -p "${LOCAL_EVIDENCE_DIR}/preflight"
```

If the instance is already provisioned, skip this creation command. It is included only as the exact on-demand recreation command for a single v6e chip with the required image:

```bash
gcloud compute instances create "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-project="${IMAGE_PROJECT}" \
  --image-family="${IMAGE_FAMILY}" \
  --maintenance-policy=TERMINATE
```

Verify all values before SSH or cleanup. Save the immutable instance and image evidence locally:

```bash
gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,status,zone,machineType,scheduling.provisioningModel,scheduling.instanceTerminationAction,scheduling.onHostMaintenance,deletionProtection,disks,networkInterfaces)' \
  | tee "${LOCAL_EVIDENCE_DIR}/preflight/00-instance.yaml"

INSTANCE_STATUS="$(gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(status)')"

ACTUAL_MACHINE_TYPE_URI="$(gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(machineType)')"

MAINTENANCE_POLICY="$(gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(scheduling.onHostMaintenance)')"

test "${INSTANCE_STATUS}" = "RUNNING"
test "${ACTUAL_MACHINE_TYPE_URI##*/}" = "${MACHINE_TYPE}"
test "${MAINTENANCE_POLICY}" = "TERMINATE"

BOOT_DISK_URI="$(gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(disks[0].source)')"
BOOT_DISK_NAME="${BOOT_DISK_URI##*/}"

SOURCE_IMAGE_URI="$(gcloud compute disks describe "${BOOT_DISK_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='value(sourceImage)')"
SOURCE_IMAGE_NAME="${SOURCE_IMAGE_URI##*/}"

gcloud compute images describe "${SOURCE_IMAGE_NAME}" \
  --project="${IMAGE_PROJECT}" \
  --format='yaml(name,family,status,creationTimestamp)' \
  | tee "${LOCAL_EVIDENCE_DIR}/preflight/01-boot-image.yaml"

ACTUAL_IMAGE_FAMILY="$(gcloud compute images describe "${SOURCE_IMAGE_NAME}" \
  --project="${IMAGE_PROJECT}" \
  --format='value(family)')"

test "${ACTUAL_IMAGE_FAMILY}" = "${IMAGE_FAMILY}"

printf 'resolved_source_image=%s\n' "${SOURCE_IMAGE_URI}" \
  | tee "${LOCAL_EVIDENCE_DIR}/preflight/02-resolved-source-image.txt"
```

Proceed only if the instance is `RUNNING`, the machine type is `ct6e-standard-1t`, the maintenance policy is `TERMINATE`, and the resolved image reports the required family. Keep the resolved date-versioned image name as the reproducibility record; the family alias can advance over time. Also note the displayed provisioning model (`STANDARD`, `SPOT`, `FLEX_START`, or `RESERVATION_BOUND`) and whether the boot disk has `autoDelete: true`.

## 2. Transfer the source

Choose either SCP or Git. Do not use both in the same run.

### Option A: transfer the local working tree with SCP

From Cloud Shell/local Bash:

```bash
test -d "${LOCAL_SOURCE_DIR}"

gcloud compute scp \
  --recurse \
  "${LOCAL_SOURCE_DIR}" \
  "${TPU_NAME}:~/fcrnnt-jax" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

If `~/fcrnnt-jax` already exists, use a new remote directory name rather than overwriting an unverified tree.

If the instance has no external IP and Identity-Aware Proxy is configured, append `--tunnel-through-iap` to every `gcloud compute ssh` and `gcloud compute scp` command in this guide.

### Option B: clone a pinned Git revision

Connect first:

```bash
gcloud compute ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

Then, on the TPU VM:

```bash
export REPO_URL="YOUR_REPOSITORY_URL"
export REPO_REF="YOUR_BRANCH_TAG_OR_COMMIT"

git clone "${REPO_URL}" "${HOME}/fcrnnt-jax"
cd "${HOME}/fcrnnt-jax"
git checkout "${REPO_REF}"
git rev-parse HEAD
```

For a private repository, use an SSH key or short-lived credential configured outside the repository. Do not put a token in `REPO_URL`, a shell history, or a log.

## 3. Enter the VM and initialize the run

If not already connected:

```bash
gcloud compute ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"
```

Run the remainder of Sections 3-9 on the TPU VM:

```bash
set -Eeuo pipefail

export POC_ROOT="${HOME}/fcrnnt-jax"
export POC_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export POC_RUN_DIR="${POC_ROOT}/runs/${POC_RUN_ID}"
export POC_LOG_DIR="${POC_RUN_DIR}/logs"
export POC_ARTIFACT_DIR="${POC_RUN_DIR}/artifacts"
export POC_VENV="${POC_ROOT}/.venvs/${POC_RUN_ID}"
export JAX_COMPILATION_CACHE_DIR="${POC_ROOT}/.jax-cache"

cd "${POC_ROOT}"
mkdir -p "${POC_LOG_DIR}" "${POC_ARTIFACT_DIR}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
chmod 700 "${JAX_COMPILATION_CACHE_DIR}"

TPU_SOURCE_IMAGE="$(curl -fsS \
  -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/image')"

{
  date -u --iso-8601=seconds
  hostname
  pwd
  git rev-parse HEAD 2>/dev/null || true
  printf 'metadata_source_image=%s\n' "${TPU_SOURCE_IMAGE}"
  cat /etc/os-release
  uname -a
  uname -r
  python3 --version
  systemctl --no-pager --type=service --state=running \
    | grep -Ei 'tpu|google' || true
} | tee "${POC_LOG_DIR}/00-host.txt"

case "${TPU_SOURCE_IMAGE}" in
  projects/ubuntu-os-accelerator-images/global/images/ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-*) ;;
  *) echo "Unexpected TPU boot image: ${TPU_SOURCE_IMAGE}" >&2; exit 1 ;;
esac

. /etc/os-release
test "${ID}" = "ubuntu"
test "${VERSION_ID}" = "22.04"
[[ "$(uname -r)" == 6.8.* ]]
```

The timestamped run and virtual-environment directories make repeated attempts independent. The private persistent compilation cache can be reused by the separate CLI processes; treat it as trusted executable content and never make it group/world writable. Google manages the TPU runtime, drivers, and agents in this image; do not run a distribution upgrade, replace its kernel, or manually install TPU drivers as part of this PoC.

## 4. Create the Python environment

On the TPU VM:

```bash
# The current qualified TPU lane uses JAX 0.11.0, which requires Python 3.12+.
# Ubuntu 22.04 defaults to an older Python, so install 3.12 only if absent.
if ! command -v python3.12 >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-dev python3.12-venv
fi

export PYTHON_BIN="python3.12"
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.version)
assert sys.version_info >= (3, 12), "JAX 0.11.0 requires Python >=3.12"
PY

"${PYTHON_BIN}" -m venv "${POC_VENV}"
source "${POC_VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "jax[tpu]==0.11.0"
python -m pip install -e ".[dev]"
python -m pip install "tpu-info==0.14.2"

python -m pip check
python -m pip show jax jaxlib libtpu flax optax orbax-checkpoint tpu-info \
  | tee "${POC_LOG_DIR}/01-package-details.txt"
python -m pip freeze | tee "${POC_LOG_DIR}/01-requirements.txt"
python -m fcrnnt_jax.cli --help | tee "${POC_LOG_DIR}/02-cli-help.txt"
```

The managed accelerator image supplies the host TPU software, but JAX and its matching `jaxlib`/`libtpu` packages still belong in the virtual environment. The explicit Python and JAX gates prevent pip on Ubuntu's older default Python from silently selecting an older JAX release. Do not upgrade `libtpu` separately from the pinned JAX TPU extra. If organizational policy disallows the Python PPA, stop and use an approved Python 3.12 interpreter or a separately qualified, digest-pinned Google JAX AI container; do not fall back silently. `pip check` must pass, and CLI help must list exactly these PoC commands: `devices`, `loss-smoke`, `model-smoke`, `train-smoke`, `benchmark`, and `checkpoint-smoke`.

If the repository later includes a tested TPU requirements lock, use it in a fresh per-run environment instead of the preceding unpinned install. The lock must preserve a TPU-enabled JAX/libtpu combination; a CPU-only JAX lock is not valid:

```bash
python -m pip install -r requirements-tpu.lock
python -m pip install --no-deps -e .
python -m pip check
```

### How this PoC uses the JAX AI Stack

The JAX AI Stack is a composable library ecosystem, not a Parakeet/FastConformer/RNN-T implementation and not a replacement for checkpoint conversion or WER parity testing. This PoC already uses four of its five core libraries directly: JAX for compilation/autodiff, Flax for the model, Optax for optimization, and Orbax for checkpoints.

- Add Grain when the PoC moves from synthetic/saved features to the two-hour audio dataset; deterministic, checkpointable input iteration is directly useful for CPT resume parity.
- Add XProf after correctness gates pass to profile the full training step.
- Consider Pallas or Tokamax only if profiling identifies the streamed RNN-T DP, joint network, or another kernel as the bottleneck.
- Defer Qwix quantization until the unquantized WER baseline is established. Pathways is unnecessary for a one-chip PoC, while MaxText, Tunix, and vLLM target different application layers.

Do not install the `jax-ai-stack` metapackage into this baseline environment yet. It pins a dated, integration-tested set of component versions and can re-resolve this repository's explicit Orbax/JAX constraints. If it is adopted later, select a dated stack release in `requirements-tpu.lock` and rerun the complete CPU, TPU, checkpoint, and parity suite. Installing the component libraries directly keeps this first result attributable to the repository's tested dependency contract.

Google also publishes preconfigured JAX AI Docker images with curated JAX, libtpu, Flax, Optax, Orbax, Grain, and diagnostic tooling. That is a strong option for the later GKE pipeline, but pin the image by immutable digest and test application dependencies without overriding its curated base. This manual single-VM baseline intentionally uses one visible virtual environment so its exact dependency resolution is easy to audit.

## 5. Run CPU tests before using TPU time

Force the CPU backend so unit failures are separated from TPU/runtime failures:

```bash
JAX_PLATFORMS=cpu python -m pytest -q 2>&1 \
  | tee "${POC_LOG_DIR}/03-pytest-cpu.log"
```

Gate: all tests pass. Do not use `-k` or skip failures merely to reach the TPU commands.

## 6. Prove JAX is using the v6e chip

Do not set `JAX_PLATFORMS=cpu` in this shell. Run:

```bash
unset JAX_PLATFORMS || true

python -m fcrnnt_jax.cli devices 2>&1 \
  | tee "${POC_LOG_DIR}/04-devices.log"
```

Gate: the output reports backend `tpu`, exactly one JAX device, and a successful finite matmul. A CPU device is a failure even if the command exits zero.

For additional environment evidence:

```bash
python - <<'PY' | tee "${POC_LOG_DIR}/05-jax-runtime.txt"
import importlib.metadata as metadata
import json
import jax
import jaxlib
import jax.numpy as jnp

devices = jax.devices()
x = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
y = jax.jit(lambda value: value @ value)(x)
y.block_until_ready()
matmul_finite = bool(jnp.isfinite(y).all())

print(json.dumps({
    "backend": jax.default_backend(),
    "device_count": len(devices),
    "devices": [{
        "description": str(device),
        "platform": device.platform,
        "device_kind": device.device_kind,
    } for device in devices],
    "jax_version": jax.__version__,
    "jaxlib_version": jaxlib.__version__,
    "libtpu_version": metadata.version("libtpu"),
    "matmul_dtype": str(y.dtype),
    "matmul_finite": matmul_finite,
    "matmul_shape": list(y.shape),
}, indent=2, sort_keys=True))
assert jax.__version__ == "0.11.0"
assert jax.default_backend() == "tpu"
assert len(devices) == 1
assert all(device.platform == "tpu" for device in devices)
assert matmul_finite
PY

tpu-info --version 2>&1 | tee "${POC_LOG_DIR}/05b-tpu-info-version.txt"
tpu-info 2>&1 | tee "${POC_LOG_DIR}/05c-tpu-info.txt"
```

Gate: the version report identifies accelerator type `v6e` and a compatible libtpu. The utilization snapshot can be mostly idle because the short Python process has exited; during the later benchmark, use a second SSH session with `tpu-info --streaming --rate 2` when live HBM, duty-cycle, or TensorCore evidence is needed.

## 7. Run the PoC gates in order

Each command has a deterministic zero-argument smoke. The commands below also use the supported flags to distinguish the cheap tiny checks from the full `parakeet-1.1b` fitment checks. Before running them, inspect their arguments and preserve the help output; this prevents an operator from applying flags from another revision.

```bash
for command in loss-smoke model-smoke train-smoke benchmark checkpoint-smoke; do
  python -m fcrnnt_jax.cli "${command}" --help \
    > "${POC_LOG_DIR}/help-${command}.txt"
done
```

### 7.1 RNN-T loss parity fixture

First run the self-contained deterministic fixture:

```bash
python -m fcrnnt_jax.cli loss-smoke 2>&1 \
  | tee "${POC_LOG_DIR}/06-loss-smoke.log"
```

If P1 produced an external NeMo value-and-gradient fixture, transfer it with the source or from approved storage and run it explicitly:

```bash
export RNNT_FIXTURE_PATH="${POC_ROOT}/fixtures/rnnt/YOUR_NEMO_FIXTURE.npz"
test -f "${RNNT_FIXTURE_PATH}"
cp "${RNNT_FIXTURE_PATH}" \
  "${POC_ARTIFACT_DIR}/rnnt-reference-fixture.npz"
sha256sum "${POC_ARTIFACT_DIR}/rnnt-reference-fixture.npz" \
  | tee "${POC_ARTIFACT_DIR}/rnnt-reference-fixture.sha256"

python -m fcrnnt_jax.cli loss-smoke \
  --fixture "${POC_ARTIFACT_DIR}/rnnt-reference-fixture.npz" 2>&1 \
  | tee "${POC_LOG_DIR}/06b-loss-nemo-fixture.log"
```

Gate: the fixed RNN-T loss value and logit gradient agree with the selected reference fixture within the CLI's printed tolerance; all values are finite. The report must identify the fixture/seed or fixture hash, tensor shapes, blank ID, precision, expected value, actual value, and maximum or relative error. The built-in fixture validates the local loss implementation; the external fixture is required before claiming NeMo parity.

### 7.2 FastConformer + predictor + joint forward

Run the tiny preset first, then the real Parakeet 1.1B architecture preset:

```bash
python -m fcrnnt_jax.cli model-smoke --preset tiny 2>&1 \
  | tee "${POC_LOG_DIR}/07a-model-smoke-tiny.log"

python -m fcrnnt_jax.cli model-smoke --preset parakeet-1.1b 2>&1 \
  | tee "${POC_LOG_DIR}/07b-model-smoke-parakeet-1.1b.log"
```

Gate: both configured models run end to end on TPU, produce the expected logits and lengths, and report finite outputs. The 1.1B run is the architecture fit gate. This remains a wiring/compilation smoke; it is not pretrained checkpoint parity unless the command explicitly loads pinned weights and reports a reference comparison.

### 7.3 Tiny end-to-end overfit

```bash
python -m fcrnnt_jax.cli train-smoke \
  --steps 100 \
  --workdir "${POC_ARTIFACT_DIR}/train-smoke" 2>&1 \
  | tee "${POC_LOG_DIR}/08-train-smoke.log"
```

Gate: `encoder -> predictor -> joint -> RNN-T loss -> backward -> optimizer` completes, gradients and updates are finite, and the deterministic tiny batch loss decreases by the command's declared threshold. A model-forward-only run does not pass this gate.

### 7.4 Full-step benchmark

```bash
python -m fcrnnt_jax.cli benchmark \
  --preset parakeet-1.1b \
  --warmup 3 \
  --steps 20 2>&1 \
  | tee "${POC_LOG_DIR}/09-benchmark.log"
```

Gate: the command reports compilation separately from steady state and benchmarks complete Parakeet 1.1B training steps, including loss, backward, and optimizer update. Preserve step count, batch/audio/text shapes, dtype, median or mean step time, throughput, and peak memory when available. Do not use first-call latency as steady-state performance. If the full preset OOMs, preserve that result; do not substitute a tiny benchmark and report it as the 1.1B fit gate.

### 7.5 Checkpoint/save/restore/resume

```bash
python -m fcrnnt_jax.cli checkpoint-smoke \
  --directory "${POC_ARTIFACT_DIR}/checkpoint-smoke" 2>&1 \
  | tee "${POC_LOG_DIR}/10-checkpoint-smoke.log"
```

Gate: parameters, optimizer state, model mutable state, step, and RNG are saved; a fresh restore resumes at the next step; and its next loss/update agrees with an uninterrupted control within the printed tolerance. A file-exists check alone is insufficient.

Record a compact operator result after all gates pass:

```bash
{
  echo "run_id=${POC_RUN_ID}"
  echo "completed_utc=$(date -u --iso-8601=seconds)"
  echo "cpu_tests=PASS"
  echo "devices=PASS"
  echo "loss_smoke=PASS"
  echo "model_smoke=PASS"
  echo "train_smoke=PASS"
  echo "benchmark=PASS"
  echo "checkpoint_smoke=PASS"
} | tee "${POC_ARTIFACT_DIR}/operator-verdict.txt"
```

If any command fails, stop at that command and write `FAIL` plus the reason instead of creating the all-pass verdict.

## 8. Preserve logs and artifacts on the VM

Capture runtime metadata and archive only this run:

```bash
{
  date -u --iso-8601=seconds
  python -m pip freeze
  python -m fcrnnt_jax.cli devices
} > "${POC_ARTIFACT_DIR}/runtime-manifest.txt" 2>&1

find "${POC_RUN_DIR}" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${POC_RUN_DIR}/SHA256SUMS"

tar -C "${POC_ROOT}/runs" \
  -czf "${POC_ROOT}/${POC_RUN_ID}.tar.gz" \
  "${POC_RUN_ID}"

echo "POC_RUN_ID=${POC_RUN_ID}"
echo "ARCHIVE=${POC_ROOT}/${POC_RUN_ID}.tar.gz"
```

Keep the printed run ID; shell variables from the VM do not automatically exist in Cloud Shell.

## 9. Copy evidence back before cleanup

Exit the TPU VM:

```bash
exit
```

Back in Cloud Shell/local Bash, set the exact run ID printed above and copy the archive:

```bash
export POC_RUN_ID="YOUR_PRINTED_RUN_ID"
mkdir -p "${LOCAL_EVIDENCE_DIR}"

gcloud compute scp \
  "${TPU_NAME}:~/fcrnnt-jax/${POC_RUN_ID}.tar.gz" \
  "${LOCAL_EVIDENCE_DIR}/" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"

tar -tzf "${LOCAL_EVIDENCE_DIR}/${POC_RUN_ID}.tar.gz"
sha256sum "${LOCAL_EVIDENCE_DIR}/${POC_RUN_ID}.tar.gz"
```

Open the archive and confirm it contains `operator-verdict.txt`, `runtime-manifest.txt`, `SHA256SUMS`, CPU test output, and all six CLI logs before deleting the TPU.

## 10. Troubleshooting

### `devices` reports CPU or no TPU

- Confirm `gcloud compute instances describe` reports the intended instance as `RUNNING`, with machine type `ct6e-standard-1t` and the expected resolved source image.
- Confirm `JAX_PLATFORMS` is unset and the command runs inside the new virtual environment.
- Run `python -m pip show jax jaxlib libtpu`; reinstall the supported `jax[tpu]` combination in a clean venv if packages conflict.
- Leave the Google-managed image's TPU drivers and agents unchanged; a missing backend is not a reason to replace the host driver stack.
- Do not accept a CPU fallback as a TPU pass.

### `libtpu` initialization or version error

- Preserve the complete traceback and `pip freeze`.
- Check compatibility between the managed OS image and the installed JAX/libtpu packages.
- Recreate only the virtual environment; do not mutate the system Python.
- Retry `devices` before any model command.

### SSH or SCP fails

- Recheck project, zone, instance name, `RUNNING` status, active gcloud identity, and IAM.
- For a private-only instance, use a reachable VPC host or append `--tunnel-through-iap` when Identity-Aware Proxy is configured.
- Avoid copying large checkpoints through SCP; stage them in Cloud Storage and record object generation/hash.

### First command appears hung

- JAX compilation can take minutes. Check CPU activity and TPU logs before interrupting.
- A second run with identical static shapes should reuse the compiled executable in-process; the benchmark command must exclude warm-up/compile time from steady-state numbers.

### Unexpected recompilations

- Check audio length, label length, batch size, dtypes, and Python booleans. They must map to a small set of static buckets.
- Preserve `JAX_LOG_COMPILES=1` evidence on a diagnostic rerun; do not enable it for the clean benchmark report.

### Out of memory

- Preserve the failing shapes, dtype, batch size, compiler message, and memory report.
- Confirm the loss does not materialize the full `[B,T,U,V]` lattice and that rematerialization is enabled where intended.
- Do not replace a representative full-step gate with encoder-only timing. An unsharded v6e-1 OOM is a valid conditional result that motivates a separately approved v6e-4/FSDP test.

### Loss parity fails

- Stop before model training.
- Compare blank ID, target padding, acoustic/label lengths, terminal-blank behavior, reduction, log-softmax precision, and fixture hash.
- Diagnose in fp32 on CPU first, then rerun the identical fixture on TPU.

### Training loss is non-finite or does not fall

- Stop before benchmarking.
- Inspect loss/gradient/update finiteness, masks, LSTM state, mutable normalization state, optimizer schedule, and dtype casts.
- Do not relax the declared overfit threshold after seeing results.

### Checkpoint resume differs

- Confirm the checkpoint contains parameters, optimizer state, mutable model state, global step, RNG, and the same config/source hashes.
- Ensure the comparison is the **next** step after restore against the same uninterrupted next step.

## 11. Clean up the exact Compute Engine instance

Cleanup is destructive. First confirm the evidence archive is safely copied back. Then describe the exact resource again and compare its name, project, zone, machine type, source disk, provisioning model, and deletion protection with the preflight record:

```bash
gcloud compute instances describe "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --format='yaml(name,status,zone,machineType,scheduling.provisioningModel,deletionProtection,disks)'

read -r -p "Type the instance name to confirm deletion: " CONFIRM_TPU_DELETE
test "${CONFIRM_TPU_DELETE}" = "${TPU_NAME}"

gcloud compute instances delete "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --quiet

REMAINING_INSTANCE="$(gcloud compute instances list \
  --project="${PROJECT_ID}" \
  --zones="${ZONE}" \
  --filter="name=('${TPU_NAME}')" \
  --format='value(name)')"

test -z "${REMAINING_INSTANCE}"
```

The final check must be empty. If `deletionProtection` is true, stop and have the resource owner remove that protection explicitly before retrying. Check `disks[].autoDelete` in the describe output; any disk with auto-delete disabled is a separate billable resource and is not deleted by the command above. The Compute Engine provisioning model does not create a separate legacy Cloud TPU queued resource to clean up.

## 12. Completion checklist

- [ ] Exact project, US zone, instance name, `ct6e-standard-1t` machine type, provisioning model, resolved image, and required image family recorded.
- [ ] Package installed from a known revision and dependency snapshot retained.
- [ ] CPU tests passed.
- [ ] `devices` proved one TPU device and finite device computation.
- [ ] `loss-smoke` passed its fixed value-and-gradient fixture.
- [ ] `model-smoke` passed full forward compilation and finiteness.
- [ ] `train-smoke` passed the end-to-end overfit/update gate.
- [ ] `benchmark` measured post-compile full training steps.
- [ ] `checkpoint-smoke` proved save/restore/next-step equivalence.
- [ ] Archive copied off the VM and inspected.
- [ ] Exact Compute Engine TPU instance deleted and absence verified; retained disks accounted for.

## References

- [Compute Engine TPU v6e quickstart](https://docs.cloud.google.com/compute/docs/tpus/quickstart-create-tpu-instance)
- [Create a TPU VM with Compute Engine](https://docs.cloud.google.com/tpu/docs/create-instance-compute)
- [Google-managed TPU OS images](https://docs.cloud.google.com/tpu/docs/tpu-os-images)
- [TPU v6e supported configurations](https://docs.cloud.google.com/tpu/docs/v6e)
- [JAX AI Stack on Cloud TPU](https://docs.cloud.google.com/tpu/docs/jax-ai-stack)
- [Install the JAX AI Stack](https://docs.jaxstack.ai/en/latest/install.html)
- [Current JAX TPU installation](https://docs.jax.dev/en/latest/installation.html#google-cloud-tpu)
- [Google JAX AI Docker images](https://docs.cloud.google.com/ai-hypercomputer/docs/images#jax-ai-images)
- [`tpu-info` installation and metrics](https://docs.cloud.google.com/tpu/docs/tpu-info-cli)
- [JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [`gcloud compute scp`](https://docs.cloud.google.com/sdk/gcloud/reference/compute/scp)
