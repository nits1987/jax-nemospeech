# Manual GKE TPU v6e Runbook for Parakeet CPT

**Status:** executable infrastructure plan plus application contract  
**Target:** one GKE Standard cluster, one single-host `ct6e-standard-4t` TPU node, four v6e chips in a `2x2` topology  
**Pilot:** load `nvidia/parakeet-rnnt-1.1b`, use a transcribed two-hour sample, run correctness checks and one smoke-test epoch, checkpoint to Cloud Storage, resume once, and export a serving artifact

This is the manual deployment document for the first Parakeet vertical slice. The reusable engineering product is the **ASR-only** JAX compatibility layer; later ASR models reuse it and pass their own qualification gates.

This runbook is deliberately split into two gates:

1. **Infrastructure gate:** provision GKE, validate all four TPU chips with JAX, and validate Cloud Storage access. These steps are runnable now.
2. **Application gate:** complete the Parakeet first slice of the [NeMo ASR on JAX migration workstream](nemo-asr-jax-migration-plan.md), then run training after its model adapter, weight converter, loss, data loader, checkpoint, and export gates pass.

There is no supported switch that makes the NeMo Parakeet training implementation run on TPU. The Parakeet model and, for supervised training, the RNN-T loss are engineering work in JAX. Kubernetes provisioning alone does not remove that dependency.

## 1. Business-facing delivery plan

| Workstream | Deliverable | Exit gate | Planning range |
|---|---|---|---|
| Cloud foundation | GKE, v6e-4 node pool, Artifact Registry, Cloud Storage, Workload Identity | JAX sees four TPU devices and writes to the bucket | 1-2 engineering days after quota is approved |
| ASR compatibility foundation | Normalized ASR config, support registry, shared audio/model/objective boundaries, conformance harness | Parakeet uses shared interfaces and unsupported configurations fail explicitly | 2-3 engineer-weeks, overlapping model work |
| Model conversion | Flax/JAX FastConformer, predictor and joint network; deterministic safetensors/NeMo weight mapping | Layer-by-layer outputs match the reference model | 3-4 engineer-weeks |
| Training path | XLA RNN-T loss, Optax state, mixed precision, data loader, and matched recipe | Tiny dataset overfits; gradients and loss match the reference | 3-5 engineer-weeks; supervised RNN-T is the high-risk item |
| Reliability | Orbax checkpoint containing parameters, optimizer, step, RNG, and data position | Deliberate stop and resume continues from the saved step | 1-2 engineer-weeks including integration fixes |
| Pilot and handoff | Four-chip qualification, two-hour run, metrics, export, clean-runtime inference smoke | One epoch completes and the exported model transcribes held-out audio | About 1 engineer-week |

For the reusable ASR foundation plus a production-quality supervised Parakeet RNN-T first slice, budget roughly **10-14 engineer-weeks**, or **7-9 calendar weeks with two experienced JAX/ASR engineers**, before calling the pilot repeatable. The workstream ranges overlap and should not be added mechanically. Additional ASR families are separately qualified through the shared layer; this estimate is not for the full NeMo toolkit or the full ASR catalog. TPU quota/capacity lead time is external.

## 2. Decisions to lock before application work

- Confirm whether CPT is supervised RNN-T training or encoder-only self-supervised pretraining. A two-hour supervised pilot requires transcripts.
- Treat one epoch over two hours as a deployment smoke test, not evidence of model improvement. Also run a 20-sample overfit test and at least 200 repeated optimizer steps to expose compilation, memory, and stability defects.
- Use JAX/Flax/Optax for the TPU training runtime and Orbax for checkpoints.
- Keep serving as an export-and-validation gate for the first MVP. Export the trained weights to the Hugging Face/NeMo-compatible layout and validate with its supported runtime. Native JAX serving requires an additional RNN-T decoder, batching, API, and serving-SLO workstream.
- Use a run-specific Cloud Storage prefix. Never share one checkpoint directory between simultaneous jobs.

## 3. Prerequisites

Run all commands in **Google Cloud Shell (bash)**. Do not paste the bash variable blocks directly into local PowerShell.

You need:

- A Google Cloud project with billing enabled.
- Permission to enable APIs, create GKE clusters/node pools, create service accounts, modify IAM, create buckets, and create Artifact Registry repositories.
- Four on-demand v6e chips in the selected region, or a matching reservation. GKE consumes the **Compute Engine** TPU Trillium quota (`tpu_family:CT6E`), not Cloud TPU API quota.
- Capacity in a supported v6e zone. This runbook uses `us-east1-d` as an example; change it to the zone where your quota or reservation exists.
- A GKE version at least `1.31.2-gke.1115000`, the stricter minimum in the current v6e availability table. A current Regular release channel is expected to exceed this, but the explicit check below is still required.

Currently supported v6e zones include `asia-northeast1-b`, `europe-west4-a`, `southamerica-west1-a`, `us-central1-b`, `us-east1-d`, `us-east5-a`, `us-east5-b`, and `us-south1-ai1b`. Re-check the live GKE TPU availability page before provisioning.

### 3.1 Set variables once

Choose a globally unique bucket name and a new image tag for each build.

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export ZONE="us-east1-d"
export REGION="us-east1"

export CLUSTER="parakeet-cpt-poc"
export CPU_POOL="default-pool"
export TPU_POOL="v6e-4-pool"
export NAMESPACE="parakeet-cpt"
export KSA="parakeet-cpt-runner"

export NODE_SA_NAME="parakeet-gke-nodes"
export NODE_SA="${NODE_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export AR_REPO="parakeet-cpt"
export BUCKET="${PROJECT_ID}-parakeet-cpt"

export HF_REVISION="2acc4c61eede2f52ddefe74935de1930a9064d4a"
export SOURCE_MODEL_ID="public-hf-${HF_REVISION}"
export SOURCE_MODEL_URI="gs://${BUCKET}/models/parakeet-rnnt-1.1b/hf/${HF_REVISION}"
export IMAGE_TAG="v0.1.0"
export IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/parakeet-cpt-jax"
export IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
export RUN_ID="smoke-$(date -u +%Y%m%d-%H%M%S)"
export RUNBOOK_ROOT="${HOME}/cpt-rnnt"

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone "${ZONE}"
gcloud config set compute/region "${REGION}"
gcloud auth list
```

Sanity check the values before creating anything:

```bash
printf 'PROJECT_ID=%s\nZONE=%s\nREGION=%s\nCLUSTER=%s\nTPU_POOL=%s\nBUCKET=%s\nSOURCE_MODEL_URI=%s\nIMAGE=%s\nRUN_ID=%s\n' \
  "${PROJECT_ID}" "${ZONE}" "${REGION}" "${CLUSTER}" "${TPU_POOL}" "${BUCKET}" "${SOURCE_MODEL_URI}" "${IMAGE}" "${RUN_ID}"
```

Do not continue if `PROJECT_ID` still says `YOUR_PROJECT_ID`, or if `ZONE` is not inside `REGION`.

### 3.2 Put this runbook checkout in Cloud Shell

Clone your repository or upload this workspace with the Cloud Shell file menu so these files exist below `RUNBOOK_ROOT`. Change the path if you placed it elsewhere.

```bash
cd "${RUNBOOK_ROOT}"
test -f gke-v6e-jax-cpt-runbook.md
test -f deploy/gke/jax-tpu-preflight.template.yaml
test -f deploy/gke/parakeet-cpt-job.template.yaml
command -v envsubst
```

Define this rendering helper once per Cloud Shell session. It rejects missing values, leaves only explicitly whitelisted substitutions, detects unresolved template variables, and runs a client-side Kubernetes validation before apply:

```bash
require_vars() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      printf 'Required variable is unset: %s\n' "${name}" >&2
      return 1
    fi
  done
}

render_manifest() {
  local template="$1"
  local output="$2"
  local whitelist="$3"
  test -f "${template}"
  envsubst "${whitelist}" < "${template}" > "${output}"
  if grep -n '\${[A-Z][A-Z0-9_]*}' "${output}"; then
    printf 'Unresolved template variable in %s\n' "${output}" >&2
    return 1
  fi
  kubectl apply --dry-run=client -f "${output}" >/dev/null
}
```

### 3.3 Enable APIs

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  iamcredentials.googleapis.com \
  storage.googleapis.com \
  tpu.googleapis.com \
  --project="${PROJECT_ID}"
```

Verify:

```bash
gcloud services list --enabled --project="${PROJECT_ID}" \
  --filter='name:(artifactregistry.googleapis.com OR compute.googleapis.com OR container.googleapis.com OR iamcredentials.googleapis.com OR storage.googleapis.com OR tpu.googleapis.com)'
```

### 3.4 Verify zone, version, quota, and capacity inputs

Check that the v6e machine type is advertised in the selected zone:

```bash
gcloud compute machine-types describe ct6e-standard-4t \
  --zone="${ZONE}" \
  --project="${PROJECT_ID}"
```

Inspect the GKE versions offered in the zone:

```bash
gcloud container get-server-config \
  --zone="${ZONE}" \
  --project="${PROJECT_ID}" \
  --format='yaml(channels)'
```

In Google Cloud Console, open **IAM & Admin -> Quotas & System Limits**, choose service `compute.googleapis.com`, and filter the dimensions for `tpu_family:CT6E` and `region:${REGION}`. The available on-demand limit must cover four chips. For Spot, the relevant quota is named `Preemptible TPU slices v6e`. Quota does not guarantee physical capacity; an on-demand node-pool creation can still wait or fail when the zone is full.

## 4. Create storage, registry, and node identity

The create commands intentionally fail on name collisions. If a bucket, repository, service account, or cluster already exists, inspect its project, region, IAM, and ownership before deciding to reuse it; do not turn an `already exists` error into an unconditional success.

### 4.1 Create the Cloud Storage bucket

Keep the bucket in the same region as the TPU node.

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access
```

Create an infrastructure provisioning record. Application launch manifests are created separately for every later `RUN_ID` in section 8, after the model revision and image digest are known:

```bash
printf '{"run_id":"%s","project":"%s","zone":"%s","machine_type":"ct6e-standard-4t","topology":"2x2"}\n' \
  "${RUN_ID}" "${PROJECT_ID}" "${ZONE}" \
  | gcloud storage cp - "gs://${BUCKET}/infra/provisioning-${RUN_ID}.json"
```

### 4.2 Create the Docker repository

```bash
gcloud artifacts repositories create "${AR_REPO}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="JAX Parakeet CPT images"
```

If the command reports that the repository already exists, verify it rather than creating another:

```bash
gcloud artifacts repositories describe "${AR_REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}"
```

### 4.3 Create a least-privilege GKE node service account

This identity is for node system operations and image pulls. It is not the training workload's Cloud Storage identity.

```bash
gcloud iam service-accounts create "${NODE_SA_NAME}" \
  --project="${PROJECT_ID}" \
  --display-name="Parakeet GKE node service account"

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${NODE_SA}" \
  --role="roles/container.defaultNodeServiceAccount"

gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --member="serviceAccount:${NODE_SA}" \
  --role="roles/artifactregistry.reader"
```

The human or automation identity creating the cluster also needs permission to attach `NODE_SA` (`roles/iam.serviceAccountUser`). If cluster creation returns `iam.serviceAccounts.actAs denied`, have an administrator grant that role on this service account to the exact operator identity; do not grant it to all project users.

## 5. Provision GKE Standard and the v6e-4 node

This is a cost-controlled PoC layout: one zonal control plane and one small CPU node for Kubernetes system workloads. Use a regional control plane for production availability.

### 5.1 Create the cluster

```bash
gcloud container clusters create "${CLUSTER}" \
  --project="${PROJECT_ID}" \
  --location="${ZONE}" \
  --release-channel=regular \
  --machine-type=e2-standard-4 \
  --num-nodes=1 \
  --disk-size=50 \
  --service-account="${NODE_SA}" \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --enable-ip-alias
```

Get credentials and confirm the selected control-plane version is at least the v6e minimum:

```bash
gcloud container clusters get-credentials "${CLUSTER}" \
  --project="${PROJECT_ID}" \
  --location="${ZONE}"

gcloud container clusters describe "${CLUSTER}" \
  --project="${PROJECT_ID}" \
  --location="${ZONE}" \
  --format='value(currentMasterVersion,workloadIdentityConfig.workloadPool)'

kubectl cluster-info
kubectl get nodes -o wide
```

Stop here if the version is older than `1.31.2-gke.1115000` or if the workload pool is blank.

### 5.2 Create namespace and Workload Identity principal

```bash
kubectl create namespace "${NAMESPACE}"
kubectl create serviceaccount "${KSA}" --namespace="${NAMESPACE}"

export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
export KSA_PRINCIPAL="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="${KSA_PRINCIPAL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None
```

This uses direct Workload Identity Federation. Do not create or download a JSON service-account key, and do not annotate the Kubernetes ServiceAccount with a Google service-account email for this direct-principal method.

For this short-lived pilot, `roles/storage.objectAdmin` is bucket-wide so the same identity can read inputs and write checkpoints. That identity can also overwrite/delete inputs, so production must split immutable model/data into an input bucket with `roles/storage.objectViewer` and checkpoints/metrics into an output bucket with `roles/storage.objectAdmin` (or use equivalently reviewed conditional IAM). The pilot binding is explicitly removed in section 11.

### 5.3 Validate Cloud Storage access before paying for TPU time

The smoke test runs on the CPU node and writes a marker through the Kubernetes ServiceAccount:

```bash
require_vars NAMESPACE KSA CPU_POOL BUCKET
kubectl delete job gcs-wif-smoke -n "${NAMESPACE}" --ignore-not-found
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/gcs-wif-smoke.template.yaml" \
  /tmp/gcs-wif-smoke.yaml \
  '${NAMESPACE} ${KSA} ${CPU_POOL} ${BUCKET}'
kubectl apply -f /tmp/gcs-wif-smoke.yaml
kubectl wait --for=condition=complete job/gcs-wif-smoke -n "${NAMESPACE}" --timeout=180s
kubectl logs job/gcs-wif-smoke -n "${NAMESPACE}"
gcloud storage cat "gs://${BUCKET}/smoke/wif.txt"
```

Expected: the job completes and both log/output commands show a UTC timestamp. If this fails, fix IAM before provisioning the TPU node.

### 5.4 Create an autoscaling single-host v6e-4 TPU node pool

The pool is configured for zero nodes while idle and a maximum of one node while a TPU Job is pending. Billing for the TPU node starts when capacity is actually provisioned. Scale-down is not instantaneous, so explicit deletion remains the final cost stop.

```bash
gcloud container node-pools create "${TPU_POOL}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}" \
  --node-locations="${ZONE}" \
  --machine-type=ct6e-standard-4t \
  --tpu-topology=2x2 \
  --threads-per-core=1 \
  --enable-autoscaling \
  --num-nodes=0 \
  --total-min-nodes=0 \
  --total-max-nodes=1 \
  --location-policy=ANY \
  --max-pods-per-node=32 \
  --service-account="${NODE_SA}" \
  --workload-metadata=GKE_METADATA
```

The explicit `--tpu-topology=2x2` plus a maximum of one `ct6e-standard-4t` node requests a single-host slice with four chips. `--threads-per-core=1` follows Google's current v6e Standard node-pool examples.

Verify the pool configuration. Before the first TPU Job, zero TPU nodes is the expected result:

```bash
gcloud container node-pools describe "${TPU_POOL}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}" \
  --format='yaml(status,config.machineType,initialNodeCount,autoscaling)'

kubectl get nodes \
  -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology
```

After the preflight Pod reaches Running, expected TPU-node labels include:

```text
cloud.google.com/gke-tpu-accelerator=tpu-v6e-slice
cloud.google.com/gke-tpu-topology=2x2
```

### 5.5 Run the JAX hardware preflight

```bash
require_vars NAMESPACE KSA
kubectl delete job jax-tpu-preflight -n "${NAMESPACE}" --ignore-not-found
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/jax-tpu-preflight.template.yaml" \
  /tmp/jax-tpu-preflight.yaml \
  '${NAMESPACE} ${KSA}'
kubectl apply -f /tmp/jax-tpu-preflight.yaml
kubectl get pods -n "${NAMESPACE}" -w
```

In a second Cloud Shell tab, follow the logs:

```bash
kubectl logs -f job/jax-tpu-preflight -n "${NAMESPACE}"
```

Then verify completion:

```bash
kubectl wait --for=condition=complete job/jax-tpu-preflight \
  -n "${NAMESPACE}" \
  --timeout=20m

kubectl get job jax-tpu-preflight -n "${NAMESPACE}"
```

The gate passes only if the log reports `device_count=4`, all devices use the TPU platform, and the matrix operation completes. `ImagePullBackOff`, a Pending Pod, or CPU devices do not count as success.

If the application image and data are not ready for sections 8-10, delete the TPU pool now rather than leaving cost control to autoscaler timing. Re-run section 5.4 immediately before the application gates:

```bash
gcloud container node-pools delete "${TPU_POOL}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}" \
  --quiet
```

## 6. Prepare model and two-hour data

These steps can run while the JAX application is being implemented.

### 6.1 Stage the public base model

The public repository provides both a `.nemo` file and Hugging Face safetensors. Prefer safetensors plus config/tokenizer files for the JAX converter unless the mapping is being derived from the NeMo archive.

```bash
python -m pip install --user --upgrade huggingface_hub

mkdir -p /tmp/base-parakeet
hf download nvidia/parakeet-rnnt-1.1b \
  model.safetensors \
  config.json \
  generation_config.json \
  processor_config.json \
  tokenizer.json \
  tokenizer_config.json \
  --revision="${HF_REVISION}" \
  --local-dir /tmp/base-parakeet

(
  cd /tmp/base-parakeet
  sha256sum \
    model.safetensors \
    config.json \
    generation_config.json \
    processor_config.json \
    tokenizer.json \
    tokenizer_config.json
) > /tmp/base-parakeet/SHA256SUMS

gcloud storage rsync --recursive /tmp/base-parakeet \
  "${SOURCE_MODEL_URI}"
```

If the source of truth is the client's CPT `.nemo` checkpoint, upload that immutable input under a different prefix, record its checksum, and change both `SOURCE_MODEL_ID` and `SOURCE_MODEL_URI` to that artifact. Do not overwrite the public base-model prefix.

### 6.2 Prepare the sample dataset

For supervised RNN-T, use train and held-out manifests with at least these fields:

```json
{"audio_uri":"gs://BUCKET/data/sample-2h/audio/000001.wav","text":"transcript text","duration_seconds":7.24,"sample_rate_hz":16000}
```

Before upload, validate:

- every audio object is mono 16 kHz or is transformed deterministically by the input pipeline;
- every supervised record has a non-empty transcript;
- duration totals approximately two hours;
- train and held-out IDs do not overlap;
- text normalization and tokenizer are the same as the reference recipe.
- `validation/fixed.wav` and `validation/reference-output.json` contain one immutable inference sample and the pinned reference runtime's expected normalized text/token output.

Upload from the directory that contains `audio/`, `train.jsonl`, `eval.jsonl`, and `validation/`:

```bash
sha256sum \
  ./sample-2h/train.jsonl \
  ./sample-2h/eval.jsonl \
  ./sample-2h/validation/fixed.wav \
  ./sample-2h/validation/reference-output.json \
  > ./sample-2h/SHA256SUMS

gcloud storage rsync --recursive ./sample-2h \
  "gs://${BUCKET}/data/sample-2h"

gcloud storage ls --recursive "gs://${BUCKET}/data/sample-2h/"
```

Individual WAV objects are acceptable for this two-hour smoke test. Full CPT data must be converted to sufficiently large sequential shards (for example ArrayRecord, TFRecord, or WebDataset) and benchmarked independently so TPU time is not lost to object-read latency.

## 7. JAX application contract

This section is the deployment interface, not the migration work itself. The ASR architecture and the detailed first-slice implementation gates are in [NeMo ASR on JAX Migration Plan](nemo-asr-jax-migration-plan.md). GKE Parakeet application work must not begin until migration gates JM0-JM7 pass.

Do not start the one-epoch job until the custom image implements all of the following:

1. `python -m nemo_asr_jax.cli.convert` converts the exact base artifact into a versioned JAX parameter tree and emits a mapping report for every source tensor.
2. `python -m nemo_asr_jax.cli.parity` compares feature extraction and each model block with fixed NeMo/Transformers golden outputs.
3. `python -m nemo_asr_jax.cli.train` accepts the flags used by `deploy/gke/parakeet-cpt-job.template.yaml`.
4. The train state uses bf16 compute with fp32 numerically sensitive operations; it contains parameters, optimizer state, global step, RNG state, and data position.
5. A four-device JAX mesh is explicit and tested. Single-host execution does not require `jax.distributed.initialize()`.
6. Orbax writes and restores `gs://` checkpoints. The program waits for asynchronous checkpoint completion before successful process exit.
7. `--resume=auto` restores the newest complete checkpoint and logs both restored and next global step.
8. SIGTERM handling requests/finalizes a checkpoint within the Pod's 300-second termination grace period.
9. `python -m nemo_asr_jax.cli.export` creates a versioned Hugging Face/NeMo-compatible inference artifact and a conversion report.
10. The image is immutable and pinned: Python, JAX/libtpu, Flax, Optax, Orbax, tokenizer and application commit are recorded in logs and checkpoint metadata.
11. Every invocation writes a non-overwriting resolved launch/resume/export manifest under `gs://${BUCKET}/runs/${RUN_ID}/provenance/`, including source hashes/revision, container digest, code commit, resolved config, data hashes, mesh, seed, start/end step, and timestamps.
12. A separate, digest-pinned serving-validation image implements `python -m serving_validation.validate_parakeet_export` with a pinned supported Transformers/NeMo runtime and no dependency on the training process. It reads the exported directory plus fixed audio/golden output and writes a machine-readable result.

For supervised CPT, the XLA-compatible RNN-T loss must match the reference loss and gradients on small tensors before the TPU pilot. It must avoid materializing an impractical full `[batch, time, label, vocabulary]` tensor at production shapes.

### 7.1 Build and push the custom image

Build after the JAX application's Dockerfile exists. The Dockerfile should inherit from a validated, pinned Cloud TPU JAX image tag or digest; use `latest` only for the disposable hardware preflight. Set `APP_ROOT` to that checkout without leaving the runbook checkout.

```bash
export APP_ROOT="${HOME}/nemo-asr-jax"
test -f "${APP_ROOT}/Dockerfile"

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker build --tag "${IMAGE}" "${APP_ROOT}"
docker push "${IMAGE}"

export IMAGE_DIGEST="$(gcloud artifacts docker images describe "${IMAGE}" \
  --project="${PROJECT_ID}" \
  --format='value(image_summary.digest)')"

test -n "${IMAGE_DIGEST}"
export IMAGE="${IMAGE_REPO}@${IMAGE_DIGEST}"
export CONVERSION_ID="${IMAGE_DIGEST#sha256:}"
export CONVERTED_MODEL_URI="gs://${BUCKET}/models/parakeet-rnnt-1.1b/jax/${SOURCE_MODEL_ID}/${CONVERSION_ID}"
printf 'Pinned training image: %s\n' "${IMAGE}"
printf 'Converted-model destination: %s\n' "${CONVERTED_MODEL_URI}"
```

All application Jobs below receive the digest-pinned `IMAGE` and an explicitly validated converted-model URI. If you open a new shell, resolve and export `IMAGE`, `CONVERSION_ID`, `CONVERTED_MODEL_URI`, and `BASE_MODEL_URI="${CONVERTED_MODEL_URI}"` again; do not fall back to the mutable tag.

Build the independent serving-validation image the same way from its own Dockerfile and resolve it to a digest:

```bash
export SERVING_APP_ROOT="${HOME}/parakeet-serving-validation"
export SERVING_IMAGE_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/parakeet-serving-validation"
export SERVING_IMAGE_TAG="v0.1.0"
export SERVING_IMAGE="${SERVING_IMAGE_REPO}:${SERVING_IMAGE_TAG}"

test -f "${SERVING_APP_ROOT}/Dockerfile"
docker build --tag "${SERVING_IMAGE}" "${SERVING_APP_ROOT}"
docker push "${SERVING_IMAGE}"

export SERVING_IMAGE_DIGEST="$(gcloud artifacts docker images describe "${SERVING_IMAGE}" \
  --project="${PROJECT_ID}" \
  --format='value(image_summary.digest)')"

test -n "${SERVING_IMAGE_DIGEST}"
export SERVING_IMAGE="${SERVING_IMAGE_REPO}@${SERVING_IMAGE_DIGEST}"
printf 'Pinned serving-validation image: %s\n' "${SERVING_IMAGE}"
```

## 8. Run the application gates

The exact model commands depend on the application contract above. The Kubernetes mechanics are fixed.

Do not recreate/provision TPU capacity until the model, data, pinned image, and CPU/GPU parity evidence are ready. Check whether the autoscaling pool exists; if it was deleted after the hardware preflight, re-run section 5.4 now:

```bash
gcloud container node-pools list \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}"
```

Every TPU Job has a server-side `activeDeadlineSeconds`; the client-side `kubectl wait --timeout` alone would not stop a runaway workload. After any failed or abandoned gate, delete its Job and, if no next gate will start immediately, delete the TPU pool using section 11.

Define this helper once per shell and call it after every new `RUN_ID`. It creates a non-overwriting operator launch record; the application must add the resolved provenance required in section 7:

```bash
record_launch() {
  require_vars RUN_ID RUN_MODE IMAGE SOURCE_MODEL_ID SOURCE_MODEL_URI BASE_MODEL_URI BUCKET EPOCHS MAX_STEPS SAVE_INTERVAL_STEPS ACTIVE_DEADLINE_SECONDS
  if [[ "${IMAGE}" != *@sha256:* ]]; then
    printf 'IMAGE must be digest-pinned, got: %s\n' "${IMAGE}" >&2
    return 1
  fi
  local launched_at
  launched_at="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '{"run_id":"%s","mode":"%s","image":"%s","source_model_id":"%s","source_model_uri":"%s","base_model_uri":"%s","train_manifest":"gs://%s/data/sample-2h/train.jsonl","eval_manifest":"gs://%s/data/sample-2h/eval.jsonl","data_hashes":"gs://%s/data/sample-2h/SHA256SUMS","epochs":"%s","max_steps":"%s","save_interval_steps":"%s","active_deadline_seconds":"%s","launched_at":"%s"}\n' \
    "${RUN_ID}" "${RUN_MODE}" "${IMAGE}" "${SOURCE_MODEL_ID}" "${SOURCE_MODEL_URI}" "${BASE_MODEL_URI}" \
    "${BUCKET}" "${BUCKET}" "${BUCKET}" "${EPOCHS}" "${MAX_STEPS}" \
    "${SAVE_INTERVAL_STEPS}" "${ACTIVE_DEADLINE_SECONDS}" "${launched_at}" \
    | gcloud storage cp - "gs://${BUCKET}/runs/${RUN_ID}/provenance/operator-launch-${launched_at}.json"
}
```

### 8.1 Gate A: conversion and numerical parity

Run the converter and parity suite on CPU/GPU first. The TPU is not the place to debug tensor-name mappings. Required evidence:

- 100% of expected source tensors mapped or explicitly documented;
- parameter shapes and total parameter count match;
- feature output and every major block remain within agreed absolute/relative tolerance;
- one fixed audio sample produces equivalent token IDs before any training.

In the pinned reference/application environment, write the converted parameter tree to the image-specific destination and validate that exact tree:

```bash
require_vars SOURCE_MODEL_URI CONVERTED_MODEL_URI

python -m nemo_asr_jax.cli.convert \
  --source-model-uri="${SOURCE_MODEL_URI}" \
  --output-model-uri="${CONVERTED_MODEL_URI}"

python -m nemo_asr_jax.cli.parity \
  --source-model-uri="${SOURCE_MODEL_URI}" \
  --converted-model-uri="${CONVERTED_MODEL_URI}"

gcloud storage ls --recursive "${CONVERTED_MODEL_URI}/"
export BASE_MODEL_URI="${CONVERTED_MODEL_URI}"
```

Do not point the TPU Job directly at the original Hugging Face or `.nemo` directory. `BASE_MODEL_URI` must identify the converted artifact that produced the passing mapping/parity report; the application must reject an incomplete artifact.

### 8.2 Gate B: load the base model and run one TPU forward pass

This is the direct proof that the converted 1.1B checkpoint loads and executes on all four v6e chips. It must run before any optimizer work.

```bash
export RUN_ID="base-load-$(date -u +%Y%m%d-%H%M%S)"
export RUN_MODE="forward-smoke"
export EPOCHS="0"
export MAX_STEPS="0"
export SAVE_INTERVAL_STEPS="0"
export ACTIVE_DEADLINE_SECONDS="1800"
record_launch

require_vars RUN_ID NAMESPACE KSA IMAGE BASE_MODEL_URI BUCKET RUN_MODE EPOCHS MAX_STEPS SAVE_INTERVAL_STEPS ACTIVE_DEADLINE_SECONDS
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/parakeet-cpt-job.template.yaml" \
  "/tmp/parakeet-cpt-${RUN_ID}.yaml" \
  '${RUN_ID} ${NAMESPACE} ${KSA} ${IMAGE} ${BASE_MODEL_URI} ${BUCKET} ${RUN_MODE} ${EPOCHS} ${MAX_STEPS} ${SAVE_INTERVAL_STEPS} ${ACTIVE_DEADLINE_SECONDS}'
kubectl apply -f "/tmp/parakeet-cpt-${RUN_ID}.yaml"

kubectl logs -f "job/parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}" --timestamps
```

Gate: the image restores every expected parameter, logs a four-device mesh, preprocesses one held-out waveform, produces finite logits with the expected shapes, and matches the CPU/GPU golden token sequence within the agreed decoder policy.

### 8.3 Gate C: tiny-overfit run on TPU

Render a new Job name; Kubernetes Job templates are immutable.

```bash
export RUN_ID="tiny-overfit-$(date -u +%Y%m%d-%H%M%S)"
export RUN_MODE="tiny-overfit"
export EPOCHS="1"
export MAX_STEPS="200"
export SAVE_INTERVAL_STEPS="25"
export ACTIVE_DEADLINE_SECONDS="7200"
record_launch

require_vars RUN_ID NAMESPACE KSA IMAGE BASE_MODEL_URI BUCKET RUN_MODE EPOCHS MAX_STEPS SAVE_INTERVAL_STEPS ACTIVE_DEADLINE_SECONDS
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/parakeet-cpt-job.template.yaml" \
  "/tmp/parakeet-cpt-${RUN_ID}.yaml" \
  '${RUN_ID} ${NAMESPACE} ${KSA} ${IMAGE} ${BASE_MODEL_URI} ${BUCKET} ${RUN_MODE} ${EPOCHS} ${MAX_STEPS} ${SAVE_INTERVAL_STEPS} ${ACTIVE_DEADLINE_SECONDS}'
kubectl apply -f "/tmp/parakeet-cpt-${RUN_ID}.yaml"

kubectl logs -f "job/parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}"
```

Gate: a fixed 20-sample subset overfits, loss materially falls, no NaN/Inf occurs, and at least one restorable Orbax checkpoint appears in GCS.

### 8.4 Gate D: requested two-hour, one-epoch smoke run

Create a new run ID and render another Job:

```bash
export RUN_ID="two-hour-$(date -u +%Y%m%d-%H%M%S)"
export RUN_MODE="train"
export EPOCHS="1"
export MAX_STEPS="-1"
export SAVE_INTERVAL_STEPS="25"
export ACTIVE_DEADLINE_SECONDS="21600"
record_launch

require_vars RUN_ID NAMESPACE KSA IMAGE BASE_MODEL_URI BUCKET RUN_MODE EPOCHS MAX_STEPS SAVE_INTERVAL_STEPS ACTIVE_DEADLINE_SECONDS
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/parakeet-cpt-job.template.yaml" \
  "/tmp/parakeet-cpt-${RUN_ID}.yaml" \
  '${RUN_ID} ${NAMESPACE} ${KSA} ${IMAGE} ${BASE_MODEL_URI} ${BUCKET} ${RUN_MODE} ${EPOCHS} ${MAX_STEPS} ${SAVE_INTERVAL_STEPS} ${ACTIVE_DEADLINE_SECONDS}'
kubectl apply -f "/tmp/parakeet-cpt-${RUN_ID}.yaml"

kubectl get pod -n "${NAMESPACE}" -l "job-name=parakeet-cpt-${RUN_ID}" -w
```

Follow logs in another tab:

```bash
kubectl logs -f "job/parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}" --timestamps
```

Inspect scheduling or failures:

```bash
kubectl describe job "parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}"
kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp
```

Wait for completion and inspect outputs:

```bash
kubectl wait --for=condition=complete "job/parakeet-cpt-${RUN_ID}" \
  -n "${NAMESPACE}" \
  --timeout=6h

gcloud storage ls --recursive "gs://${BUCKET}/runs/${RUN_ID}/"
```

The one-epoch gate passes only when:

- all four chips train, not merely enumerate;
- global step increases and finite loss is logged;
- the exact sample count/audio duration consumed is reported;
- final checkpoint and metrics JSON are present;
- the job exits zero only after Orbax has finalized the checkpoint.

## 9. Prove checkpoint resume

Do this before calling the platform reliable.

1. Start the explicit 500-step run below.
2. Wait until at least checkpoint step 50 is finalized in Cloud Storage.
3. Record the step, delete the Job with foreground cascading, prove its old Pod has terminated, and recreate it with the same `RUN_ID` and `--resume=auto`.

```bash
export RUN_ID="resume-test-$(date -u +%Y%m%d-%H%M%S)"
export RUN_MODE="train"
export EPOCHS="1000"
export MAX_STEPS="500"
export SAVE_INTERVAL_STEPS="25"
export ACTIVE_DEADLINE_SECONDS="21600"
record_launch

require_vars RUN_ID NAMESPACE KSA IMAGE BASE_MODEL_URI BUCKET RUN_MODE EPOCHS MAX_STEPS SAVE_INTERVAL_STEPS ACTIVE_DEADLINE_SECONDS
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/parakeet-cpt-job.template.yaml" \
  "/tmp/parakeet-cpt-${RUN_ID}.yaml" \
  '${RUN_ID} ${NAMESPACE} ${KSA} ${IMAGE} ${BASE_MODEL_URI} ${BUCKET} ${RUN_MODE} ${EPOCHS} ${MAX_STEPS} ${SAVE_INTERVAL_STEPS} ${ACTIVE_DEADLINE_SECONDS}'
kubectl apply -f "/tmp/parakeet-cpt-${RUN_ID}.yaml"
kubectl logs -f "job/parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}" --timestamps

# Run these after a complete checkpoint at step 50 or later is visible.
gcloud storage ls --recursive "gs://${BUCKET}/runs/${RUN_ID}/checkpoints/"

kubectl delete job "parakeet-cpt-${RUN_ID}" \
  -n "${NAMESPACE}" \
  --cascade=foreground \
  --wait=true

export OLD_TRAINING_PODS="$(kubectl get pods -n "${NAMESPACE}" -l "job-name=parakeet-cpt-${RUN_ID}" -o name)"
if [[ -n "${OLD_TRAINING_PODS}" ]]; then
  printf 'STOP: old training Pod still exists: %s\n' "${OLD_TRAINING_PODS}" >&2
else
  gcloud storage ls --recursive "gs://${BUCKET}/runs/${RUN_ID}/checkpoints/"

  # Same RUN_ID and rendered manifest; --resume=auto is already in the template.
  record_launch
  kubectl apply -f "/tmp/parakeet-cpt-${RUN_ID}.yaml"
  kubectl logs -f "job/parakeet-cpt-${RUN_ID}" -n "${NAMESPACE}" --timestamps
fi
```

Foreground deletion plus the empty selector prevents the old and resumed jobs from writing the same checkpoint prefix concurrently.

The resume gate passes only if logs identify the same checkpoint URI, restore the recorded global step, continue at the next step without replaying data beyond the documented policy, and create a newer valid checkpoint. A model-only weight load is not a training resume.

## 10. Export and serving handoff

The training output is an Orbax checkpoint, not automatically a NeMo or Transformers serving model. The export job must produce:

- serving weights and config/tokenizer files;
- tensor-mapping and dtype report;
- source run ID and checkpoint step;
- a fixed-audio transcription comparison against the pre-export JAX model;
- hashes for every exported file.

While the TPU pool still exists, set `RUN_ID` to the completed training run (do not create a new ID) and run the exporter:

```bash
export RUN_ID="THE_COMPLETED_TRAINING_RUN_ID"

kubectl delete job "parakeet-export-${RUN_ID}" \
  -n "${NAMESPACE}" \
  --ignore-not-found

require_vars RUN_ID NAMESPACE KSA IMAGE BUCKET
render_manifest \
  "${RUNBOOK_ROOT}/deploy/gke/parakeet-export-job.template.yaml" \
  "/tmp/parakeet-export-${RUN_ID}.yaml" \
  '${RUN_ID} ${NAMESPACE} ${KSA} ${IMAGE} ${BUCKET}'
kubectl apply -f "/tmp/parakeet-export-${RUN_ID}.yaml"

kubectl logs -f "job/parakeet-export-${RUN_ID}" -n "${NAMESPACE}" --timestamps

kubectl wait --for=condition=complete "job/parakeet-export-${RUN_ID}" \
  -n "${NAMESPACE}" \
  --timeout=60m

gcloud storage ls --recursive "gs://${BUCKET}/runs/${RUN_ID}/export/"
```

Do not type the placeholder literally. Now validate from the independent, digest-pinned supported runtime on the CPU node:

```bash
if [[ "${SERVING_IMAGE}" != *@sha256:* ]]; then
  printf 'SERVING_IMAGE must be digest-pinned, got: %s\n' "${SERVING_IMAGE}" >&2
else
  require_vars RUN_ID NAMESPACE KSA CPU_POOL SERVING_IMAGE BUCKET
  kubectl delete job "parakeet-validate-${RUN_ID}" \
    -n "${NAMESPACE}" \
    --ignore-not-found
  render_manifest \
    "${RUNBOOK_ROOT}/deploy/gke/parakeet-export-validation.template.yaml" \
    "/tmp/parakeet-validate-${RUN_ID}.yaml" \
    '${RUN_ID} ${NAMESPACE} ${KSA} ${CPU_POOL} ${SERVING_IMAGE} ${BUCKET}'
  kubectl apply -f "/tmp/parakeet-validate-${RUN_ID}.yaml"
  kubectl logs -f "job/parakeet-validate-${RUN_ID}" -n "${NAMESPACE}" --timestamps
  kubectl wait --for=condition=complete "job/parakeet-validate-${RUN_ID}" \
    -n "${NAMESPACE}" \
    --timeout=30m
  gcloud storage cat "gs://${BUCKET}/runs/${RUN_ID}/export-validation/result.json"
fi
```

The export gate passes only when this clean runtime loads the exported directory, transcribes the fixed audio, matches the golden policy, writes `result.json`, and exits zero.

Recommended MVP boundary:

```text
JAX/Flax training on TPU -> Orbax checkpoint in GCS -> export to safetensors/NeMo -> inference smoke test in supported serving runtime
```

Do not claim native JAX serving from this pilot unless a JAX RNN-T greedy/beam decoder and a serving API have separately passed correctness, concurrency, latency, batching, health-check, and load tests. That is a second deployment plan.

## 11. Cost control and teardown

The completed Kubernetes Job does not itself guarantee that the TPU node stops billing. Delete the TPU node pool immediately after collecting logs and verifying GCS outputs:

```bash
gcloud container node-pools delete "${TPU_POOL}" \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}" \
  --quiet
```

Confirm no TPU node remains:

```bash
gcloud container node-pools list \
  --project="${PROJECT_ID}" \
  --cluster="${CLUSTER}" \
  --location="${ZONE}"

kubectl get nodes
```

After all outputs have been verified, revoke the workload's bucket-wide write access even if the bucket is retained. Recompute the principal so this works in a fresh shell:

```bash
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
export KSA_PRINCIPAL="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA}"

gcloud storage buckets remove-iam-policy-binding "gs://${BUCKET}" \
  --member="${KSA_PRINCIPAL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None
```

If no more tests are planned, remove the workload namespace and delete the cluster as separate, explicit actions:

```bash
kubectl delete namespace "${NAMESPACE}" --wait=true

gcloud container clusters delete "${CLUSTER}" \
  --project="${PROJECT_ID}" \
  --location="${ZONE}" \
  --quiet
```

If this node identity will not be attached to another cluster, remove its repository/project bindings and delete it:

```bash
gcloud artifacts repositories remove-iam-policy-binding "${AR_REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --member="serviceAccount:${NODE_SA}" \
  --role="roles/artifactregistry.reader"

gcloud projects remove-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${NODE_SA}" \
  --role="roles/container.defaultNodeServiceAccount"

gcloud iam service-accounts delete "${NODE_SA}" \
  --project="${PROJECT_ID}" \
  --quiet
```

The bucket and Artifact Registry repository are intentionally retained because they contain evidence, checkpoints, and images. Review them manually before any deletion. Storage continues to incur a smaller charge.

## 12. Troubleshooting map

| Symptom | Most likely check |
|---|---|
| Node-pool create says quota exceeded | Verify Compute Engine `tpu_family:CT6E` quota in the TPU region, not Cloud TPU API quota |
| Node-pool stays provisioning | Zone capacity/reservation availability; try only an approved zone with quota, or use a reservation/flex-start plan |
| Pod Pending with `Insufficient google.com/tpu` | Node pool status, exact accelerator/topology selectors, and request/limit both equal to 4 |
| Pod Pending with node affinity mismatch | Confirm generated labels are `tpu-v6e-slice` and `2x2` with `kubectl get nodes -L ...` |
| `ImagePullBackOff` | Image name/digest and `roles/artifactregistry.reader` on the repository for `NODE_SA` |
| JAX shows CPU devices | TPU-compatible JAX/libtpu image, GKE version, TPU resource request, and node placement |
| GCS returns 403 | Correct namespace/KSA in `serviceAccountName`, Workload Identity pool, exact principal URI, and bucket-scoped role |
| First step takes minutes | Expected XLA compilation once; log compile time separately and require later steps to stabilize |
| TPU idle while host busy | Profile audio decoding, feature extraction, object reads, padding shapes, and prefetch depth |
| OOM on TPU | Reduce per-chip batch, rematerialize encoder blocks, shard optimizer state, and inspect RNN-T joint/loss materialization |
| Job exits but checkpoint is incomplete | Wait for Orbax async finalization and make process exit conditional on a successful final checkpoint |
| Recreated Job starts at step zero | Fix checkpoint discovery/metadata; verify optimizer, step, RNG, and data position are restored, not only parameters |

## 13. Evidence package for the go/no-go review

Retain these artifacts under `gs://${BUCKET}/runs/${RUN_ID}/`:

- resolved config, code commit, container digest, and dependency versions;
- source checkpoint hash and full tensor mapping report;
- data manifest hashes, sample count, and total audio duration;
- parity and tiny-overfit reports;
- training metrics including compile time, steady-state step time, examples/audio-hours per second, memory, and loss;
- checkpoint list plus deliberate-resume proof;
- export hashes and fixed-audio inference comparison;
- actual TPU wall time and cost estimate.

The decision is **go** only after correctness, checkpoint-resume, and export/load gates pass. A fast one-epoch run without those artifacts proves hardware access, not a viable CPT platform.

## 14. Current official references

- [Plan TPUs in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)
- [Deploy TPU workloads in GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus)
- [TPU v6e architecture and configurations](https://docs.cloud.google.com/tpu/docs/v6e)
- [Workload Identity Federation for GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)
- [Configure GKE node service accounts](https://docs.cloud.google.com/kubernetes-engine/security/configure-node-service-accounts)
- [Artifact Registry Docker authentication](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication)
- [Orbax checkpointing](https://orbax.readthedocs.io/en/stable/guides/checkpoint/orbax_checkpoint_101.html)
- [NVIDIA Parakeet RNN-T 1.1B model files](https://huggingface.co/nvidia/parakeet-rnnt-1.1b/tree/main)
