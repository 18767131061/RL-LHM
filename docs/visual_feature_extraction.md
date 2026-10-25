# Fixed visual features

`scripts/extract_qwen_visual_latents.py` supplies an executable, image-only
reference implementation of the 1152-dimensional `extra_info.visual_z` cache
interface. It extracts once from each seed's **local SFT checkpoint** and keeps
the resulting vectors fixed during RL. The versioned extraction recipe below defines the cache contents, checkpoint
requirements and extraction settings.

## Definition

The official Qwen3-VL-4B-Instruct configuration has visual hidden width **1024**
and merged language-facing width **2560**. Neither width is 1152. Reference v1
therefore defines its dimension conversion explicitly:

1. Supply three local phase images in **arterial, portal-venous, delayed** order.
   The script preserves this order. Optional `phase` attributes must spell
   `arterial`, `portal_venous`, and `delayed`, respectively. It does not infer
   phases from labels, filenames or a language prompt.
2. Use the image processor saved with the checkpoint, RGB input, the record's
   `max_pixels` (default 262144), and the processor's remaining resize/rescale/
   normalization settings.
3. Run only the checkpoint's visual tower in evaluation/inference mode with
   eager attention. Capture the final visual block output **before the final
   patch merger and its normalization**, excluding all DeepStack merged outputs.
   Pool the raw tokens of each phase by their arithmetic mean in float32, giving
   three 1024-dimensional phase vectors. Per-phase boundaries come from each
   `image_grid_thw`, before spatial merging.
4. Multiply each phase vector by the same fixed 1024-by-384 matrix `P`. Concatenate
   the resulting vectors in the above phase order and L2-normalize the complete
   1152-dimensional vector in float32. This is not an average of the phases and
   not three independently unit-normalized vectors. Zero/nonfinite vectors fail.

For phase token matrix `H_j` with `N_j` tokens, the implementation is
`u_j = mean_tokens(H_j)`, `v = concat(u_AP P, u_PVP P, u_DP P)`, and
`z = v / ||v||_2`. No labels, targets, bounding boxes, generated completions,
language-model tokens, validation outcomes, or fitted data-dependent projection
parameters enter this computation. The source row's targets are copied to the output JSONL for subsequent training.

## Fixed projection

The bundled `models/assets/qwen3vl_projection_1024x384_v1.npy` is a fixed
projection matrix. It was generated once by NumPy's
`Generator(PCG64(20261012)).standard_normal((1024, 384), dtype=float64)`, reduced
QR, and a per-column sign correction making the corresponding diagonal of `R`
nonnegative (zero uses positive sign). `Q` was then serialized as little-endian
float32 in NumPy `.npy` format, with pickling disabled. Its columns are orthonormal
up to float32 rounding.

The canonical file's SHA-256 is
`76d1919c689850c70e0a7f2d481c71a7ee06d3587f27d434f716abfba9a7082b`.
The extractor checks and loads these exact serialized coefficients. It never
regenerates the QR matrix at runtime, avoiding platform/BLAS-dependent QR changes.
Pooling/projection/normalization use float32 even when vision inference uses
bfloat16. Reference defaults are `--device cuda --dtype bfloat16`; CPU validation
requires `--device cpu --dtype float32`. Float16 is explicit opt-in. Record hardware and dtype with each extraction run.

## Local execution

Install `requirements-optional.txt` in a prepared environment. Reference v1 pins
`transformers==4.57.1`; no internet or model-hub download is used at extraction.
The model directory must contain a **complete merged** Qwen3-VL SFT checkpoint,
`config.json`, `preprocessor_config.json`, and local `.safetensors` shards (or
`pytorch_model*.bin` files). Only `model.visual.*` parameters are loaded; the large
language-model weights are not allocated. All visual parameters must load strictly.
Adapter-only/PEFT directories are rejected with instructions to merge with the
exact locally available base checkpoint. The script does not silently substitute
a base model or ignore an SFT adapter.

```bash
python scripts/extract_qwen_visual_latents.py \
  --model-dir /path/to/models/seed_42 \
  --input-jsonl /path/to/prepared/real961_rl_train.jsonl \
  --output-jsonl /path/to/rl_data/seed_42_visual_z.jsonl \
  --device cuda --dtype bfloat16
```

Repeat for the five SFT seeds, changing both checkpoint and output path. Each row
requires a distinct nonempty `extra_info.sample_uid` and exactly three local
images. Relative image paths resolve against the input JSONL directory, unless
`--image-root` is supplied. The converter's absolute image paths work unchanged.
The output preserves every record and adds `visual_z`, `visual_z_contract` and
`visual_z_cache_id` to `extra_info`, including its reward-model copy.

The launcher consumes this output as `--train-file`; grid `seed_train_files`
points to the corresponding five caches and `seed_model_paths` to their SFT
checkpoints. The desktop scheduler invokes the same CLI and validates an existing
cache before reusing it. A formal grid validates each reference cache against its
configured seed checkpoint; renaming another seed's cache is not accepted.
Existing caller-supplied feature files without the reference contract remain
supported by the general cache interface and are not relabeled as reference-v1.

## Cache integrity and interruption recovery

The `.audit.json` sidecar binds the complete input JSONL, every image byte hash,
checkpoint configuration/processor/weight hashes, checkpoint path, extraction
source, phase order, serialized projection, dtype/device and versions of Torch,
Transformers, NumPy, Pillow and safetensors. Each completed vector is validated for its
1152 dimensions, finite entries and unit norm; output rows and target preservation
are checked on reuse. No patient's image bytes appear in the audit sidecar.

Concurrent writers are excluded by an OS file lock. An interrupted run resumes
only the exact same identity and validates every complete partial row. A torn
last line is discarded; changed inputs/checkpoints/runtime or malformed complete
rows fail without overwriting them. The complete output is atomically renamed
only after successful extraction and a final input/checkpoint/image identity
check. If output or sidecar conflicts, choose a new output path; there is no
silent overwrite option. Validation alone needs no GPU/model loading:

```bash
python scripts/extract_qwen_visual_latents.py \
  --model-dir /path/to/models/seed_42 \
  --input-jsonl /path/to/prepared/real961_rl_train.jsonl \
  --output-jsonl /path/to/rl_data/seed_42_visual_z.jsonl \
  --device cuda --dtype bfloat16 --validate-only
```

## Extraction

Run the extractor once for each seed checkpoint and retain the resulting cache
with the corresponding model configuration.

## Upstream definitions

- [Official Qwen3-VL-4B-Instruct configuration](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [Transformers v4.57.1 Qwen3-VL visual model](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)
- [Transformers v4.57.1 Qwen2-VL image processor reused by Qwen3-VL](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)
