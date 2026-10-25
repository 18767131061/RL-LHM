import gzip
import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except Exception:
    torch = None


# The server used private defaults. Public consumers must provide equivalent
# locations explicitly through environment variables or their own CLI layer.
PRIVATE_DATASET_ROOT = os.environ.get("RL_LHM_PRIVATE_DATASET_ROOT", "")
PRIVATE_TRAIN_JSON = os.environ.get("RL_LHM_PRIVATE_TRAIN_JSON", "")
PRIVATE_VAL_JSON = os.environ.get("RL_LHM_PRIVATE_VAL_JSON", "")

MCT_DATASET_ROOT = os.environ.get("RL_LHM_DATASET_ROOT", "")
MCT_TRAIN_JSON_3CLS = os.environ.get("RL_LHM_TRAIN_JSON_3CLS", "")
MCT_VAL_JSON_3CLS = os.environ.get("RL_LHM_VAL_JSON_3CLS", "")
MCT_TRAIN_JSON_5CLS = os.environ.get("RL_LHM_TRAIN_JSON_5CLS", "")
MCT_VAL_JSON_5CLS = os.environ.get("RL_LHM_VAL_JSON_5CLS", "")

PRIVATE_OUTPUT_ROOT = os.environ.get("RL_LHM_PRIVATE_OUTPUT_ROOT", "")
MCT_OUTPUT_ROOT = os.environ.get("RL_LHM_OUTPUT_ROOT", "")
PROJECT_ROOT = Path(os.environ.get("RL_LHM_PROJECT_ROOT", "."))

DEFAULT_GLOBAL_BATCH = 32
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SAVE_TOTAL_LIMIT = 2


def infer_bf16() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def effective_framework_gpus(default_gpus: int) -> int:
    env_world = int(os.environ.get("WORLD_SIZE", "0") or "0")
    if env_world > 0:
        return env_world
    if torch is not None and torch.cuda.is_available():
        try:
            return max(1, int(torch.cuda.device_count()))
        except Exception:
            pass
    return max(1, int(default_gpus))


def compute_grad_accum(global_batch: int, per_device_batch: int, gpu_count: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (global_batch, per_device_batch, gpu_count)):
        raise ValueError("Batch sizes and data-parallel GPU count must be positive integers")
    denom = per_device_batch * gpu_count
    if global_batch % denom:
        raise ValueError("global_batch must be divisible by per_device_batch * gpu_count")
    return global_batch // denom


def sanitize_name(value: str) -> str:
    value = str(value).strip().replace("\\", "/").rstrip("/")
    value = value.split("/")[-1] if "/" in value else value
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "model"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return
        except Exception:
            pass
    path.write_text(text, encoding="utf-8")


def append_or_write_json(path: Path, key: str, value: Dict[str, Any]) -> None:
    obj = {}
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
    obj[key] = value
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    s = str(v).replace("\\", "\\\\").replace("\"", "\\\"")
    return f"\"{s}\""


def render_yaml_lines(items: List[Tuple[str, Any]]) -> str:
    return "\n".join(f"{k}: {yaml_scalar(v)}" for k, v in items) + "\n"


def load_json_array(path: str) -> List[Dict[str, Any]]:
    p = str(path).lower()
    open_fn = gzip.open if p.endswith(".gz") else open
    is_jsonl = p.endswith(".jsonl") or p.endswith(".ndjson") or (p.endswith(".gz") and (".jsonl" in p or ".ndjson" in p))
    with open_fn(path, "rt", encoding="utf-8") as f:
        if is_jsonl:
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
    return data


def resolve_data_paths(data_mode: str) -> Tuple[str, str, str]:
    if data_mode == "private":
        return PRIVATE_DATASET_ROOT, PRIVATE_TRAIN_JSON, PRIVATE_VAL_JSON
    if data_mode == "3cls":
        return MCT_DATASET_ROOT, MCT_TRAIN_JSON_3CLS, MCT_VAL_JSON_3CLS
    if data_mode == "5cls":
        return MCT_DATASET_ROOT, MCT_TRAIN_JSON_5CLS, MCT_VAL_JSON_5CLS
    raise ValueError(f"Unsupported data_mode: {data_mode}")


def content_segments_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for seg in content:
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type")
        if seg_type == "image":
            parts.append("<image>")
        elif seg_type == "text":
            txt = str(seg.get("text", "")).strip()
            if txt:
                parts.append(txt)
    return "\n".join(parts).strip()


def split_system_user_answer(messages: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    system_parts: List[str] = []
    user_parts: List[str] = []
    assistant_parts: List[str] = []
    for m in messages:
        role = m.get("role") or m.get("from")
        txt = content_segments_to_text(m.get("content", ""))
        if not txt:
            continue
        if role == "system":
            system_parts.append(txt)
        elif role in ("user", "human"):
            user_parts.append(txt)
        elif role == "assistant":
            assistant_parts.append(txt)
    return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip(), "\n\n".join(assistant_parts).strip()


def extract_images_from_user(messages: List[Dict[str, Any]], dataset_root: str) -> List[str]:
    imgs: List[str] = []
    for m in messages:
        role = m.get("role") or m.get("from")
        if role not in ("user", "human"):
            continue
        content = m.get("content", [])
        if not isinstance(content, list):
            continue
        for seg in content:
            if not isinstance(seg, dict):
                continue
            if seg.get("type") != "image" or "image" not in seg:
                continue
            img_path = str(seg["image"]).replace("\\", "/")
            if not os.path.isabs(img_path):
                img_path = str(Path(dataset_root) / img_path)
            imgs.append(img_path)
    return imgs


def export_llamafactory_sft_dataset(
    json_path: str,
    dataset_name: str,
    out_dir: Path,
    dataset_root: str,
    min_images: int = 3,
    user_text_transform: Optional[Callable[[str, Dict[str, Any], List[str]], str]] = None,
    answer_text_transform: Optional[Callable[[str, Dict[str, Any], List[str]], str]] = None,
) -> Tuple[Path, Path, int]:
    raw = load_json_array(json_path)
    records: List[Dict[str, Any]] = []
    if not isinstance(raw, list) or not raw:
        raise ValueError("SFT source must contain a nonempty list of records")
    for index, obj in enumerate(raw):
        msgs = obj.get("messages") or obj.get("conversations")
        if not msgs:
            raise ValueError(f"SFT record {index} is missing messages")
        system_text, user_text, answer_text = split_system_user_answer(msgs)
        images = extract_images_from_user(msgs, dataset_root)
        if not user_text or not answer_text or len(images) != min_images:
            raise ValueError(f"SFT record {index} needs a prompt, answer and exactly {min_images} images")
        if not all(Path(image).is_file() for image in images):
            raise ValueError(f"SFT record {index} has missing image files")
        sample_images = images
        if user_text_transform is not None:
            user_text = user_text_transform(user_text, obj, sample_images)
        if answer_text_transform is not None:
            answer_text = answer_text_transform(answer_text, obj, sample_images)
        if not user_text or not answer_text:
            raise ValueError(f"SFT transform removed the prompt or answer for record {index}")
        chat_messages = []
        if system_text:
            chat_messages.append({"role": "system", "content": system_text})
        chat_messages.append({"role": "user", "content": user_text})
        chat_messages.append({"role": "assistant", "content": answer_text})
        records.append({"messages": chat_messages, "images": sample_images})

    dataset_file = out_dir / f"{dataset_name}.json"
    write_text(dataset_file, json.dumps(records, ensure_ascii=False, indent=2))

    dataset_info = out_dir / "dataset_info.json"
    append_or_write_json(
        dataset_info,
        dataset_name,
        {
            "file_name": dataset_file.name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    )
    return dataset_file, dataset_info, len(records)


def build_llamafactory_sft_config(
    *,
    dataset_dir: Path,
    dataset_name: str,
    model_name_or_path: str,
    output_dir: str,
    template: str,
    gpu_count: int,
    per_device_batch: int,
    global_batch: int,
    cutoff_len: int,
    learning_rate: float,
    num_train_epochs: float,
    preprocessing_num_workers: int,
    overwrite_cache: bool,
    lr_scheduler_type: str,
    warmup_ratio: float,
    optim: str = "adamw_torch",
    torch_empty_cache_steps: int = 1,
    finetuning_type: str = "full",
    lora_rank: int = 0,
    lora_alpha: int = 0,
    lora_dropout: float = 0.0,
    lora_target: str = "",
    additional_target: str = "",
    use_rslora: bool = False,
    seed: int = 2026,
    data_seed: int | None = None,
    precision: str | None = None,
) -> str:
    if precision not in {None, "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16 or fp32")
    grad_acc = compute_grad_accum(global_batch, per_device_batch, gpu_count)
    lines = [
        ("model_name_or_path", model_name_or_path),
        ("trust_remote_code", True),
        ("stage", "sft"),
        ("do_train", True),
        ("finetuning_type", finetuning_type),
        ("template", template),
        ("dataset_dir", str(dataset_dir)),
        ("dataset", dataset_name),
        ("cutoff_len", cutoff_len),
        ("overwrite_cache", overwrite_cache),
        ("preprocessing_num_workers", preprocessing_num_workers),
        ("output_dir", output_dir),
        ("logging_steps", DEFAULT_LOGGING_STEPS),
        ("save_strategy", "epoch"),
        ("save_total_limit", DEFAULT_SAVE_TOTAL_LIMIT),
        ("overwrite_output_dir", False),
        ("seed", int(seed)),
        ("data_seed", int(seed if data_seed is None else data_seed)),
        ("plot_loss", True),
        ("per_device_train_batch_size", per_device_batch),
        ("gradient_accumulation_steps", grad_acc),
        ("learning_rate", learning_rate),
        ("num_train_epochs", float(num_train_epochs)),
        ("lr_scheduler_type", lr_scheduler_type),
        ("warmup_ratio", warmup_ratio),
        ("optim", optim),
        ("bf16", infer_bf16() if precision is None else precision == "bf16"),
        ("fp16", precision == "fp16"),
        ("gradient_checkpointing", True),
        # Multimodal custom models using trust_remote_code can conflict with reentrant checkpointing.
        # Use non-reentrant checkpointing to avoid repeated-backward graph errors.
        ("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        # Single-GPU full-parameter training is more prone to memory fragmentation and peaks;
        # periodic empty_cache calls trade a small slowdown for a safer first step.
        ("torch_empty_cache_steps", int(torch_empty_cache_steps)),
        ("remove_unused_columns", False),
        ("report_to", "none"),
    ]
    if finetuning_type == "lora":
        lines.extend([
            ("lora_rank", int(lora_rank)),
            ("lora_alpha", int(lora_alpha)),
            ("lora_dropout", float(lora_dropout)),
            ("use_rslora", bool(use_rslora)),
        ])
        if lora_target:
            lines.append(("lora_target", lora_target))
        if additional_target:
            lines.append(("additional_target", additional_target))
    return render_yaml_lines(lines)


def build_llamafactory_command(config_path: Path, gpu_count: int | None = None) -> str:
    prefix = ""
    if gpu_count is not None:
        if type(gpu_count) is not int or gpu_count <= 0:
            raise ValueError("gpu_count must be a positive integer")
        prefix = f"FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE={gpu_count} "
    return f"{prefix}llamafactory-cli train {shlex.quote(str(config_path))}"


def build_eval_command(data_mode: str, model_family: str, model_dir: str, out_csv: Path,
                       *, json_path: str, data_root: str, manifest: str, group_map: str,
                       seed: int = 2026, model_name: str = "SFT") -> str:
    root = Path(__file__).resolve().parents[1]
    entry = "infer_qwen_real961.py" if model_family in {"qwen", "qwen3vl", "qwen3_vl"} else "infer_hf_vlm_revision.py"
    predictions = out_csv.with_suffix(".jsonl")
    commands = [
        ["python3", str(root / "inference" / entry), "--model-dir", model_dir,
         "--json-path", json_path, "--data-root", data_root, "--output", str(out_csv),
         "--seed", str(seed), "--max-new-tokens", "256", "--decoding", "greedy"],
        ["python3", str(root / "evaluation/import_inference_predictions.py"), "--csv", str(out_csv),
         "--manifest", manifest, "--group-map", group_map, "--output", str(predictions)],
        ["python3", str(root / "evaluation/evaluate_paper_protocol.py"), "--manifest", manifest,
         "--predictions", str(predictions), "--output-dir", str(out_csv.parent / "metrics"),
         "--model-name", model_name, "--model-seed", str(seed)],
    ]
    return "\n".join(shlex.join(command) for command in commands)


def default_output_root(data_mode: str) -> str:
    return PRIVATE_OUTPUT_ROOT if data_mode == "private" else MCT_OUTPUT_ROOT


def default_framework_root(data_mode: str) -> str:
    return f"{default_output_root(data_mode)}/framework_artifacts"


def default_sft_output_dir(family_tag: str, data_mode: str, output_root: Optional[str] = None) -> str:
    root = output_root or default_output_root(data_mode)
    return str(Path(root) / f"{family_tag}_det_{data_mode}_sft")


def framework_artifact_root(family_tag: str, data_mode: str, exp_name: str, override_root: str = "") -> Path:
    root = Path(override_root) if override_root else Path(default_framework_root(data_mode))
    return root / family_tag / data_mode / exp_name


def maybe_launch_framework(command: str, tool_name: str) -> int:
    print(f"[framework] launching {tool_name}:")
    print(command)
    return subprocess.call(command, shell=True)


def prepare_sft_framework_artifacts(
    *,
    family_tag: str,
    model_family: str,
    data_mode: str,
    exp_name: str,
    model_name_or_path: str,
    llamafactory_template: str,
    sft_output_dir: str,
    dataset_root: str,
    train_json: str,
    val_json: str,
    framework_root: str,
    framework_gpus: int,
    per_device_batch: int,
    global_batch: int,
    cutoff_len: int,
    learning_rate: float,
    num_train_epochs: float,
    preprocessing_num_workers: int,
    overwrite_cache: bool,
    min_images: int,
    lr_scheduler_type: str = "constant",
    warmup_ratio: float = 0.0,
    optim: str = "adamw_torch",
    torch_empty_cache_steps: int = 1,
    finetuning_type: str = "full",
    lora_rank: int = 0,
    lora_alpha: int = 0,
    lora_dropout: float = 0.0,
    lora_target: str = "",
    additional_target: str = "",
    use_rslora: bool = False,
    seed: int = 2026,
    data_seed: int | None = None,
    eval_json: str = "",
    eval_manifest: str = "",
    eval_group_map: str = "",
    precision: str | None = None,
    user_text_transform: Optional[Callable[[str, Dict[str, Any], List[str]], str]] = None,
    answer_text_transform: Optional[Callable[[str, Dict[str, Any], List[str]], str]] = None,
) -> Dict[str, Any]:
    if any((eval_json, eval_manifest, eval_group_map)) and not all((eval_json, eval_manifest, eval_group_map)):
        raise ValueError("Evaluation requires eval_json, eval_manifest and eval_group_map together")
    root = framework_artifact_root(family_tag, data_mode, exp_name, framework_root)
    sft_data_dir = root / "llamafactory_data"
    cfg_dir = root / "configs"
    cmd_dir = root / "commands"
    eval_dir = root / "eval"
    for d in (sft_data_dir, cfg_dir, cmd_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_name = f"{data_mode}_sft_train"
    val_name = f"{data_mode}_sft_val"
    train_file, dataset_info_file, train_n = export_llamafactory_sft_dataset(
        train_json,
        train_name,
        sft_data_dir,
        dataset_root,
        min_images=min_images,
        user_text_transform=user_text_transform,
        answer_text_transform=answer_text_transform,
    )
    val_file, _, val_n = export_llamafactory_sft_dataset(
        val_json,
        val_name,
        sft_data_dir,
        dataset_root,
        min_images=min_images,
        user_text_transform=user_text_transform,
        answer_text_transform=answer_text_transform,
    )

    sft_cfg = cfg_dir / "llamafactory_sft.yaml"
    write_text(
        sft_cfg,
        build_llamafactory_sft_config(
            dataset_dir=sft_data_dir,
            dataset_name=train_name,
            model_name_or_path=model_name_or_path,
            output_dir=sft_output_dir,
            template=llamafactory_template,
            gpu_count=framework_gpus,
            per_device_batch=per_device_batch,
            global_batch=global_batch,
            cutoff_len=cutoff_len,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            preprocessing_num_workers=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optim=optim,
            torch_empty_cache_steps=torch_empty_cache_steps,
            finetuning_type=finetuning_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target=lora_target,
            additional_target=additional_target,
            use_rslora=use_rslora,
            seed=seed,
            data_seed=data_seed,
            precision=precision,
        ),
    )

    sft_cmd = build_llamafactory_command(sft_cfg, framework_gpus)
    eval_csv = eval_dir / "sft.csv"
    eval_cmd = (build_eval_command(data_mode, model_family, sft_output_dir, eval_csv,
                                  json_path=eval_json, data_root=dataset_root, manifest=eval_manifest,
                                  group_map=eval_group_map, seed=seed, model_name=family_tag + " SFT")
                if eval_json else "")

    sft_cmd_file = cmd_dir / "run_sft.sh"
    eval_cmd_file = cmd_dir / "run_eval_sft.sh"
    write_text(sft_cmd_file, f"#!/usr/bin/env bash\nset -e\n{sft_cmd}\n")
    if eval_cmd:
        write_text(eval_cmd_file, f"#!/usr/bin/env bash\nset -euo pipefail\n{eval_cmd}\n")
    os.chmod(sft_cmd_file, 0o755)
    if eval_cmd:
        os.chmod(eval_cmd_file, 0o755)

    readme = root / "README.md"
    readme_lines = [
        f"# {family_tag} SFT Framework Artifacts",
        "",
        f"- data_mode: `{data_mode}`",
        f"- exp_name: `{exp_name}`",
        f"- model_family: `{model_family}`",
        f"- model_name_or_path: `{model_name_or_path}`",
        f"- template: `{llamafactory_template}`",
        f"- finetuning_type: `{finetuning_type}`",
        f"- learning_rate: `{learning_rate}`",
        f"- num_train_epochs: `{num_train_epochs}`",
        f"- per_device_batch: `{per_device_batch}`",
        f"- global_batch: `{global_batch}`",
        f"- cutoff_len: `{cutoff_len}`",
        f"- seed: `{seed}`; data_seed: `{seed if data_seed is None else data_seed}`",
        f"- evaluation prepared: `{bool(eval_cmd)}` (requires explicit source/group mapping)",
        f"- optim: `{optim}`",
        f"- train_json: `{train_json}`",
        f"- val_json: `{val_json}`",
        f"- sft_output_dir: `{sft_output_dir}`",
    ]
    if finetuning_type == "lora":
        readme_lines.extend([
            f"- lora_rank: `{lora_rank}`",
            f"- lora_alpha: `{lora_alpha}`",
            f"- lora_dropout: `{lora_dropout}`",
            f"- lora_target: `{lora_target}`",
            f"- additional_target: `{additional_target}`",
            f"- use_rslora: `{use_rslora}`",
        ])
    readme_lines.extend([
        "",
        "## Commands",
        "",
        "```bash",
        sft_cmd,
        "",
        eval_cmd,
        "```",
    ])
    write_text(readme, "\n".join(readme_lines) + "\n")

    return {
        "root": root,
        "dataset_info": dataset_info_file,
        "sft_train_file": train_file,
        "sft_val_file": val_file,
        "sft_cfg": sft_cfg,
        "sft_cmd": sft_cmd,
        "sft_cmd_file": sft_cmd_file,
        "eval_cmd": eval_cmd,
        "eval_cmd_file": eval_cmd_file if eval_cmd else None,
        "eval_csv": eval_csv,
        "readme": readme,
        "counts": {
            "sft_train": train_n,
            "sft_val": val_n,
        },
    }
