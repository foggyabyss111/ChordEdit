from __future__ import annotations # 支持类型注解中的自引用（如类名作为类型)

import argparse      # 命令行参数解析库
import json          # JSON文件处理
import logging       # 日志记录
import os
import shutil        # 文件复制等高级文件操作
from dataclasses import dataclass  # 数据类装饰器
from pathlib import Path            # 跨平台路径处理
from typing import Any, Dict, List, Optional  # 类型注解

import torch
from PIL import Image

from pipeline_chord import ChordEditPipeline # 导入核心编辑管道
from utils import first_param_point, load_yaml_config # 工具函数


LOGGER = logging.getLogger("pie_bench") #获取日志记录器

# 模型组件的子目录映射
# key: pipeline_chord.py中期望的路径键名
# value: 相对于模型根目录的子文件夹名称
# model root + expected component subdirectories
COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",  #uNet 模型权重
    "scheduler_path": "scheduler", #调度器配置
    "text_encoder_path": "text_encoder", #文本编码器
    "tokenizer_path": "tokenizer", #分词器
    "vae_path": "vae", #VAE 模型
}
def _default_model_root() -> str:
    project_dir = Path(__file__).resolve().parent
    env_root = os.getenv("CHORDEDIT_MODEL_ROOT")
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            project_dir / "sd-turbo",
            project_dir.parent / "sd-turbo",
            Path("/sd-turbo"),
            Path("D:/sd-turbo"),
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_dir():
            return str(resolved)
    return str((project_dir / "sd-turbo").resolve())


DEFAULT_MODEL_ROOT = _default_model_root()
DEFAULT_COMPONENT_PATHS: Dict[str, str] = {
    key: str(Path(DEFAULT_MODEL_ROOT) / subdir) for key, subdir in COMPONENT_SUBDIRS.items()
}

DEFAULT_EDIT_CONFIG = {
    "noise_samples": 1,  #噪声样本数（用于蒙特卡洛估计）
    "n_steps": 1,
    "t_start": 0.90, #起始时间步 [0,1] (1=纯噪声,0=干净图像)
    "t_end": 0.30,
    "t_delta": 0.15, #时间差分，用于计算编辑方向
    "step_scale": 1.0, #步长缩放因子，控制编辑强度
    "cleanup": True, #是否在编辑后执行清理步骤
}

DEFAULT_SEED = 42  # 默认随机种子
DEFAULT_PRECISION = "fp32"   # 默认计算精度

# PIE-Bench数据集相关默认路径
DEFAULT_PIE_ROOT = Path(__file__).resolve().parent / "pie_bench"  #PIE 数据集根目录
DEFAULT_MAPPING_FILE = "mapping_file.json"      #映射文件名
DEFAULT_IMAGE_SUBDIR = "annotation_images"      #图像子目录名
DEFAULT_METHOD_NAME = "ChordEdit"                 #方法名称（用于输出目录）


@dataclass(frozen=True) # frozen=True 使实例不可变，类似只读对象
class PieRecord:
    sample_id: str      #样本唯一ID
    image_path: Path
    relative_path: Path     #相对于数据集根目录的路径（用于保持目录结构）
    original_prompt: str    #原始图像描述（源文本）
    edited_prompt: str      #编辑后图像描述（目标文本）
    edit_instruction: str      #编辑指令


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChordEdit on PIE-Bench data and export PIE-format results.")
    # === 配置相关参数 ===
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config describing edit params.")
    # === 模型相关参数 ===
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None, help="Computation precision.")
    # === 编辑参数（可覆盖配置文件的设置）===
    parser.add_argument("--seed", type=int, default=None, help="Random seed overriding the config file.")
    parser.add_argument("--noise-samples", type=int, default=None, help="Number of MC noise samples.")
    parser.add_argument("--n-steps", type=int, default=None, help="Number of Chord iterations.")
    parser.add_argument("--t-start", type=float, default=None, help="Edit timestep start.")
    parser.add_argument("--t-end", type=float, default=None, help="Edit timestep end.")
    parser.add_argument("--t-delta", type=float, default=None, help="Edit timestep delta.")
    parser.add_argument("--step-scale", type=float, default=None, help="Edit update magnitude.")
    parser.add_argument("--cleanup", action="store_true", help="Force cleanup on.")
    parser.add_argument("--no-cleanup", action="store_true", help="Force cleanup off.")
    # cleanup参数
    parser.add_argument(
        "--center-crop",
        dest="center_crop",
        action="store_true",
        default=True,
        help="Center-crop before resize for VAE preprocessing (default).",
    )
    # === 预处理参数 ===
    parser.add_argument(
        "--no-center-crop",
        dest="center_crop",
        action="store_false",
        help="Disable center crop before resizing.",
    )
    parser.add_argument(
        "--use-attention-mask",
        action="store_true",
        help="Pass attention masks to the text encoder (defaults off to mirror chord/src).",
    )
    # === 安全检查参数 ===
    parser.add_argument(
        "--safety-checker",
        dest="use_safety_checker",
        action="store_true",
        help="Enable StableDiffusion safety checker before exporting images.",
    )
    parser.add_argument(
        "--no-safety-checker",
        dest="use_safety_checker",
        action="store_false",
        default=False,
        help="Disable safety checker (default).",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Resolution used when feeding the VAE.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only process the first N records.")

    # === PIE数据集路径参数 ===
    parser.add_argument("--pie-root", type=str, default=None, help="Root directory of PIE-Bench data.")
    parser.add_argument(
        "--mapping-file",
        type=str,
        default=DEFAULT_MAPPING_FILE,
        help="Mapping file path relative to --pie-root.",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory (inside --pie-root) containing the original PIE annotation images.",
    )
    parser.add_argument(
        "--export-root",
        type=str,
        default=None,
        help="Directory that follows PIE-Bench layout (data/... + output/...). Defaults to --pie-root.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=DEFAULT_METHOD_NAME,
        help="Name used under export_root/output/<method_name>/annotation_images.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory inside output/<method_name>/ for generated images.",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default=DEFAULT_IMAGE_SUBDIR,
        help="Subdirectory inside data/ where original images are copied when --copy-source is set.",
    )
    parser.add_argument("--copy-source", action="store_true", help="Copy the source PIE images into export_root/data.")
    parser.add_argument(
        "--mapping-dest",
        type=str,
        default="data/mapping_file.json",
        help="Relative path (from export_root) to write the mapping file.",
    )
    parser.add_argument(
        "--no-sync-mapping",
        action="store_true",
        help="Skip copying the PIE mapping file into export_root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions when present.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Progress logging interval in number of saved samples (0 disables incremental logs).",
    )

    return parser.parse_args()


def dtype_from_precision(value: Optional[str]) -> torch.dtype:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def expand_component_paths(path_map: Dict[str, Optional[str]]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    for key in COMPONENT_SUBDIRS:   #遍历所以必需的组件
        value = path_map.get(key)
        fallback = DEFAULT_COMPONENT_PATHS.get(key) #默认路径
        final_value = value if value is not None else fallback
        if final_value is None:
            raise ValueError(f"Missing required path for '{key}'. Provide via config or CLI.")
        # 扩展用户目录(~)并转换为绝对路径
        expanded[key] = str(Path(final_value).expanduser().resolve())
    return expanded


def paths_from_model_root(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def validate_component_paths(component_paths: Dict[str, str], model_root: str | Path) -> None:
    missing = [f"{key}={value}" for key, value in component_paths.items() if not Path(value).is_dir()]
    if not missing:
        return
    required = ", ".join(COMPONENT_SUBDIRS.values())
    checked_root = str(Path(model_root).expanduser().resolve())
    missing_text = "; ".join(missing)
    raise FileNotFoundError(
        f"Model components not found. model_root={checked_root}. "
        f"Required subfolders: {required}. Missing: {missing_text}. "
        "Please download https://huggingface.co/stabilityai/sd-turbo and run: "
        "python run_pie_bench.py --model-root <path-to-sd-turbo> --pie-root <path-to-pie_bench>"
    )


def load_pipeline_config(path: Optional[str]) -> tuple[Dict[str, Any], int, Optional[str]]:
    if path is None:
        return (dict(DEFAULT_EDIT_CONFIG), DEFAULT_SEED, DEFAULT_PRECISION)

    cfg = load_yaml_config(path)        # 加载YAML
    editor_cfg = cfg.get("editor", {})      # 获取editor部分
    seed_value = editor_cfg.get("seed")
    if seed_value is None:
        seed_list = editor_cfg.get("seed_list")
        if isinstance(seed_list, (list, tuple)) and seed_list:
            seed_value = seed_list[0]
        elif seed_list is not None:
            seed_value = seed_list
    seed_value = int(seed_value) if seed_value is not None else DEFAULT_SEED
    precision = editor_cfg.get("precision", DEFAULT_PRECISION)

    params_grid = editor_cfg.get("params_grid", {})
    edit_config = first_param_point(params_grid) if params_grid else dict(DEFAULT_EDIT_CONFIG)

    return edit_config, seed_value, precision


def apply_cli_overrides(args: argparse.Namespace, edit_config: Dict[str, Any], seed: Optional[int]) -> tuple[Dict[str, Any], int]:
    overrides = {
        "noise_samples": args.noise_samples,
        "n_steps": args.n_steps,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "t_delta": args.t_delta,
        "step_scale": args.step_scale,
    }
    for key, value in overrides.items():
        if value is not None:
            edit_config[key] = value

    if args.cleanup:
        edit_config["cleanup"] = True
    elif args.no_cleanup:
        edit_config["cleanup"] = False

    cli_seed = args.seed
    seed_value = seed if cli_seed is None else cli_seed
    if seed_value is None:
        seed_value = DEFAULT_SEED
    return edit_config, int(seed_value)


def resolve_path(base: Path, maybe_relative: str | Path) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate.expanduser().resolve()
    return (base / candidate).expanduser().resolve()


def load_pie_records(root: Path, mapping_path: Path, image_subdir: str) -> List[PieRecord]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"PIE mapping file not found: {mapping_path}")

    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict):
        raise ValueError(f"Expected mapping JSON to be a dict, got {type(mapping).__name__}")

    img_root = (root / image_subdir).expanduser().resolve()
    if not img_root.exists():
        raise FileNotFoundError(f"PIE image directory does not exist: {img_root}")

    records: List[PieRecord] = []
    for sample_id in sorted(mapping.keys()):
        meta = mapping[sample_id]
        rel_value = meta.get("image_path")
        if rel_value is None:
            LOGGER.warning("Sample %s is missing 'image_path'; skipping.", sample_id)
            continue
        rel_path = Path(rel_value)
        abs_path = (img_root / rel_path).expanduser().resolve()
        if not abs_path.exists():
            LOGGER.warning("Sample %s image not found at %s; skipping.", sample_id, abs_path)
            continue

        original_prompt = meta.get("original_prompt") or meta.get("source_prompt") or ""
        edited_prompt = meta.get("editing_prompt") or meta.get("edited_prompt") or meta.get("target_prompt") or ""
        edit_instruction = meta.get("editing_instruction") or meta.get("edit_prompt") or edited_prompt

        records.append(
            PieRecord(
                sample_id=sample_id,
                image_path=abs_path,
                relative_path=rel_path,
                original_prompt=original_prompt,
                edited_prompt=edited_prompt,
                edit_instruction=edit_instruction,
            )
        )

    if not records:
        raise FileNotFoundError(f"No valid PIE records found in {mapping_path}.")
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    ensure_dir(dst.parent)
    if overwrite or not dst.exists():
        shutil.copy2(src, dst)


def sync_mapping_file(mapping_path: Path, export_root: Path, dest_relative: str, *, overwrite: bool) -> None:
    dest_path = resolve_path(export_root, dest_relative)
    copy_file(mapping_path, dest_path, overwrite=overwrite)


def save_prediction(image: Image.Image, destination: Path, *, overwrite: bool) -> None:
    ensure_dir(destination.parent)
    if overwrite or not destination.exists():
        image.save(destination)

# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    # === 1. 解析命令行参数 ===
    args = parse_args()
    # === 2. 配置日志 ===
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # === 3. 加载配置 ===
    #从YAML文件加载编辑配置、种子和精度
    edit_config, seed, precision = load_pipeline_config(args.config)
    # 用命令行参数覆盖配置
    edit_config, seed = apply_cli_overrides(args, edit_config, seed)
    # 构建组件路径
    component_paths = expand_component_paths(paths_from_model_root(args.model_root))
    validate_component_paths(component_paths, model_root=args.model_root)

    #=== 4. 处理精度 ===
    precision_choice_raw = args.precision or precision or DEFAULT_PRECISION
    precision_choice = precision_choice_raw.lower()
    if precision_choice != "fp32":
        LOGGER.warning(
            "Precision '%s' requested, but PIE export forces fp32 for numerical stability.",
            precision_choice_raw,
        )
        precision_choice = "fp32" # PIE导出强制使用fp32保证数值稳定性
    torch_dtype = dtype_from_precision(precision_choice) # 模型权重精度
    compute_dtype = torch.float32  # 计算精度（固定为fp32）
    # === 5. 解析PIE数据集路径 ===
    pie_root = Path(args.pie_root).expanduser().resolve() if args.pie_root else DEFAULT_PIE_ROOT
    export_root = Path(args.export_root).expanduser().resolve() if args.export_root else pie_root
    mapping_path = resolve_path(pie_root, args.mapping_file)
    # === 6. 加载PIE记录 ===
    records = load_pie_records(pie_root, mapping_path, args.image_subdir)
    if args.max_samples is not None:
        records = records[: args.max_samples] # 只处理前N个样本

    if not records:
        LOGGER.error("No PIE records to process. Check dataset paths.")
        return

    LOGGER.info(
        "Loaded %d PIE samples from %s (mapping=%s)",
        len(records),
        pie_root,
        mapping_path,
    )
    LOGGER.info("Seed %s | Edit config %s", seed, edit_config)
    # === 7. 初始化管道 ===
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=args.device,
        torch_dtype=torch_dtype,
        image_size=args.image_size,
        use_center_crop=args.center_crop,
        compute_dtype=compute_dtype,
        use_attention_mask=args.use_attention_mask,
        use_safety_checker=args.use_safety_checker,
    )
    # === 8. 准备输出目录 ===
    # 输出目录: export_root/output/method_name/output_subdir/
    output_dir = export_root / "output" / args.method_name / args.output_subdir
    # 源图像目录: export_root/data/source_subdir/
    source_dir = export_root / "data" / args.source_subdir
    ensure_dir(output_dir)
    if args.copy_source:
        ensure_dir(source_dir)

    # === 9. 同步映射文件（如果需要） ===
    if not args.no_sync_mapping:
        sync_mapping_file(mapping_path, export_root, args.mapping_dest, overwrite=args.overwrite)
        LOGGER.info("Synchronized mapping file to %s", resolve_path(export_root, args.mapping_dest))
    
    # === 10. 处理所有样本 ===
    processed = 0
    skipped = 0

    for idx, record in enumerate(records, start=1):
        rel_output_path = output_dir / record.relative_path
        if rel_output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        # === 10.1 加载原始图像 ===
        try:
            with Image.open(record.image_path) as img:
                source_image = img.convert("RGB") #确保是RGB 格式
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Failed to read %s: %s", record.image_path, exc)
            skipped += 1
            continue
        # === 10.2 运行编辑管道 ===
        try:
            result = pipeline(
                image=source_image,
                source_prompt=record.original_prompt,   #源描述
                target_prompt=record.edited_prompt, #目标描述
                seed=seed,
                output_type="pil",      #输出PIL格式
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.error("Pipeline failed on %s: %s", record.sample_id, exc)
            skipped += 1
            continue
        # === 10.3 提取生成的图像 ===
        images = result.images
        if isinstance(images, list) and images:
            generated = images[0]
        elif torch.is_tensor(images):
            # 如果输出是tensor，转换为PIL
            generated = pipeline._tensor_to_pil(images)[0]  # type: ignore[attr-defined]
        else:
            LOGGER.warning("No images returned for sample %s; skipping.", record.sample_id)
            skipped += 1
            continue
        # === 10.4 保存结果 ===
        save_prediction(generated, rel_output_path, overwrite=args.overwrite)
        # === 10.5 复制源图像（如果需要） ===
        if args.copy_source:
            target_source_path = source_dir / record.relative_path
            copy_file(record.image_path, target_source_path, overwrite=args.overwrite)

        processed += 1
        if args.log_every and processed % args.log_every == 0:
            LOGGER.info("Saved %d/%d samples (skipped=%d)", processed, len(records), skipped)

    LOGGER.info(
        "Finished PIE export. Saved %d sample(s), skipped %d (existing/errors). Results: %s",
        processed,
        skipped,
        output_dir,
    )


if __name__ == "__main__":
    main()
