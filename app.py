from __future__ import annotations # 支持类型注解中的自引用

import argparse # 用于解析命令行参数
import json #处理示例元数据
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr #Gradio: 快速构建机器学习Web应用的库
import torch
from PIL import Image

from pipeline_chord import ChordEditPipeline #导入核心编辑管道
from utils import DEFAULT_DATA_ROOT # 默认数据集根目录


LOGGER = logging.getLogger("chord_app")


# Model root and component layout.
# 模型组件子目录映射
# 定义了每个模型组件在模型根目录下的子文件夹名称
COMPONENT_SUBDIRS: Dict[str, str] = {   
    "unet_path": "unet",   # UNet：扩散模型的核心去噪网络，负责预测噪声
    "scheduler_path": "scheduler",   # 调度器配置 定义如何添加/去除噪声的时间表
    "text_encoder_path": "text_encoder",   # 文本编码器
    "tokenizer_path": "tokenizer",   # 分词器：将文本拆分为token ID
    "vae_path": "vae",   # 变分自编码器
}
# 默认模型目录
DEFAULT_MODEL_ROOT = "/sd-turbo"

# 根据默认根目录构建完整的组件路径
DEFAULT_COMPONENT_PATHS: Dict[str, str] = {
    key: str(Path(DEFAULT_MODEL_ROOT) / subdir) for key, subdir in COMPONENT_SUBDIRS.items()
}

# 默认编辑配置参数
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,      # 噪声样本数（用于蒙特卡洛估计）
    "n_steps": 1,            # 编辑迭代步数
    "t_start": 0.90,         # 起始时间步 [0,1] (1=纯噪声,0=干净图像)，数值越大（接近1.0）表示越靠近噪声端（编辑空间大）
    "t_end": 0.30,           # 结束时间步 [0, t_start]，数值越小表示越接近原始图像（保真度高）
    "t_delta": 0.15,         # 时间差分，用于计算编辑方向
    "step_scale": 1.0,       # 步长缩放因子，控制沿 dv 方向（目标与源的残差）移动的幅度。
    "cleanup": True,         # 是否执行清理步骤
}

DEFAULT_SEED = 42                         # 默认随机种子
DEFAULT_PRECISION = "fp32"                 # 默认计算精度
DEFAULT_IMAGE_SIZE = 512                   # 默认图像尺寸
DEFAULT_MAX_EXAMPLES = 24                   # 最大示例数量
DEFAULT_SERVER_NAME = "127.0.0.1"           # 服务器地址（本地）
DEFAULT_SERVER_PORT = 7860                   # 服务器端口
DEFAULT_CENTER_CROP = True                   # 默认使用中心裁剪
DEFAULT_USE_ATTENTION_MASK = False           # 默认不使用注意力掩码
DEFAULT_USE_SAFETY_CHECKER = False           # 默认不使用安全检查器
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================================
# CSS样式定义
# ============================================================================

SQUARE_PREVIEW_CSS = """
/* 源图像输入容器：强制正方形显示 */
#source-image-input {
    width: 100% !important;
}

/* 图像容器,保持1:1的宽高比 */
#source-image-input .image-container,
#source-image-input [data-testid="image"] {
    aspect-ratio: 1 / 1 !important;
    overflow: hidden !important;
    background: #00000008 !important;
}

#source-image-input img,
#source-image-input canvas {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    object-position: center center !important;
    display: block !important;
    background: transparent !important;
}

#editor-main-row {
    align-items: center !important;
}

#source-prompt textarea,
#target-prompt textarea {
    height: 112px !important;
    max-height: 112px !important;
    overflow-y: auto !important;
    resize: none !important;
}

.panel-note p {
    margin: 0 0 12px 0 !important;
    line-height: 1.4 !important;
    font-size: 0.95rem !important;
    color: #666 !important;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch ChordEdit web app.")
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument("--server-port", type=int, default=DEFAULT_SERVER_PORT, help="Web server port.")
    return parser.parse_args()


def _dtype_from_precision(value: Optional[str]) -> torch.dtype:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def _paths_from_model_root(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def _expand_paths(path_map: Dict[str, str | None]) -> Dict[str, str]:
    expanded: Dict[str, str] = {}
    missing: List[str] = []
    for key in COMPONENT_SUBDIRS:
        value = path_map.get(key)
        final_value = value if value is not None else DEFAULT_COMPONENT_PATHS.get(key)
        if final_value is None:
            missing.append(key)
            continue
        expanded[key] = str(Path(final_value).expanduser().resolve())
    if missing:
        raise ValueError(
            f"Missing required component paths for: {missing}. "
            "Set --model-root or provide per-component paths."
        )
    return expanded


def _resolve_component_paths(model_root: str | Path) -> Dict[str, str]:
    return _expand_paths(_paths_from_model_root(model_root))


def _select_image_file(folder: Path) -> Path:
    """从文件夹中选择一个图像文件
    
    优先级:
        1. 文件名是 i, image, original 的文件（不区分大小写）
        2. 按文件名排序的第一个图像文件
    """
    # 找出所有图像文件
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No RGB image found inside {folder}")

    # 查找优先文件（i, image, original）
    preferred = sorted(
        (p for p in candidates if p.stem.lower() in {"i", "image", "original"}),
        key=lambda p: p.name,
    )
    if preferred:
        return preferred[0]
    return sorted(candidates, key=lambda p: p.name)[0]


def load_examples(dataset_root: Path, max_examples: Optional[int]) -> List[List[Any]]:
    examples: List[List[Any]] = []
    if not dataset_root.exists():
        LOGGER.warning("Example dataset does not exist: %s", dataset_root)
        return examples

    for subdir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue

        try:
            image_path = _select_image_file(subdir)
        except FileNotFoundError:
            LOGGER.warning("No image found in %s", subdir)
            continue

        with meta_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping invalid JSON in %s:%d (%s)", meta_file, line_number, exc)
                    continue

                src_prompt = str(record.get("original_prompt", "")).strip()
                tgt_prompt = str(record.get("edited_prompt", "")).strip()
                examples.append([str(image_path), src_prompt, tgt_prompt])

                if max_examples is not None and len(examples) >= max_examples:
                    return examples

    return examples


def _validate_inputs(
    image: Optional[Image.Image],
    source_prompt: str,
    target_prompt: str,
    t_start: float,  # 编辑强度起点：扩散模型去噪的时间步。数值越大，对原图的改动潜力越大
    t_end: float,  # 编辑强度终点：停止去噪的时间步。数值越小，细节还原越真实
    t_delta: float,
) -> None:
    if image is None:
        raise gr.Error("Please upload a source image first.")
    if not source_prompt or not source_prompt.strip():
        raise gr.Error("Please provide the source image prompt.")
    if not target_prompt or not target_prompt.strip():
        raise gr.Error("Please provide the target image prompt.")
    if t_start <= t_end: #必须从噪声多的地方走向噪声少的地方
        raise gr.Error("Invalid parameters: t_start must be greater than t_end.")
    if t_delta < 0: #时间差不能为负
        raise gr.Error("Invalid parameters: t_delta must be greater than or equal to 0.")


def build_demo(
    pipeline: ChordEditPipeline,
    default_seed: int,
    default_edit_config: Dict[str, Any],
    examples: List[List[Any]],
) -> gr.Blocks:
    #run_edit :将网页上用户的点击操作和输入，转换成 Python 能够处理的变量和数值
    def run_edit(
        image: Optional[Image.Image],  #网页上传的图片
        source_prompt: str,      #用户输入的“原图描述”
        target_prompt: str,      #用户输入的“目标描述”
        seed: float,             #随机种子，控制生成的随机性
        n_samples: float,
        t_start: float,
        t_end: float,
        t_delta: float,
        step_scale: float,
    ) -> Image.Image:
        #1.验证输入：防止用户没传图就点运行
        _validate_inputs(image, source_prompt, target_prompt, t_start, t_end, t_delta)

        seed_int = int(seed)
        #2. 组装配置字典：把网页滑块的值打包好，将UI参数传递给模型
        edit_config = {
            "noise_samples": int(n_samples),
            "n_steps": int(default_edit_config.get("n_steps", 1)),
            "t_start": float(t_start),
            "t_end": float(t_end),
            "t_delta": float(t_delta),
            "step_scale": float(step_scale),
            "cleanup": bool(default_edit_config.get("cleanup", True)),
        }

        try:
        #3. 调用 pipeline：这一步会跳到 pipeline_chord.py 执行
            result = pipeline(
                image=image,
                source_prompt=source_prompt.strip(),
                target_prompt=target_prompt.strip(),
                edit_config=edit_config,
                seed=seed_int,
            )
        except Exception as exc:
            LOGGER.exception("Editing failed.")
            raise gr.Error(f"Editing failed: {exc}") from exc
        #处理输出（模型返回的是列表，取第一张）
        images = result.images
        if not isinstance(images, list):
            raise gr.Error("The pipeline did not return PIL images. Please check output_type.")
        if not images:
            raise gr.Error("The pipeline returned no output image.")
        return images[0] #返回处理后的第一张图片到网页显示

    #demo 是整个网页的根节点。SQUARE_PREVIEW_CSS 负责网页的样式装修
    with gr.Blocks(title="ChordEdit App", css=SQUARE_PREVIEW_CSS) as demo: #创建整个网页的大框架
        gr.Markdown("# ChordEdit App") #在网页顶部写大标题
        gr.Markdown(
            'To study artifacts and background leakage of the one-step editor without Chord Control, set `t_delta` to `0`.\n'
            'Images shown in the paper are available in the "Examples" list below.',
            elem_classes=["panel-note"],
        )

        with gr.Row(elem_id="editor-main-row"):  #创建主交互行
            with gr.Column(scale=5, elem_id="left-input-panel"): # 左边栏，宽度比例为5
                with gr.Group(): #将下面的组件打包在一起，外观上会有个边框
                    with gr.Row():
                        with gr.Column(scale=1, min_width=280):
                            input_image = gr.Image(
                                type="pil",
                                label="Source Image",
                                sources=["upload", "clipboard"], # 允许上传或剪贴板粘贴
                                elem_id="source-image-input",
                                height=320,  #固定显示高度
                            )
                        with gr.Column(scale=1, min_width=280):
                            source_prompt = gr.Textbox(
                                label="Source Prompt",
                                lines=4,
                                max_lines=4,
                                placeholder="Example: A cat on a sofa",
                                elem_id="source-prompt",
                            )
                            target_prompt = gr.Textbox(
                                label="Target Prompt",
                                lines=4,
                                max_lines=4,
                                placeholder="Example: A cat wearing sunglasses",
                                elem_id="target-prompt",
                            )

                gr.Markdown("### Parameters")
                n_samples_default = int(
                    default_edit_config.get("n_samples", default_edit_config.get("noise_samples", 1))
                )
                with gr.Row(): # 开启一个“行”容器。里面缩进的东西都会横着排
                    # seed_input: 随机数种子输入框。precision=0 表示只能输入整数
                    seed_input = gr.Number(label="Seed", value=int(default_seed), precision=0)

                    # n_samples_input: 采样数量滑块。物理含义：一次生成几张图供参考
                    n_samples_input = gr.Slider(  #gr.Slider:调节阀
                        label="n_samples",
                        minimum=1,
                        maximum=16,
                        step=1,  #每次拖动的最小间距
                        value=n_samples_default,
                    )
                    # step_scale_input: 编辑步长缩放。物理含义：每一刀切多深（修改幅度）
                    step_scale_input = gr.Slider(
                        label="step_scale",
                        minimum=0.1,
                        maximum=5.0,
                        step=0.1,
                        value=float(default_edit_config.get("step_scale", 1.0)),
                    )
                with gr.Row():
                    t_start_input = gr.Slider(
                        label="t_start",
                        minimum=0.01,
                        maximum=1.0,
                        step=0.01,
                        value=float(default_edit_config.get("t_start", 0.90)), 
                    )
                    t_end_input = gr.Slider(
                        label="t_end",
                        minimum=0.0,
                        maximum=0.99,
                        step=0.01,
                        value=float(default_edit_config.get("t_end", 0.30)),
                    )
                    t_delta_input = gr.Slider(
                        label="t_delta",
                        minimum=0.0,
                        maximum=0.5,
                        step=0.01,
                        value=float(default_edit_config.get("t_delta", 0.15)),
                    )

                run_button = gr.Button("Run Edit", variant="primary")

            with gr.Column(scale=5, elem_id="right-output-panel"):
                with gr.Group():
                    output_image = gr.Image(
                        type="pil",
                        label="Editing Result",
                        elem_id="result-image-output",
                        height=440,
                    )

        run_inputs = [
            input_image,
            source_prompt,
            target_prompt,
            seed_input,
            n_samples_input,
            t_start_input,
            t_end_input,
            t_delta_input,
            step_scale_input,
        ]

        run_button.click(fn=run_edit, inputs=run_inputs, outputs=output_image)
        target_prompt.submit(fn=run_edit, inputs=run_inputs, outputs=output_image)

        if examples:
            gr.Markdown("## Examples")
            gr.Examples(
                examples=examples,
                inputs=[input_image, source_prompt, target_prompt],
                label="Click an example to auto-fill the left-side inputs.",
            )
        else:
            gr.Markdown("## Examples")
            gr.Markdown("No valid examples were found under the current dataset path.")

    return demo


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    component_paths = _resolve_component_paths(model_root=args.model_root)
    edit_config = dict(DEFAULT_EDIT_CONFIG)
    seed = DEFAULT_SEED
    torch_dtype = _dtype_from_precision(DEFAULT_PRECISION)
    compute_dtype = torch.float32

    dataset_root = DEFAULT_DATA_ROOT
    examples = load_examples(dataset_root=dataset_root, max_examples=DEFAULT_MAX_EXAMPLES)

    LOGGER.info("Loaded %d example records from %s", len(examples), dataset_root)
    LOGGER.info("Seed: %s | Default edit config: %s", seed, edit_config)
    LOGGER.info("Component paths: %s", component_paths)
# 模型加载，创建ChordEditPipeline实例
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=edit_config,
        device=None,                     # None表示自动选择GPU（如果有）
        torch_dtype=torch_dtype,
        image_size=DEFAULT_IMAGE_SIZE,      # 输入图像尺寸（512x512）
        use_center_crop=DEFAULT_CENTER_CROP,    # 是否中心裁剪
        compute_dtype=compute_dtype,
        use_attention_mask=DEFAULT_USE_ATTENTION_MASK,
        use_safety_checker=DEFAULT_USE_SAFETY_CHECKER,
    )

    demo = build_demo(
        pipeline=pipeline,
        default_seed=seed,
        default_edit_config=edit_config,
        examples=examples,
    )

    demo.queue(api_open=False)
    demo.launch(
        server_name=DEFAULT_SERVER_NAME,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
