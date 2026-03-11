from __future__ import annotations # 支持类型注解中的自引用

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml # YAML文件解析库
from PIL import Image, ImageOps # PIL图像处理
from torch.utils.data import Dataset # PyTorch数据集基类

# 默认数据集根目录（位于当前文件的上级目录的images文件夹)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "images"
# 支持的图像文件扩展名（小写
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a python dictionary.
    从YAML文件加载配置,返回字典
    参数:
        path: YAML文件的路径
    返回:
        配置字典
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def first_param_point(params_grid: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    """Select the first value from each parameter list for a quick default run."""

    def _pick(value: Sequence[Any] | Any) -> Any:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # 如果是序列且不是字符串/字节串，取第一个元素
            if not value:
                raise ValueError("Param grid contains an empty list; cannot determine default.")
            return value[0]
        return value # 不是序列，直接返回

    return {key: _pick(values) for key, values in params_grid.items()}

# ============================================================================
# 数据集数据结构
# ============================================================================

@dataclass(frozen=True) #frozen=True 使实例不可变，类似只读对象
class EditRecord:
    image_path: Path
    src_prompt: str
    tgt_prompt: str
    edit_prompt: str #编辑指令（通常同tgt_prompt）
    edit_id: Optional[str] = None #edit_id: 可选的编辑ID，用于标识不同的编辑变体


class LocalEditDataset(Dataset):
    """Simple dataset mirroring src/utils/mydataset.py for local demos.
        本地编辑数据集，用于加载和预处理本地图像数据
        继承自PyTorch的Dataset,可以与DataLoader配合使用
    """

    def __init__(self, records: List[EditRecord], image_size: int = 512, use_center_crop: bool = False) -> None:
        """初始化数据集
        
        参数:
            records: EditRecord列表,包含所有样本的元数据
            image_size: 目标图像大小（正方形边长）
            use_center_crop: 是否使用中心裁剪
        """

        if not records:
            raise ValueError("No records found in the dataset root.")
        self._records = records
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)

    def __len__(self) -> int:  # type: ignore[override]
        """返回数据集中的样本数量"""
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        record = self._records[idx]
        image = Image.open(record.image_path).convert("RGB") #确保RGB格式
        # 中心裁剪（如果需要）
        if self._use_center_crop:
            image = _center_square_crop(image)
        # 缩放到目标尺寸
        image = _resize_image(image, (self.image_size, self.image_size))
        # === 2. 创建空白图像（占位符）===
        # 白色背景，用于保持与原始数据集相同的输出格式
        blank = Image.new("RGB", image.size, color=(255, 255, 255))
        return {
            "id": record.edit_id or Path(record.image_path).stem, #id:样本ID
            "original_image": image, #original_image: 预处理后的原始图像
            "edited_image": blank, #edited_image: 空白图像（占位符）
            "original_prompt": record.src_prompt, 
            "edited_prompt": record.tgt_prompt,
            "edit_prompt": record.edit_prompt, #edit_prompt: 编辑指令
            "image_path": str(record.image_path), #图像路径
        }


def load_local_dataset(
    path: str | Path | None = None,
    image_size: int = 512,
    center_crop: bool = True,
) -> LocalEditDataset:
    root = _resolve_dataset_root(path)  #解析数据集根目录
    records = _parse_edit_records(root)  #解析所有编辑记录
    return LocalEditDataset(records=records, image_size=image_size, use_center_crop=center_crop)


def _resolve_dataset_root(path: str | Path | None) -> Path:
    if path is not None:
        root = Path(path).expanduser().resolve()  #展开~并转换为绝对路径
    else:
        root = DEFAULT_DATA_ROOT #使用默认路径
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def _parse_edit_records(root: Path) -> List[EditRecord]:
    records: List[EditRecord] = []
    # 遍历所有子目录（按名称排序保证顺序稳定）
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue  # 没有meta.jsonl的目录跳过
        try:
            # 在子目录中查找图像文件
            image_path = _select_image_file(subdir)
        except FileNotFoundError:
            continue # 没有找到图像文件的目录跳过
        
        # 读取meta.jsonl文件
        with meta_file.open("r", encoding="utf-8") as handle:
            for line_num, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)   # 解析JSON
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {meta_file} at line {line_num}: {exc}") from exc

                records.append(
                    EditRecord(
                        image_path=image_path,
                        src_prompt=record.get("original_prompt", ""),
                        tgt_prompt=record.get("edited_prompt", ""),
                        edit_prompt=record.get("edit_prompt", record.get("edited_prompt", "")),
                        edit_id=record.get("edit_id"),
                    )
                )

    if not records:
        raise FileNotFoundError(
            f"No edit samples found under {root}. Expected subdirectories with 'meta.jsonl' files."
        )
    return records


def _select_image_file(folder: Path) -> Path:
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
        return preferred[0] # 返回第一个优先文件
    # 没有优先文件，返回按文件名排序的第一个
    return sorted(candidates, key=lambda p: p.name)[0]


def _center_square_crop(image: Image.Image) -> Image.Image: # 强制把任何形状的图片变成“正方形”
    width, height = image.size  #获取原始图片的宽和高
    if width == height:
        return image 

    target_size = min(width, height)  # 取短的那根轴作为基准
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS

    return ImageOps.fit(
        image,
        (target_size, target_size), # 目标尺寸：正方形
        method=resample,
        centering=(0.5, 0.5),   #中心对齐
    )


def _resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        resample = Image.LANCZOS
    # LANCZOS重采样提供高质量缩放
    return image.resize(size, resample=resample)
