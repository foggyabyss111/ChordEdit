#1.编码阶段：把图片变成 Latent（压缩数据），把文字变成 Embeddings（特征向量）
#2.编辑循环（核心）：在 Latent 空间里，根据“源描述”和“目标描述”的差异，计算出一个“修改方向” 
#3.解码阶段：把修改后的 Latent 变回图片

from __future__ import annotations  # 支持类型注解中的自引用（如类名作为类型）

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np  # 导入 NumPy，用于多维数组的数学运算
import torch
from PIL import Image, ImageOps # 导入 PIL 库，用于加载、保存和基础图像处理
# DDPMScheduler: 调度器，控制扩散过程中加噪和去噪的步长/逻辑
# AutoencoderKL: 变分自编码器（VAE），负责将图像转换到隐空间（Latent Space）及还原
# UNet2DConditionModel: UNet网络，扩散模型的核心，负责在隐空间预测噪声
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel 
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils import BaseOutput
from torchvision import transforms
from torchvision.transforms import InterpolationMode
#CLIPTextModel：文本编码器，将文本转为特征向量
#CLIPImageProcessor：图像预处理，用于安全检查
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel

DEFAULT_SEED = 42 # 默认随机种子，确保生成结果的可复现性
DEFAULT_COMPUTE_DTYPE = torch.float32 # 默认计算精度
DEFAULT_SAFETY_CHECKER_ID = "CompVis/stable-diffusion-safety-checker"

LOGGER = logging.getLogger(__name__) #日志记录器，用于输出调试信息和警告


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------


@dataclass
class ChordEditPipelineOutput(BaseOutput):
    images: List[Image.Image] | torch.Tensor #输出的图像（PIL格式或张量）
    latents: torch.Tensor #对应的潜在空间表示

class _CenterSquareCropTransform:
    """Center-crop the shorter image dimension before resizing."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image  #已经是正方形，直接返回
        
        target = min(width, height) #取较短边作为目标尺寸
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined] 高质量重采样
        except AttributeError:  # pragma: no cover
            resample = Image.LANCZOS # 兼容旧版本PIL
        return ImageOps.fit(
            image,
            (target, target), #目标尺寸：正方形
            method=resample,
            centering=(0.5, 0.5), #中心对齐
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# ChordEdit Pipeline 核心类： 负责串联所有的模型组件
# ---------------------------------------------------------------------------


class ChordEditPipeline(DiffusionPipeline):
    """Standalone pipeline that wires up diffusers modules with the Chord editor."""

    def __init__(
        self,
        unet: UNet2DConditionModel, #预测噪声的核心模型，输入：图像特征+文本特征；输出：噪声。
        scheduler: DDPMScheduler, #步进调度器，控制扩散步长
        vae: AutoencoderKL, #变分自编码器，图像↔latent
        tokenizer, #文本分词器，文本→token IDs
        text_encoder: CLIPTextModel, #文本编码器，token IDs→embeddings，为UNet提供条件信息
        default_edit_config: Optional[Dict[str, Any]] = None, #默认的编辑配置参数，可以在每次调用时覆盖
        image_size: int = 512, #输入图像的标准物理分辨率，默认为512x512
        device: Optional[str | torch.device] = None, #计算设备
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE, #数值精度，决定显存消耗
        use_attention_mask: bool = False, 
        use_center_crop: bool = True,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> None:
        # 1. 基础初始化
        super().__init__()
        # 2. 模块注册：注册所有模块到Diffusers框架，方便后续管理和调用
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        # 3. 设置运行设备：优先使用GPU
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # 4.准备计算精度与设备转移
        self._compute_dtype = compute_dtype
        self._use_attention_mask = bool(use_attention_mask)
        self.to(self._device) #将所有模块移到目标设备
        self._set_compute_precision()

        self.default_edit_config = default_edit_config or {}
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)
        # 6. 核心工具准备：构建 VAE 的图像预处理流水线
        # 负责把原始 PIL 图片剪裁、缩放、并标准化到 [-1, 1] 区间
        self._vae_transform = self._build_vae_transform()
        # 7. 模型设为推理模式：eval() 会关闭模型中的 Dropout 等训练专用层，保证结果稳定
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        #8.时间步上限：获取训练时的最大步数索引（999）
        self._max_unet_timestep = self.scheduler.config.num_train_timesteps - 1
        #9.安全检查器初始化
        self._use_safety_checker = bool(use_safety_checker)
        self._safety_checker_id = safety_checker_id
        self._safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self._safety_feature_extractor: Optional[CLIPImageProcessor] = None

        # 如果开启了安全检查，则执行初始化逻辑·
        if self._use_safety_checker:
            self._init_safety_checker()

    def _set_compute_precision(self) -> None: #内部工具函数：遍历核心模块，确保它们的计算精度Dtype一致，防止计算冲突报错
        modules = (self.unet, self.vae, self.text_encoder)
        for module in modules:
            if module is not None:
                module.to(device=self._device, dtype=self._compute_dtype)
    def _init_safety_checker(self) -> None:
        if not self._safety_checker_id:
            LOGGER.warning("Safety checker requested but no identifier provided; disabling safety checks.")
            self._use_safety_checker = False
            return
        try:
            #加载安全检查模型
            self._safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                self._safety_checker_id,
                torch_dtype=self._compute_dtype,
            ).to(self._device)
            #加载图像特征提取器
            self._safety_feature_extractor = CLIPImageProcessor.from_pretrained(self._safety_checker_id)
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("Failed to initialize safety checker (%s). Safety checks disabled.", exc)
            self._safety_checker = None
            self._safety_feature_extractor = None
            self._use_safety_checker = False

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_local_weights(
        cls,
        component_paths: Dict[str, str], # 包含各组路径的字典 
        *,
        default_edit_config: Optional[Dict[str, Any]] = None,
        device: Optional[str | torch.device] = None, 
        torch_dtype: torch.dtype = torch.float32,
        image_size: int = 512, #image_size=52
        use_center_crop: bool = True,
        compute_dtype: torch.dtype = DEFAULT_COMPUTE_DTYPE,
        use_attention_mask: bool = False,
        use_safety_checker: bool = False,
        safety_checker_id: Optional[str] = DEFAULT_SAFETY_CHECKER_ID,
    ) -> "ChordEditPipeline":
        """Instantiate the pipeline from individual component checkpoints."""
        # 分别加载各个组件
        unet = UNet2DConditionModel.from_pretrained(
            component_paths["unet_path"],
            torch_dtype=torch_dtype,
        )
        scheduler = DDPMScheduler.from_pretrained(component_paths["scheduler_path"])
        vae = AutoencoderKL.from_pretrained(component_paths["vae_path"], torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(component_paths["tokenizer_path"])
        text_encoder = CLIPTextModel.from_pretrained(
            component_paths["text_encoder_path"],
            torch_dtype=torch_dtype,
        )

        # 创建管道实例
        return cls(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_attention_mask=use_attention_mask,
            use_center_crop=use_center_crop,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()  #推理阶段，不需要计算梯度，省内存
    def __call__(
        self,
        image: Image.Image | torch.Tensor, #输入：原始图片（PIL格式或张量）
        *,
        source_prompt: str, #输入：对原始图片的描述
        target_prompt: str, #输入：对目标图片的描述
        edit_config: Optional[Dict[str, Any]] = None, #edit_config：编辑参数配置，控制编辑强度、步数等
        seed: Optional[int] = None, #随机种子，保证结果可复现
        output_type: str = "pil", #输出格式："pil" 或 "tensor"
    ) -> ChordEditPipelineOutput:
        """Run ChordEdit once on a single image."""

    #1.准备配置参数（如步数、噪声采样）
        cfg = dict(self.default_edit_config) #复制默认配置
        if edit_config:
            cfg.update(edit_config) #用传入的配置覆盖默认值
        # 必需的配置项
        required_keys = ["noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale"]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise ValueError(f"edit_config is missing required keys: {missing}")
    #2.图像预处理：把图片从像素空间压缩到VAE的latent空间
      #2.1 像素空间归一化（Pixel Space）
        #状态：RGB 图像 -> 归一化张量。Shape: [Batch, 3, 512, 512], 值域: [-1, 1]。
        pixel_values = self._prepare_image_tensor(image) #pixel_values:标准化的像素值,把图片像素从 [0, 255] 变成了 [-1, 1]
      #2.2 空间降维映射
        # 计算目的: 通过 VAE 将高维像素映射到低维连续分布的众数 (Mode) 上。
        # 状态: Shape 变为 [Batch, 4, 64, 64]。空间尺寸缩小 8 倍，通道数变为 4，保留了深度语义特征。
        latents = self._encode_image_to_latent(pixel_values)
    #3.文本编码：把源文本和目标文本转换成CLIP的特征向量，作为UNet的条件输入
        src_embed = self.encode_prompt([source_prompt]) #src_embed:源文本的CLIP特征，[1, 77, 768]，77：文本的序列长度（padding后），768是每个token的特征维度
        tgt_embed = self.encode_prompt([target_prompt]) #tgt_embed: [1, 77, 768] - 目标文本的CLIP特征
    #4. 准备编辑参数和噪声
        #准备输出容器
        output_latents: List[torch.Tensor] = []
        decoded_batches: List[torch.Tensor] = []

        #格式化编辑参数
        edit_params = self._prepare_edit_params(cfg)
        seed_value = int(seed) if seed is not None else DEFAULT_SEED

        #生成多个噪声样本 
        #noise_list: 每个元素是 [1, 4, 64, 64] 的随机噪声
        noise_list = self._prepare_noise_list(
            latents=latents,
            seed_value=seed_value,
            num_noises=edit_params["noise_samples"],
        )
    #5.核心编辑循环：在Latent空间里，根据文本条件和噪声，迭代地调整图像特征
        # 输入：原始latent + 文本条件 + 噪声
        # 输出：编辑后的latent [1, 4, 64, 64]
        x0_pred = self._run_edit(   
            x_src=latents,
            src_embed=src_embed,
            edit_embed=tgt_embed,
            noise=noise_list,
            params=edit_params,
        )
        # 这个输出包含了被编辑后的图像信息：
        # - 保持了原始图像的基本结构（从x_src继承）
        # - 融入了目标文本的语义（从edit_embed引导）
        # - 经过多步迭代优化（通过_run_edit循环）

    #6.解码和后处理
        #decoded: [1, 3, 512, 512] - 解码回像素空间，值域[0,1]
        decoded = self._decode_latent_to_image(x0_pred)

        #安全检查
        decoded, _ = self._apply_safety_checker(decoded)
        #保存结果（移到CPU，释放GPU显存）
        output_latents.append(x0_pred.detach().cpu())
        decoded_batches.append(decoded.detach().cpu())

        #拼接结果
        images_tensor = torch.cat(decoded_batches, dim=0) #[1, 3, 512, 512]
        latents_tensor = torch.cat(output_latents, dim=0) #[1, 4, 64, 64]
        # 转换为PIL格式（如果需要）
        images = self._tensor_to_pil(images_tensor) if output_type == "pil" else images_tensor

        return ChordEditPipelineOutput(
            images=images,
            latents=latents_tensor,
        )

    def encode_prompt(self, prompts: Sequence[str]) -> torch.Tensor:
        """Public helper mirroring diffusers pipelines for text encoding."""
        return self._encode_text(prompts)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _prepare_image_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """将输入图像转换为标准化张量 [0,255] → [0,1] → [-1,1]"""
        if isinstance(image, Image.Image):
            #=== 情况1：输入是PIL图片 ===
            #_vae_transform 执行: 中心裁剪→resize→ToTensor→Normalize
            #输出: [3, 512, 512], 值域[-1,1]
            vae_tensor = self._vae_transform(image)
        elif torch.is_tensor(image):
            #=== 情况2：输入是tensor ===
            tensor = image.float()
            # 如果没有batch维度，添加batch维度
            if tensor.ndim == 3: #[C, H, W]
                tensor = tensor.unsqueeze(0) #→ [1, C, H, W]
            # 如果值域是[0,255]，缩放到[0,1]
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            # 从[0,1]缩放到[-1,1] (VAE期望的输入范围)
            tensor = tensor * 2.0 - 1.0
            vae_tensor = tensor
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor.")

        #确保有batch维度 [batch, C, H, W]
        if vae_tensor.ndim == 3:
            vae_tensor = vae_tensor.unsqueeze(0)
        #如果需要中心裁剪且不是正方形
        if self._use_center_crop and vae_tensor.ndim == 4:
            _, _, height, width = vae_tensor.shape
            if height != width:
                side = min(height, width)
                top = (height - side) // 2
                left = (width - side) // 2
                #执行裁剪 [batch, C, top:top+side, left:left+side]
                vae_tensor = vae_tensor[:, :, top : top + side, left : left + side]
        return vae_tensor.to(device=self._device, dtype=self._compute_dtype)

    def _encode_image_to_latent(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """将像素空间的图像编码到VAE的latent空间"""

        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) #scaling_factor: VAE的缩放因子，通常是0.18215，是训练时为了稳定
        #输入：pixel_values: [batch, 3, 512, 512], 值域[-1,1]
        pixel_values = pixel_values.to(device=self._device, dtype=self._compute_dtype)

        #VAE编辑器输出：latent_dist (高斯分布的参数)
        # latent_dist.mode() 取分布的众数（峰值），即最可能的latent
        # 也可以取sample()进行采样，但mode更稳定
        latents = self.vae.encode(pixel_values).latent_dist.mode()
        # 输出：[batch, 4, 64, 64]
        latents = latents * scaling_factor  # 乘以缩放因子，匹配训练时的尺度
        return latents.to(device=self._device, dtype=self._compute_dtype)

    def _decode_latent_to_image(self, latents: torch.Tensor) -> torch.Tensor:
        """
        将latent解码回像素空间
        输入: latents [batch, 4, 64, 64] (编辑后的特征)
        输出: decoded [batch, 3, 512, 512], 值域[0,1]
        """
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents.to(device=self._device, dtype=self._compute_dtype) #输入
        # .sample 获取解码后的图像
        decoded = self.vae.decode(latents / scaling_factor).sample #输出，[batch, 3, 512, 512], 值域[-1,1]

        # 从[-1,1]转换到[0,1] (方便保存为图片)
        decoded = (decoded.clamp(-1.0, 1.0) + 1.0) / 2.0
        return decoded.to(dtype=self._compute_dtype)

    def _apply_safety_checker(self, images: torch.Tensor) -> tuple[torch.Tensor, List[bool]]:
        """应用安全检查,检测NSFW内容
    
           输入: images [batch, 3, 512, 512], 值域[0,1]
           输出: 
         - 过滤后的images (NSFW的会被黑屏替换)
         - has_nsfw列表 [bool] 每个样本是否包含NSFW内容
        """
        batch = images.shape[0]

        # === 安全检查的条件检查 ===
        # 如果: 功能未启用 OR 模型未加载 OR 特征提取器未加载 OR batch为空
        if (
            not self._use_safety_checker #功能未启用
            or self._safety_checker is None #模型不存在
            or self._safety_feature_extractor is None #特征提取器不存在 
            or batch == 0
        ):
            return images, [False] * batch  #直接返回，不进行安全检查
        
        # === 准备输入 ===
        images_clamped = images.detach().clamp(0.0, 1.0) #确保输入在[0,1]范围内
        pil_images = self._tensor_to_pil(images_clamped) #tensor → PIL
        try:
            #提取CLIP特征用于安全检查
            clip_input = self._safety_feature_extractor(images=pil_images, return_tensors="pt").to(self._device)
            #准备图像输入[batch, H, W, C] 值域[-1,1]
            images_np = np.stack([np.array(img).astype(np.float32) / 255.0 for img in pil_images], axis=0)
            images_np = images_np * 2.0 - 1.0 #从[0,1]转换到[-1,1]
            #运行安全检查器
            _, has_nsfw_concept = self._safety_checker(
                images=images_np,
                clip_input=clip_input.pixel_values.to(self._device),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.warning("Safety checker failed (%s). Skipping safety checks.", exc)
            return images, [False] * batch
        
        # === 处理检查结果 ===
        # 转换为bool列表
        if isinstance(has_nsfw_concept, torch.Tensor):
            has_nsfw = has_nsfw_concept.detach().cpu().to(dtype=torch.bool).tolist()
        else:
            has_nsfw = [bool(flag) for flag in has_nsfw_concept]

        #如果有NSFW内容，用黑屏替换
        if any(has_nsfw):
            for idx, flagged in enumerate(has_nsfw):
                if flagged:
                    images[idx] = torch.zeros_like(images[idx]) #全黑图像
        return images, has_nsfw

    def _encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """将文本提示编码为CLIP特征向量,作为UNet的条件输入
        输入:prompts:文本列表
        输出:[batch, 77, 768] CLIP特征
            - batch: 文本数量
            - 77: CLIP的最大文本长度(padding后)
            - 768: 每个token的特征维度
            过程:
            1. 分词器将文本转换为token IDs,进行padding和截断,得到固定长度的输入
            2. 文本编码器将token IDs转换为特征向量,作为UNet的条件输入,引导图像编辑
            3. 输出的特征张量被移动到目标设备,并转换为计算精度dtype
        """
        # === 1. 分词: 文本 → token IDs ===
        inputs = self.tokenizer(
            list(prompts),
            padding="max_length", #padding到相同长度
            truncation=True,    #截断超长文本
            max_length=self.tokenizer.model_max_length, #最大长度(77)
            return_tensors="pt", 

        )
        # input_ids: [batch, 77] - 每个位置是token在词典中的索引
        # attention_mask: [batch, 77] - 1表示真实token, 0表示padding

        input_ids = inputs.input_ids.to(self._device)
        attn_mask = inputs.attention_mask.to(self._device) if self._use_attention_mask else None

        # === 2. 文本编码: token IDs → CLIP特征向量 ===
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        
        # last_hidden_state: [batch, 77, 768]
        # 每个位置的768维向量表示该token在上下文中的语义
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs[0] # 兼容旧版本
        return hidden.to(device=self._device, dtype=self._compute_dtype)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """
        将图像张量转换为PIL Image列表
        输入: tensor [batch, 3, H, W], 值域[0,1]
        输出: List[PIL.Image] - 每个元素是一个PIL图像
        """
        # 确保在CPU上，值域在[0,1]之间，适合转换为PIL格式
        tensor = tensor.detach().cpu().clamp(0.0, 1.0)

        # ToPILImage转换器: [C, H, W] → PIL Image
        to_pil = transforms.ToPILImage()
        return [to_pil(sample) for sample in tensor]

    def _build_vae_transform(self) -> transforms.Compose:
        """Create image->latent preprocessing transform."""
        """
        构建VAE的图像预处理流水线
    步骤:
        1. 中心裁剪成正方形 (可选)
        2. Resize到512x512
        3. PIL → Tensor [0,255] → [0,1]
        4. Normalize [0,1] → [-1,1]
        """
        ops: List[Any] = []
        # === 步骤1: 中心裁剪 (可选) ===
        if self._use_center_crop:
            ops.append(_CenterSquareCropTransform())
            resize_interp = InterpolationMode.LANCZOS
        else:
            resize_interp = InterpolationMode.BILINEAR
        
        # === 步骤2: Resize到512x512 ===
        ops.append(
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=resize_interp,
            )
        )

        # === 步骤3-4: ToTensor + Normalize ===
        ops.extend(
            [
                transforms.ToTensor(), #[0,255] PIL → [0,1] Tensor
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), #[0,1] → [-1,1]
            ]
        )
        return transforms.Compose(ops)

    def _prepare_edit_params(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备并验证编辑参数
    
        输入配置参数:
        noise_samples: 噪声样本数 
        n_steps: 编辑迭代步数 
        t_start: 起始时间步 [0,1] 
        t_end: 结束时间步 [0, t_start] (默认0.3)
        t_delta: 时间差分 
        step_scale: 步长缩放因子 
        cleanup: 是否执行清理步骤 
        """
        params = dict(cfg)
        # === 参数验证和范围限制 ===
        params["noise_samples"] = int(max(1, params["noise_samples"])) #至少1个
        params["n_steps"] = int(max(1, params["n_steps"]))   #至少1步
        params["t_start"] = float(max(0.0, min(1.0, params["t_start"]))) # [0,1]
        params["t_end"] = float(max(0.0, min(params["t_start"], params["t_end"]))) # [0, t_start]
        # t_delta 处理: 确保 t_s - delta >= 0
        t_delta = float(max(0.0, min(1.0, params["t_delta"]))) 
        if t_delta >= params["t_start"]:
            # 如果delta太大，调整到略小于t_start
            safe_max = max(1, self._max_unet_timestep)
            t_delta = max(0.0, params["t_start"] - 1.0 / safe_max)
        params["t_delta"] = t_delta
        params["step_scale"] = float(params["step_scale"]) # 编辑强度
        params["cleanup"] = bool(params.get("cleanup", False)) # 是否执行清理
        return params

    def _prepare_noise_list(
        self,
        latents: torch.Tensor,  #[1, 4, 64, 64] - 用于确定形状和设备
        seed_value: int,        # 随机种子，确保可复现
        num_noises: int,        # 要生成的噪声数量
    ) -> List[torch.Tensor]:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        #生成 num_noises 个与 latents 相同形状的随机噪声
        noise_list = [
            torch.randn_like(latents, device=latents.device, dtype=self._compute_dtype)
            for _ in range(num_noises)
        ]
        return noise_list

    def _time_to_index(self, batch: int, t_scalar: float, device, dtype=torch.long):
        idx = round(self._max_unet_timestep * float(t_scalar))
        idx = max(0, min(self._max_unet_timestep, idx))
        return torch.full((batch,), idx, device=device, dtype=dtype)

    def _get_alpha_sigma(self, tensor: torch.Tensor, timesteps: torch.Tensor):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=torch.float32, device=tensor.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)  
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(dtype=tensor.dtype, device=tensor.device)
        sigma_t = sigma_t.to(dtype=tensor.dtype, device=tensor.device)
        eps = torch.finfo(alpha_t.dtype).eps
        alpha_t = alpha_t.clamp(min=eps)
        return alpha_t, sigma_t

    def _pred_x0(self, x_anchor, timesteps, cond, noise):
        alpha_t, sigma_t = self._get_alpha_sigma(x_anchor, timesteps)
        z_t = alpha_t * x_anchor + sigma_t * noise
        noise_pred = self.unet(
            sample=z_t,
            timestep=timesteps,
            encoder_hidden_states=cond,
            return_dict=False,
        )[0]
        x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
        return x0_pred

#_u_estimate 函数：计算修改方向
    def _u_estimate(self, x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float):
        """
        计算在当前时间步应该向哪个方向修改图像

        参数：
            x_anchor: [1, 4, 64, 64] - 当前latent(锚点)
            src_embed: [1, 77, 768] - 源文本的embedding
            edit_embed: [1, 77, 768] - 目标文本的embedding
            noise: List[torch.Tensor] - 多个噪声样本 [num_noises, 4, 64, 64]
            t_s: float - 当前时间步 [0,1]
            delta: float - 时间差分，用于计算两个时间步的编辑方向
        返回：
            u_hat: [1, 4, 64, 64] - 编辑方向向量
        """
        # 1. 获取基础元信息：batch_size 以及设备信息 (确保计算在同一块 GPU 上)
        batch, device = x_anchor.shape[0], x_anchor.device  #x_anchor: 当前图像在 Latent 空间的特征,Shape [1, 4, 64, 64]
        # 2. 将 [0,1] 的浮点时间戳转为扩散模型能够识别的整数步数 (如 0-999)
        t_idx_s = self._time_to_index(batch, t_s, device=device) #t_idx_s:当前时间步的“模糊程度”,t_s：当前时间步，1表示纯噪声，0表示干净图像
        t_idx_s0 = self._time_to_index(batch, max(0.0, t_s - delta), device=device) #t_idx_s0: 向前平移 delta 后的时间步索引,t_s - delta：稍微"更干净"的时间步

        # 3.规范化噪声输入：确保noise是一个列表，方便后续循环或堆叠处理
        noises = noise if isinstance(noise, (list, tuple)) else [noise] #使用 4 个噪声样本

        # 4. 获取扩散系数 (Schedulers Parameters):
        # alpha: 原始信息的保留比例；sigma: 噪声的添加比例
        alpha_s, sigma_s = self._get_alpha_sigma(x_anchor, t_idx_s)
        alpha_prev, sigma_prev = self._get_alpha_sigma(x_anchor, t_idx_s0)

        num_noises = len(noises)
        # 5. 将噪声列表压成一个张量 
        noise_stack = torch.stack(noises, dim=0) # 噪声堆叠后形状: torch.Size([4, 1, 4, 64, 64])
        """
        堆叠前:4个独立的tensor,无法批量处理
        noise_0.shape  # [1, 4, 64, 64]
        noise_1.shape  # [1, 4, 64, 64]
        noise_2.shape  # [1, 4, 64, 64]
        noise_3.shape  # [1, 4, 64, 64]

        # 堆叠后:1个5D tensor,可以批量处理
        noise_stack.shape  # [4, 1, 4, 64, 64]

        # 这样就可以一次性把所有噪声传给UNet
        # 在后续的代码中,会用这个堆叠的tensor进行广播和批量计算
        """

        # -------------------------------------------------------------------------
        # 6. 张量广播 (Broadcasting)：为了一次性跑完所有组合，将变量维度进行扩展
        # 物理含义：准备好“原材料”，让每一组噪声都能对应到当前的图像特征
        # -------------------------------------------------------------------------
        x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        #x_anchor:[1, 4, 64, 64] 
        # unsqueeze(0): [1, 1, 4, 64, 64] - 在开头添加一个维度
        # expand: [num_noises, 1, 4, 64, 64] - 复制num_noises份
        alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        alpha_prev_b = alpha_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
        sigma_prev_b = sigma_prev.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
      # 模拟前向扩散过程
        # 计算目的: 根据 DDPM 公式构造加噪观测值 $z_t = \alpha_t x_0 + \sigma_t \epsilon$。
        # alpha_b: 信号保留衰减率；sigma_b: 高斯噪声方差注入量。
        z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack # z_s: 当前时间步 t_s 的加噪特征图
        z_prev = alpha_prev_b * x_anchor_b + sigma_prev_b * noise_stack #z_prev: 较小时间步 t_s0 (噪声较少) 的加噪特征图

        # -------------------------------------------------------------------------
        # 8. 核心堆叠 (Crucial Stack)：将 (s, s0) 和 (src, edit) 组合成一个大批次
        # 维度变化: [num_noises, 4, batch, 4, 64, 64] -> 再 reshape 成扁平的 Batch 维度
        # -------------------------------------------------------------------------
        samples = torch.stack([z_s, z_s, z_prev, z_prev], dim=1)
        samples = samples.reshape(num_noises * 4 * batch, *x_anchor.shape[1:])

        # 9. 文本条件堆叠：[src, tgt,src,tgt] 对应上面samples的顺序
        conds = torch.cat([src_embed, edit_embed, src_embed, edit_embed], dim=0)
        repeat_dims = [num_noises] + [1] * (conds.dim() - 1)
        conds = conds.repeat(*repeat_dims)

        timesteps = torch.cat([t_idx_s, t_idx_s, t_idx_s0, t_idx_s0], dim=0)
        timesteps = timesteps.repeat(num_noises)

        alpha_cat = torch.stack(
            [alpha_s_b, alpha_s_b, alpha_prev_b, alpha_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)
        sigma_cat = torch.stack(
            [sigma_s_b, sigma_s_b, sigma_prev_b, sigma_prev_b],
            dim=1,
        ).reshape(num_noises * 4 * batch, 1, 1, 1)

        noise_pred = self.unet(
            sample=samples,
            timestep=timesteps,
            encoder_hidden_states=conds,
            return_dict=False,
        )[0]

        x0_all = (samples - sigma_cat * noise_pred) / alpha_cat
        x0_all = x0_all.reshape(num_noises, 4, batch, *x_anchor.shape[1:])
        x_src_p_s, x_tar_p_s, x_src_p_s0, x_tar_p_s0 = x0_all.unbind(dim=1)
      # 计算语义位移向量
        # 计算目的: 在相同的噪声扰动下，计算模型在“目标条件”与“源条件”下的预测差异（即残差）
        # dv_s 和 dv_s0 是【编辑方向向量】，形状都是 [1, 4, 64, 64]
        dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises) #dv_s (大噪声层): “粗调宏观方向”，把握图像的整体轮廓和结构变化。
        dv_s0 = (x_tar_p_s0 - x_src_p_s0).sum(dim=0) / float(num_noises)#dv_s0 (小噪声层): “细调微观方向”，修正边缘和纹理细节。
       

    #时间修正与加权平衡：
    # denom: 时间分母，防止除零
        denom = (t_s + delta) #denom：分母（时间修正）
        if denom <= 1e-6:
            return dv_s
        return (delta * dv_s + t_s * dv_s0) / denom #返回：两个不同时间步速度的加权平均，作为最终的引导方向

    #核心编辑执行器 （这是一个迭代过程，每一步都让图像更接近目标）
    """
    过程：
    1.从x_src开始
    2.在多个时间步上迭代
    3.每个时间步用_u_estimate 计算编辑方向
    4.用step_scale 控制移动步长
    5.逐渐从x_src 移动到目标
    """

    def _run_edit( #_run_edit 函数：迭代循环，根据修改方向逐步调整图像特征
        self,
        x_src: torch.Tensor,    #[1, 4, 64, 64] (vae-latents空间) - 输入1：原始图像的latent
        src_embed: torch.Tensor, #[1, 77, 768] （CLIP空间）- 输入2：源文本的CLIP特征
        edit_embed: torch.Tensor, #[1, 77, 768] （CLIP空间）- 输入3：目标文本的CLIP特征
        noise: List[torch.Tensor], #输入4：多个噪声样本 [num_noises, 4, 64, 64]
        params: Dict[str, Any], #输入5：编辑参数
    ) -> torch.Tensor:          #输出：[1, 4, 64, 64] - 编辑后的latent
        device = x_src.device
        if params["n_steps"] == 1:
            t_grid = [params["t_start"]] 
        else:
            t_grid = torch.linspace(
                params["t_start"],
                params["t_end"],
                steps=params["n_steps"],
                device=device,
            ).tolist()
      # 在 Latent 空间中，以 x_src 为起点，一步步走向目标语义。
        x_curr = x_src #从原始latent开始
        for t_s in t_grid: #在多个时间步上迭代
        #u_hat: 当前时间步的修改方向，编辑位移向量
        #物理含义：在latent空间中，指向“目标文本语义” 减去“原始文本语义” 的方向
            u_hat = self._u_estimate(  # u_hat：这一步要走的方向和距离,是一个【方向向量】，指向目标文本的语义方向
                x_curr,   # 当前latent
                src_embed, #原始描述
                edit_embed, #目标描述
                noise,     #噪声样本
                float(t_s),  #当前时间
                params["t_delta"],  #时间差分
            )
            # 状态演变: x_{new} = x_{old} + 步长 * 位移向量
            x_curr = x_curr + params["step_scale"] * u_hat #更新当前图像特征

        if params["cleanup"]:
            # [收尾清理阶段]
            # 计算目的: 利用目标文本引导，直接预测出时间步为 0 的干净状态 $x_0$。
            # 物理含义: 消除累积的截断误差，强制将 Latent 对齐到清晰的图像流形 (Image Manifold) 上。
            t_end_idx = self._time_to_index(x_src.shape[0], params["t_end"], device=device)
            x_curr = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0]) #_pred_x0 的作用:利用 UNet 预测出完全不含噪声的原始状态，把最后一点模糊感去掉，让导出的图片边缘更清晰、纹理更真实

        return x_curr #返回编辑后的latent
