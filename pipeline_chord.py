"""
ChordEdit Pipeline 核心机制：
1. 编码阶段：将图片编码为 Latent（压缩隐特征），将文字编码为 Embeddings（语义特征向量）。
2. 编辑循环：在 Latent 空间中，根据“源描述”和“目标描述”的语义差计算出“修改方向”（位移向量），并逐步迭代。
3. 解码阶段：将修改后的 Latent 还原为像素空间的图片。
"""

from __future__ import annotations  # 支持类型注解中的自引用（如类名作为类型）

import logging  # 日志记录模块，用于输出调试信息和警告
from dataclasses import dataclass   # 用于创建数据类（简单的容器）
from typing import Any, Dict, List, Optional, Sequence  # 类型提示工具

import numpy as np  # 导入 NumPy，用于多维数组的数学运算
import torch
from PIL import Image, ImageOps # 导入 PIL 库，用于加载、保存和基础图像处理
# DDPMScheduler: 调度器，控制扩散过程中加噪和去噪的步长/逻辑
# AutoencoderKL: 变分自编码器（VAE），负责将图像转换到隐空间（Latent Space）及还原
# UNet2DConditionModel: UNet网络，扩散模型的核心，负责在隐空间预测噪声
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel 
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
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


class ContextFlowInsertAttnProcessor:
    """
    ContextFlow 核心注意力机制处理器 (Adaptive Context Enrichment)
    物理含义: 在模型去噪过程中，拦截指定层 (浅层) 的自注意力计算。
             通过提取“重建路径”的背景信息 (Key/Value)，并与“编辑路径”拼接，
             强制让模型在生成新物体时，“看”到原图的背景结构，从而避免破坏背景并实现自然融合。
    """
    def __init__(self, layer_index: int, enabled: bool) -> None:
        self.layer_index = int(layer_index) # 记录当前注意力层的索引
        self.enabled = bool(enabled)        # 是否在该层启用 ContextFlow 拦截

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,        # 当前层输入的特征图 [batch, seq_len, dim]
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 文本条件输入 (若是自注意力则为 None)
        attention_mask: Optional[torch.Tensor] = None,         # 注意力掩码
        temb: Optional[torch.Tensor] = None,                   # 时间步嵌入特征
        contextflow_state: Optional[Dict[str, Any]] = None,    # 外部传入的状态控制信号
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # ==========================================
        # [基础注意力准备阶段]
        # ==========================================
        residual = hidden_states    # 保存残差连接的输入

        # 空间归一化处理（如果该层存在）
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        # 将 2D 图像特征 [B, C, H, W] 展平为 1D 序列 [B, H*W, C] 供 Transformer 处理
        if input_ndim == 4: 
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        # 判断当前是交叉注意力（图文交互）还是自注意力（图图交互）
        is_cross_attention = encoder_hidden_states is not None
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # 准备注意力掩码和组归一化
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            
        # 提取 Query (Q 永远只来自当前分支)
        query = attn.to_q(hidden_states)
        
        # 处理 Key 和 Value 的来源
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states   # 自注意力：K/V 来自图像本身
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        key_states_edit = encoder_hidden_states
        value_states_edit = encoder_hidden_states

        # ==========================================
        # [ContextFlow 拦截判定]
        # 条件: 1. 外部传入激活信号 2. 当前层允许注入 3. 必须是自注意力层 (保护文本交叉注意力)
        # ==========================================
        should_use_contextflow = (
            bool(contextflow_state is not None)
            and bool(contextflow_state.get("enabled", False))
            and self.enabled
            and not is_cross_attention
        )
        
        if should_use_contextflow and contextflow_state is not None:
            mode = str(contextflow_state.get("mode", ""))
            # 确认当前处于双路径并行模式且批次大小为偶数
            if mode == "paired" and key_states_edit.shape[0] % 2 == 0 and attention_mask is None:
                half = key_states_edit.shape[0] // 2
                
                # 1. 拆分 Batch，分离重建分支 (res) 和编辑分支 (edit)
                query_res = query[:half]      # 重建分支的 Q [1, seq_len, dim]
                query_edit = query[half:]     # 编辑分支的 Q [1, seq_len, dim]
                
                key_res = attn.to_k(key_states_edit[:half])      # 重建分支的 K (包含原图背景信息)
                value_res = attn.to_v(value_states_edit[:half])  # 重建分支的 V (包含原图结构纹理)
                
                key_edit = attn.to_k(key_states_edit[half:])     # 编辑分支自身的 K
                value_edit = attn.to_v(value_states_edit[half:]) # 编辑分支自身的 V
                
                # 2. 核心富集逻辑 (Adaptive Context Enrichment)
                # 物理含义: 将重建分支的背景 KV 拼接给编辑分支，让编辑分支在生成新物体时参考原图背景
                # 形状变化: [1, seq_len, dim] concat [1, seq_len, dim] -> [1, seq_len*2, dim]
                key_aug = torch.cat([key_edit, key_res], dim=1)
                value_aug = torch.cat([value_edit, value_res], dim=1)
                
                # 调整维度以适应多头注意力 (Multi-Head Attention) 的计算格式
                query_res = attn.head_to_batch_dim(query_res)
                key_res = attn.head_to_batch_dim(key_res)
                value_res = attn.head_to_batch_dim(value_res)
                query_edit = attn.head_to_batch_dim(query_edit)
                key_aug = attn.head_to_batch_dim(key_aug)
                value_aug = attn.head_to_batch_dim(value_aug)
                
                # 3. 分别计算注意力
                # 优化: 重建分支仅用于提供 KV，其输出可直接置零以节省显存和算力
                out_res = torch.zeros_like(query_res)
                out_res = attn.batch_to_head_dim(out_res)
                
                # 编辑分支: 在扩充后的 KV 空间中寻找相似性进行特征融合
                probs_edit = attn.get_attention_scores(query_edit, key_aug, None)
                out_edit = torch.bmm(probs_edit, value_aug)     
                out_edit = attn.batch_to_head_dim(out_edit)    
                
                # 4. 重新拼接回 Batch=2 并输出 (为了保持后续网络层的 Batch 维度一致)
                hidden_states = torch.cat([out_res, out_edit], dim=0)
                hidden_states = attn.to_out[0](hidden_states)   # 线性层投影
                hidden_states = attn.to_out[1](hidden_states)   # Dropout层
                
                # 恢复图像原始的 2D 空间形状 [B, C, H, W]
                if input_ndim == 4:
                    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                # 加上之前的残差
                if attn.residual_connection:
                    hidden_states = hidden_states + residual
                hidden_states = hidden_states / attn.rescale_output_factor
                return hidden_states
                
        # [原生注意力计算 (Fallback)]
        # 如果不满足 ContextFlow 条件（比如处于深层，或过了时间窗口 τ），走正常流程
        key = attn.to_k(key_states_edit)
        value = attn.to_v(value_states_edit)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Pipeline output container
# ---------------------------------------------------------------------------


@dataclass
class ChordEditPipelineOutput(BaseOutput):
    images: List[Image.Image] | torch.Tensor    # 输出的图像（PIL格式或张量）
    latents: torch.Tensor                      # 对应的潜在空间表示

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
        # 8.时间步上限：获取训练时的最大步数索引（999）
        self._max_unet_timestep = self.scheduler.config.num_train_timesteps - 1
        self._contextflow_layer_ratio = 0.25
        self._contextflow_threshold_tau = 0.5
        self._setup_contextflow_processors("insertion")
        # 9.安全检查器初始化
        self._use_safety_checker = bool(use_safety_checker)
        self._safety_checker_id = safety_checker_id
        self._safety_checker: Optional[StableDiffusionSafetyChecker] = None
        self._safety_feature_extractor: Optional[CLIPImageProcessor] = None

        # 如果开启了安全检查，则执行初始化逻辑
        if self._use_safety_checker:
            self._init_safety_checker()

    def _set_compute_precision(self) -> None:   # 内部工具函数：遍历核心模块，确保它们的计算精度Dtype一致，防止计算冲突报错
        modules = (self.unet, self.vae, self.text_encoder)
        for module in modules:
            if module is not None:
                module.to(device=self._device, dtype=self._compute_dtype)

    def _setup_contextflow_processors(self, task_type: str = "insertion", custom_layer_ratio: Optional[float] = None) -> None:
        """
        改进3：根据任务类型（插入/替换/删除）动态选择最佳的特征注入层。
        支持通过 custom_layer_ratio 动态调整浅层注入的深度。
        """
        processor_names = list(self.unet.attn_processors.keys())
        total_layers = len(processor_names)
        if total_layers == 0:
            return
            
        task = task_type.lower()
        if task in ["insert", "insertion", "object_insertion", "object-insertion"]:
            start_ratio = 0.0
            # <--- 核心修改：如果有自定义比例就用自定义的，没有就用默认的
            end_ratio = custom_layer_ratio if custom_layer_ratio is not None else 0.25
        elif task in ["swap", "swapping", "object_swapping", "object-swapping"]:
            start_ratio, end_ratio = 0.25, 0.75
        elif task in ["delete", "deletion", "remove", "removal"]:
            start_ratio, end_ratio = 0.5, 1.0
        else:
            start_ratio, end_ratio = 0.0, 1.0 # 默认全开
            
        start_idx = int(total_layers * start_ratio)
        end_idx = int(total_layers * end_ratio)
        
        processors: Dict[str, ContextFlowInsertAttnProcessor] = {}
        for idx, name in enumerate(processor_names):
            is_target_layer = (start_idx <= idx < end_idx)
            processors[name] = ContextFlowInsertAttnProcessor(layer_index=idx, enabled=is_target_layer)
        self.unet.set_attn_processor(processors)

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
        image_size: int = 512,  # image_size=512
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
    #4. 准备编辑参数、ContextFlow 层级和初始随机噪声      
        output_latents: List[torch.Tensor] = []
        decoded_batches: List[torch.Tensor] = []

        #格式化编辑参数
        edit_params = self._prepare_edit_params(cfg)
        seed_value = int(seed) if seed is not None else DEFAULT_SEED
        #获取动态比例（如果配置里没有，则默认使用0.25）
        current_layer_ratio = edit_params.get("contextflow_layer_ratio", 0.25)
        
        #将自定义比例传给层分配函数
        self._setup_contextflow_processors(
            task_type=edit_params.get("edit_task", "insertion"),
            custom_layer_ratio=current_layer_ratio
        )

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

    @torch.no_grad()
    def insert_object(
        self,
        image: Image.Image | torch.Tensor,
        *,
        source_prompt: str,
        target_prompt: str,
        edit_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_type: str = "pil",
        threshold_tau: float = 0.5,
        layer_ratio: float = 0.25, # ---新增参数---
    ) -> ChordEditPipelineOutput:
        cfg = dict(edit_config or {})
        cfg["contextflow_enabled"] = True
        cfg["edit_task"] = "insertion"
        cfg["contextflow_threshold_tau"] = float(threshold_tau)
        cfg["contextflow_layer_ratio"] = float(layer_ratio) # ---写入配置---
        return self(
            image=image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_config=cfg,
            seed=seed,
            output_type=output_type,
        )

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
        edit_task = str(params.get("edit_task", "")).strip().lower()
        default_contextflow = edit_task in {"insert", "insertion", "object_insertion", "object-insertion"}
        params["contextflow_enabled"] = bool(params.get("contextflow_enabled", default_contextflow))
        params["contextflow_threshold_tau"] = float(
            max(0.0, min(1.0, float(params.get("contextflow_threshold_tau", self._contextflow_threshold_tau))))
        )
        # --- 新增这行，解析自定义的层数比例 ---
        params["contextflow_layer_ratio"] = float(
            max(0.0, min(1.0, float(params.get("contextflow_layer_ratio", self._contextflow_layer_ratio))))
        )

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
        """将[0,1]的浮点时间转换为离散时间步索引 (0-999)
    
           例如: t=0.8 → idx=800 (假设max_timestep=1000)
        """
        idx = round(self._max_unet_timestep * float(t_scalar))
        idx = max(0, min(self._max_unet_timestep, idx)) #确保在有效范围内
        return torch.full((batch,), idx, device=device, dtype=dtype)

    def _get_alpha_sigma(self, tensor: torch.Tensor, timesteps: torch.Tensor):
        """
        获取扩散过程的缩放系数。
        作用: 根据给定的时间步 timesteps，从 scheduler 中提取对应的 alpha 和 sigma。
        物理含义: 
            alpha_t: 原始信号(图像)的保留比例。时间步越大(越接近纯噪声)，alpha 越小。
            sigma_t: 添加噪声的比例。时间步越大，sigma 越大。
            满足公式: z_t = alpha_t * x_0 + sigma_t * noise
        输入:
            - tensor: 用于参考 shape、dtype 和 device 的张量。
            - timesteps: 当前所处的时间步索引 [Batch]
        输出:
            - alpha_t, sigma_t: 对应时间步的扩散系数 [Batch, 1, 1, 1]
        """
        # alphas_cumprod: 累积乘积，来自scheduler
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=torch.float32, device=tensor.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)  
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha_t = alpha_t.to(dtype=tensor.dtype, device=tensor.device)
        sigma_t = sigma_t.to(dtype=tensor.dtype, device=tensor.device)
        eps = torch.finfo(alpha_t.dtype).eps
        alpha_t = alpha_t.clamp(min=eps)
        return alpha_t, sigma_t

    def _pred_x0(self, x_anchor, timesteps, cond, noise):
        """
        从带噪图像直接预测干净图像 x0 (主要用于 cleanup 阶段)。
        作用: 强制对齐图像流形，去除因为截断误差导致的最后一点模糊感。
        物理含义: 利用 UNet 预测出当前图像中的噪声成分，然后将其减去，从而暴露底层的真实结构。
        输入:
            - x_anchor: 当前的 latent 状态 [B, C, H, W]
            - timesteps: 对应的时间步索引
            - cond: 用于引导去噪的文本条件
            - noise: 注入的基础噪声
        输出:
            - x0_pred: 预测出的纯净 latent [B, C, H, W]
        """
        # 获取扩散系数
        # alpha_t: 信号保留比例（越大表示图像越清晰）；# sigma_t: 噪声比例（越大表示噪点越多）
        alpha_t, sigma_t = self._get_alpha_sigma(x_anchor, timesteps) #x_anchor: [1, 4, 64, 64] - 当前latent ；# timesteps.shape: [1]
        # 添加噪声：从当前图像得到带噪版本
        z_t = alpha_t * x_anchor + sigma_t * noise
        # UNet 预测噪声
        noise_pred = self.unet(
            sample=z_t,     # 带噪图像
            timestep=timesteps, # 当前时间步（告诉UNet现在是第几步）
            encoder_hidden_states=cond, # 文本条件（告诉UNet想要什么内容）
            return_dict=False,
        )[0]
        # 4. 反推干净图像：从带噪图像中减去预测的噪声
        # 这是扩散模型的核心公式：x0 = (z_t - sigma_t * noise_pred) / alpha_t
        x0_pred = (z_t - sigma_t * noise_pred) / alpha_t
        return x0_pred  # ← 返回更清晰的版本

    def _predict_noise(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        *,
        recon_cond: Optional[torch.Tensor] = None,
        recon_sample: Optional[torch.Tensor] = None, # [新增参数] 传入纯净的重建样本
        use_contextflow: bool = False,
    ) -> torch.Tensor:
        """
        单步去噪的噪声预测封装。
        如果启用了 ContextFlow，将自动构建“双路径”并行计算。
        
        输入:
        - sample: 当前编辑分支的加噪样本 [B, 4, 64, 64]
        - timesteps: 当前时间步
        - cond: 编辑分支的文本特征 (目标描述)
        - recon_cond: 重建分支的文本特征 (源描述)
        - recon_sample: 重建分支的纯净加噪样本 (原图加噪)，专供提取背景 KV
        - use_contextflow: 是否开启上下文富集
        """
        # [常规分支] 不使用 ContextFlow 或没有提供重建条件时，走标准的 UNet 前向
        if not use_contextflow or recon_cond is None:
            return self.unet(
                sample=sample,
                timestep=timesteps,
                encoder_hidden_states=cond,
                return_dict=False,
            )[0]
            
        # [双路径分支] ContextFlow 启用时：
        # 1. 在 Batch 维度拼接，构建并行路径: [重建分支, 编辑分支]
        # 如果没有传入独立的 recon_sample，就回退到使用当前 sample
        r_sample = recon_sample if recon_sample is not None else sample
        paired_sample = torch.cat([r_sample, sample], dim=0)       # 形状: [2*N, 4, 64, 64]
        paired_timesteps = torch.cat([timesteps, timesteps], dim=0) # 形状: [2*N]
        paired_cond = torch.cat([recon_cond, cond], dim=0)       # 形状: [2*N, 77, 768]
        
        # 2. 传递控制信号给注意力层
        context_state = {"enabled": True, "mode": "paired"}
        
        # 3. 执行一次 UNet 前向，同时跑完两条路径
        paired_noise = self.unet(
            sample=paired_sample,
            timestep=paired_timesteps,
            encoder_hidden_states=paired_cond,
            cross_attention_kwargs={"contextflow_state": context_state}, # 将状态信号通过 kwargs 透传进所有的 AttnProcessor
            return_dict=False,
        )[0]
        
        # 4. 拆分输出，重建分支只是为了提供过程中的 KV，最终只要编辑分支的预测结果
        _, edit_noise = paired_noise.chunk(2, dim=0)
        return edit_noise

    def _predict_x0_from_noisy(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        *,
        recon_cond: Optional[torch.Tensor] = None,
        recon_sample: Optional[torch.Tensor] = None, # [新增参数] 传入纯净的重建样本
        use_contextflow: bool = False,
    ) -> torch.Tensor:
        """
        根据带噪图像 z_t 和预测出的噪声，反推回 t=0 时的清晰图像 x_0。
        """
        alpha_t, sigma_t = self._get_alpha_sigma(z_t, timesteps)
        noise_pred = self._predict_noise(
            sample=z_t,
            timesteps=timesteps,
            cond=cond,
            recon_cond=recon_cond,
            recon_sample=recon_sample, # 向下传递给 _predict_noise
            use_contextflow=use_contextflow,
        )
        # 这是扩散模型的核心公式：x0 = (z_t - sigma_t * noise_pred) / alpha_t
        return (z_t - sigma_t * noise_pred) / alpha_t

#_u_estimate 函数：计算修改方向
    def _u_estimate(
        self,
        x_src,       # [新增] 永远纯净的原图 (专供重建分支提取KV)
        x_anchor,    # 当前带有编辑痕迹的图 (专供编辑分支)
        src_embed,
        edit_embed,
        noise,
        t_s: float,
        delta: float,
        *,
        use_contextflow: bool = False,
    ):
        """
        在 Latent 空间中，计算当前时间步的编辑位移向量 u_hat。
        物理含义: 指向“目标文本语义”减去“原始文本语义”的方向，告诉模型下一步该往哪走。
        """
        # 1. 获取基础元信息：batch_size 以及设备信息 (确保计算在同一块 GPU 上)
        batch, device = x_anchor.shape[0], x_anchor.device  #x_anchor: 当前图像在 Latent 空间的特征,Shape [1, 4, 64, 64]
        # 2. 将 [0,1] 的浮点时间戳转为扩散模型能够识别的整数步数 (如 0-999)
        t_idx_s = self._time_to_index(batch, t_s, device=device) #t_idx_s:当前时间步的“模糊程度”,t_s：当前时间步，1表示纯噪声，0表示干净图像
        t_idx_s0 = self._time_to_index(batch, max(0.0, t_s - delta), device=device) #t_idx_s0: 向前平移 delta 后的时间步索引,t_s - delta：稍微"更干净"的时间步

        # 3. 向量化处理所有噪声样本，消除 for 循环，大幅提升效率
        if isinstance(noise, (list, tuple)):
            noise_tensor = torch.cat(noise, dim=0) # [num_noises, 4, 64, 64]
        else:
            noise_tensor = noise
            
        num_noises = noise_tensor.shape[0]
        
        # 沿着 Batch 维度扩展所有输入张量，以便一次性并行计算
        # 使用 expand 代替 repeat 优化内存占用
        x_anchor_batch = x_anchor.expand(num_noises, -1, -1, -1)     # [num_noises, 4, 64, 64]
        x_src_batch = x_src.expand(num_noises, -1, -1, -1)           # [新增] 扩展纯净原图
        src_embed_batch = src_embed.expand(num_noises, -1, -1)      # [num_noises, 77, 768]
        edit_embed_batch = edit_embed.expand(num_noises, -1, -1)    # [num_noises, 77, 768]
        
        t_idx_s_batch = t_idx_s.expand(num_noises)
        t_idx_s0_batch = t_idx_s0.expand(num_noises)

        # 4. 获取扩散系数 (Schedulers Parameters):
        # alpha: 原始信息的保留比例；sigma: 噪声的添加比例
        alpha_s, sigma_s = self._get_alpha_sigma(x_anchor_batch, t_idx_s_batch)
        alpha_prev, sigma_prev = self._get_alpha_sigma(x_anchor_batch, t_idx_s0_batch)

        # 5. 并行计算加噪状态 (当前编辑图)
        z_s_curr = alpha_s * x_anchor_batch + sigma_s * noise_tensor
        z_prev_curr = alpha_prev * x_anchor_batch + sigma_prev * noise_tensor
        
        # [新增] 并行计算加噪状态 (纯净原图，专供重建分支)
        z_s_pure = alpha_s * x_src_batch + sigma_s * noise_tensor
        z_prev_pure = alpha_prev * x_src_batch + sigma_prev * noise_tensor
        
        # 6. 并行预测 x0 (一次网络前向传播处理所有噪声)
        x_src_p_s = self._predict_x0_from_noisy(z_s_curr, t_idx_s_batch, src_embed_batch, recon_sample=None, use_contextflow=False)
        x_tar_p_s = self._predict_x0_from_noisy(z_s_curr, t_idx_s_batch, edit_embed_batch, recon_cond=src_embed_batch, recon_sample=z_s_pure, use_contextflow=use_contextflow)
        
        x_src_p_s0 = self._predict_x0_from_noisy(z_prev_curr, t_idx_s0_batch, src_embed_batch, recon_sample=None, use_contextflow=False)
        x_tar_p_s0 = self._predict_x0_from_noisy(z_prev_curr, t_idx_s0_batch, edit_embed_batch, recon_cond=src_embed_batch, recon_sample=z_prev_pure, use_contextflow=use_contextflow)
        
        # 7. 计算方向并沿着 num_noises 维度求平均
        dv_s_batch = x_tar_p_s - x_src_p_s
        dv_s0_batch = x_tar_p_s0 - x_src_p_s0
        
        dv_s = dv_s_batch.mean(dim=0, keepdim=True)
        dv_s0 = dv_s0_batch.mean(dim=0, keepdim=True)
       

    # 8. 时间修正与加权平衡：
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
        # 创建从 t_start 到 t_end 的时间步网格
        if params["n_steps"] == 1:
            t_grid = [params["t_start"]] 
        else:
            t_grid = torch.linspace(
                params["t_start"],      # ← 从高噪声开始（编辑广度）
                params["t_end"],        # ← 到低噪声结束（细节保真）
                steps=params["n_steps"],
                device=device,
            ).tolist()
      # 在 Latent 空间中，以 x_src 为起点，一步步走向目标语义。
        x_curr = x_src #从原始latent开始
        total_steps = max(1, len(t_grid))
        for step_idx, t_s in enumerate(t_grid): #在多个时间步上迭代
            normalized_progress = float(step_idx) / float(total_steps)
            use_contextflow = (
                bool(params.get("contextflow_enabled", False))
                and normalized_progress < float(params.get("contextflow_threshold_tau", self._contextflow_threshold_tau))
            )
        #u_hat: 当前时间步的修改方向，编辑位移向量
        #物理含义：在latent空间中，指向“目标文本语义” 减去“原始文本语义” 的方向
            u_hat = self._u_estimate(  # u_hat：这一步要走的方向和距离,是一个【方向向量】，指向目标文本的语义方向
                x_src,     # [新增] 永远纯净的原图 (专供重建分支提取KV)
                x_curr,    # 当前latent
                src_embed, #原始描述
                edit_embed, #目标描述
                noise,     #噪声样本
                float(t_s),  #当前时间
                params["t_delta"],  #时间差分
                use_contextflow=use_contextflow,
            )
            # 状态演变: x_{new} = x_{old} + 步长 * 位移向量
            x_curr = x_curr + params["step_scale"] * u_hat #更新当前图像特征,

        if params["cleanup"]:
            # [收尾清理阶段]
            # 计算目的: 利用目标文本引导，直接预测出时间步为 0 的干净状态 $x_0$。
            # 物理含义: 消除累积的截断误差，强制将 Latent 对齐到清晰的图像流形 (Image Manifold) 上。
            t_end_idx = self._time_to_index(x_src.shape[0], params["t_end"], device=device)
            x_curr = self._pred_x0(x_curr, t_end_idx, edit_embed, noise[0]) #_pred_x0 的作用:利用 UNet 预测出完全不含噪声的原始状态，把最后一点模糊感去掉，让导出的图片边缘更清晰、纹理更真实

        return x_curr #返回编辑后的latent
