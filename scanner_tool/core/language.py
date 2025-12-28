"""
语言查询模块

支持:
- 文本到特征编码
- 基于文本的语义分割
- 开放词汇分割

复用来源:
- autolabel/scripts/language/evaluate.py
- autolabel/scripts/language/pointcloud.py
- autolabel/autolabel/utils/feature_utils.py

Requirements: 15.1, 15.2, 15.3, 15.4
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

# 可选依赖
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


@dataclass
class LanguageConfig:
    """语言查询配置"""
    feature_type: str = 'lseg'  # 'lseg' or 'clip'
    checkpoint: Optional[str] = None
    temperature: float = 0.07  # CLIP 温度参数


@dataclass
class LanguageQueryResult:
    """语言查询结果"""
    success: bool
    segmentation: Optional[np.ndarray] = None  # [H, W] 类别索引
    similarity_map: Optional[np.ndarray] = None  # [H, W, C] 相似度
    class_names: List[str] = field(default_factory=list)
    error_message: str = ""


def check_language_available() -> Dict[str, bool]:
    """检查语言查询功能可用性"""
    cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    
    return {
        'torch': TORCH_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'h5py': H5PY_AVAILABLE,
        'cuda': cuda_available,
        # LSeg 需要 CUDA
        'lseg': cuda_available,
        # CLIP 可以在 CPU/MPS 上运行
        'clip': TORCH_AVAILABLE,
        'available': TORCH_AVAILABLE and PIL_AVAILABLE
    }


def _get_device():
    """获取计算设备"""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class TextEncoder:
    """文本编码器基类"""
    
    def __init__(self, config: LanguageConfig = None):
        self.config = config or LanguageConfig()
        self.device = _get_device()
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            特征向量 [N, D]
        """
        raise NotImplementedError


class CLIPTextEncoder(TextEncoder):
    """CLIP 文本编码器
    
    使用 OpenAI CLIP 模型编码文本。
    跨平台支持 (CUDA/MPS/CPU)。
    """
    
    def __init__(self, config: LanguageConfig = None):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        try:
            import clip
            return True
        except ImportError:
            return False
    
    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return
        
        try:
            import clip
            self.model, _ = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        if not self.is_available:
            raise RuntimeError("CLIP not available. Install with: pip install clip")
        
        self._load_model()
        
        import clip
        
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


class LSegTextEncoder(TextEncoder):
    """LSeg 文本编码器
    
    使用 LSeg 模型编码文本。
    仅支持 CUDA。
    """
    
    def __init__(self, config: LanguageConfig = None):
        super().__init__(config)
        self.model = None
        self._loaded = False
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        
        # 检查 LSeg 是否可用
        try:
            from autolabel.lseg import LSegNet
            return True
        except ImportError:
            return False
    
    def _load_model(self):
        """延迟加载模型"""
        if self._loaded:
            return
        
        if self.config.checkpoint is None:
            raise ValueError("LSeg requires checkpoint path")
        
        try:
            from autolabel.lseg import LSegNet
            
            self.model = LSegNet(
                backbone='clip_vitl16_384',
                features=256,
                crop_size=480,
                arch_option=0,
                block_depth=0,
                activation='lrelu'
            )
            self.model.load_state_dict(
                torch.load(self.config.checkpoint, map_location='cpu')
            )
            self.model = self.model.cuda().eval()
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load LSeg model: {e}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本"""
        if not self.is_available:
            raise RuntimeError("LSeg not available (requires CUDA)")
        
        self._load_model()
        
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()


class LanguageSegmenter:
    """基于语言的分割器
    
    使用文本提示进行语义分割。
    """
    
    def __init__(self, config: LanguageConfig = None):
        self.config = config or LanguageConfig()
        self.device = _get_device()
        
        # 选择文本编码器
        if self.config.feature_type == 'clip':
            self.text_encoder = CLIPTextEncoder(self.config)
        else:
            self.text_encoder = LSegTextEncoder(self.config)
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return self.text_encoder.is_available
    
    def segment_with_features(self, features: np.ndarray,
                              prompts: List[str]) -> LanguageQueryResult:
        """使用预提取的特征进行分割
        
        Args:
            features: 图像特征 [H, W, D]
            prompts: 文本提示列表
            
        Returns:
            分割结果
        """
        if not self.is_available:
            return LanguageQueryResult(
                success=False,
                error_message=f"{self.config.feature_type} encoder not available"
            )
        
        try:
            # 编码文本
            text_features = self.text_encoder.encode(prompts)  # [C, D]
            
            H, W, D = features.shape
            C = len(prompts)
            
            # 归一化图像特征
            features_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
            
            # 计算相似度
            similarity = np.zeros((H, W, C), dtype=np.float32)
            for i in range(C):
                similarity[:, :, i] = (features_norm * text_features[i]).sum(axis=-1)
            
            # 应用温度缩放
            similarity = similarity / self.config.temperature
            
            # 获取分割结果
            segmentation = similarity.argmax(axis=-1)
            
            return LanguageQueryResult(
                success=True,
                segmentation=segmentation,
                similarity_map=similarity,
                class_names=prompts
            )
            
        except Exception as e:
            return LanguageQueryResult(
                success=False,
                error_message=str(e)
            )
    
    def segment_scene(self, scene_path: str,
                      prompts: List[str],
                      output_path: str = None) -> LanguageQueryResult:
        """分割整个场景
        
        Args:
            scene_path: 场景目录
            prompts: 文本提示列表
            output_path: 输出目录 (可选)
            
        Returns:
            分割结果
        """
        scene_path = Path(scene_path)
        features_file = scene_path / 'features.hdf'
        
        if not features_file.exists():
            return LanguageQueryResult(
                success=False,
                error_message=f"Features file not found: {features_file}"
            )
        
        if not H5PY_AVAILABLE:
            return LanguageQueryResult(
                success=False,
                error_message="h5py not available"
            )
        
        try:
            # 编码文本
            text_features = self.text_encoder.encode(prompts)  # [C, D]
            
            # 读取特征并分割
            with h5py.File(features_file, 'r') as f:
                feature_type = self.config.feature_type
                if f'features/{feature_type}' not in f:
                    return LanguageQueryResult(
                        success=False,
                        error_message=f"Feature type '{feature_type}' not found in features.hdf"
                    )
                
                features_group = f[f'features/{feature_type}']
                frame_names = list(features_group.keys())
                
                if output_path:
                    output_path = Path(output_path)
                    output_path.mkdir(parents=True, exist_ok=True)
                
                for frame_name in frame_names:
                    if frame_name in ['pca', 'min', 'range']:
                        continue
                    
                    features = features_group[frame_name][:]  # [H, W, D]
                    
                    # 归一化
                    features_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
                    
                    # 计算相似度
                    H, W, D = features.shape
                    C = len(prompts)
                    similarity = np.zeros((H, W, C), dtype=np.float32)
                    for i in range(C):
                        similarity[:, :, i] = (features_norm * text_features[i]).sum(axis=-1)
                    
                    # 获取分割
                    segmentation = similarity.argmax(axis=-1).astype(np.uint8)
                    
                    # 保存
                    if output_path and PIL_AVAILABLE:
                        Image.fromarray(segmentation).save(
                            output_path / f"{frame_name}.png"
                        )
            
            return LanguageQueryResult(
                success=True,
                class_names=prompts
            )
            
        except Exception as e:
            return LanguageQueryResult(
                success=False,
                error_message=str(e)
            )


def get_text_encoder(feature_type: str = 'clip',
                     checkpoint: str = None) -> TextEncoder:
    """获取文本编码器
    
    Args:
        feature_type: 'clip' 或 'lseg'
        checkpoint: LSeg 检查点路径 (LSeg 必需)
        
    Returns:
        文本编码器实例
    """
    config = LanguageConfig(
        feature_type=feature_type,
        checkpoint=checkpoint
    )
    
    if feature_type == 'clip':
        return CLIPTextEncoder(config)
    else:
        return LSegTextEncoder(config)
