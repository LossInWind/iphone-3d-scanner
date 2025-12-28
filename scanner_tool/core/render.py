"""
NeRF 渲染模块

复用 autolabel/scripts/render.py 的功能。

功能:
- 从训练好的 NeRF 模型渲染视频
- 支持 RGB、深度、语义分割的可视化
- 支持开放词汇的语义渲染

Requirements: CUDA (NVIDIA GPU)
"""

import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

# autolabel 路径
AUTOLABEL_PATH = Path(__file__).parent.parent / 'autolabel'
if str(AUTOLABEL_PATH) not in sys.path:
    sys.path.insert(0, str(AUTOLABEL_PATH))

# 可选依赖
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def _check_cuda():
    """检查 CUDA 是否可用"""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


CUDA_AVAILABLE = _check_cuda()


@dataclass
class RenderConfig:
    """渲染配置"""
    fps: int = 5
    stride: int = 1  # 每 N 帧渲染一次
    max_depth: float = 7.5  # 深度可视化最大值
    size: Tuple[int, int] = (960, 720)  # 输出视频尺寸
    batch_size: int = 16384


@dataclass
class RenderResult:
    """渲染结果"""
    success: bool
    output_path: str = ""
    num_frames: int = 0
    error_message: str = ""


def check_render_available() -> Dict[str, bool]:
    """检查渲染功能可用性"""
    autolabel_available = False
    try:
        from autolabel.dataset import SceneDataset
        from autolabel import model_utils
        autolabel_available = True
    except ImportError:
        pass
    
    return {
        'torch': TORCH_AVAILABLE,
        'cuda': CUDA_AVAILABLE,
        'cv2': CV2_AVAILABLE,
        'h5py': H5PY_AVAILABLE,
        'autolabel': autolabel_available,
        'available': CUDA_AVAILABLE and CV2_AVAILABLE and autolabel_available
    }


class FeatureTransformer:
    """特征可视化转换器"""
    
    def __init__(self, scene_path: str, feature_name: str, 
                 classes: List[str] = None, checkpoint: str = None):
        self.pca = None
        self.feature_min = None
        self.feature_range = None
        self.text_features = None
        
        # 加载 PCA 参数
        features_file = os.path.join(scene_path, 'features.hdf')
        if os.path.exists(features_file) and H5PY_AVAILABLE:
            with h5py.File(features_file, 'r') as f:
                if f'features/{feature_name}' in f:
                    features = f[f'features/{feature_name}']
                    if 'pca' in features.attrs:
                        blob = features.attrs['pca'].tobytes()
                        self.pca = pickle.loads(blob)
                    if 'min' in features.attrs:
                        self.feature_min = features.attrs['min']
                    if 'range' in features.attrs:
                        self.feature_range = features.attrs['range']
        
        # 编码文本
        if feature_name is not None and classes is not None and checkpoint is not None:
            from autolabel.utils.feature_utils import get_feature_extractor
            extractor = get_feature_extractor(feature_name, checkpoint)
            self.text_features = extractor.encode_text(classes)
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """将特征转换为 RGB 可视化"""
        if self.pca is None:
            return np.zeros((*features.shape[:2], 3), dtype=np.uint8)
        
        H, W, C = features.shape
        features_flat = self.pca.transform(features.reshape(H * W, C))
        
        if self.feature_min is not None and self.feature_range is not None:
            features_flat = np.clip(
                (features_flat - self.feature_min) / self.feature_range,
                0., 1.
            )
        
        return (features_flat.reshape(H, W, 3) * 255.).astype(np.uint8)


class NeRFRenderer:
    """NeRF 渲染器"""
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self.model = None
        self.dataset = None
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        availability = check_render_available()
        return availability['available']
    
    def _get_nerf_dir(self, scene_path: str, workspace: str = None) -> str:
        """获取 NeRF 模型目录"""
        if workspace is None:
            return os.path.join(scene_path, 'nerf')
        else:
            scene_name = os.path.basename(os.path.normpath(scene_path))
            return os.path.join(workspace, scene_name)
    
    def _find_model(self, nerf_dir: str) -> Optional[str]:
        """查找训练好的模型"""
        if not os.path.exists(nerf_dir):
            return None
        
        for model_name in os.listdir(nerf_dir):
            checkpoint_dir = os.path.join(nerf_dir, model_name, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                return model_name
        return None
    
    def _load_model(self, scene_path: str, model_dir: str = None):
        """加载 NeRF 模型"""
        from autolabel.dataset import SceneDataset
        from autolabel import model_utils
        
        if model_dir is None:
            nerf_dir = self._get_nerf_dir(scene_path)
            model_name = self._find_model(nerf_dir)
            if model_name is None:
                raise FileNotFoundError(f"No trained model found in {nerf_dir}")
            model_dir = os.path.join(nerf_dir, model_name)
        
        params = model_utils.read_params(model_dir)
        
        # 计算数据集尺寸
        half_size = (self.config.size[0] // 2, self.config.size[1] // 2)
        
        # 创建数据集
        self.dataset = SceneDataset(
            'test',
            scene_path,
            size=half_size,
            batch_size=self.config.batch_size,
            features=params.features,
            load_semantic=False,
            lazy=True
        )
        
        # 创建模型
        n_classes = self.dataset.n_classes if self.dataset.n_classes is not None else 2
        self.model = model_utils.create_model(
            self.dataset.min_bounds,
            self.dataset.max_bounds,
            n_classes,
            params
        ).cuda()
        
        # 加载检查点
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        model_utils.load_checkpoint(self.model, checkpoint_dir)
        self.model = self.model.eval()
        
        return params
    
    def _visualize_depth(self, depth: np.ndarray, max_depth: float = None) -> np.ndarray:
        """深度可视化"""
        max_depth = max_depth or self.config.max_depth
        depth_normalized = np.clip(depth / max_depth, 0, 1)
        depth_colored = (plt.cm.viridis(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        return depth_colored
    
    def _render_frame(self, frame_index: int, 
                      feature_transform: FeatureTransformer = None,
                      classes: List[str] = None) -> np.ndarray:
        """渲染单帧"""
        from autolabel.constants import COLORS
        
        batch = self.dataset._get_test(frame_index)
        
        rays_o = torch.tensor(batch['rays_o']).cuda()
        rays_d = torch.tensor(batch['rays_d']).cuda()
        direction_norms = torch.tensor(batch['direction_norms']).cuda()
        
        outputs = self.model.render(
            rays_o,
            rays_d,
            direction_norms,
            staged=True,
            perturb=False,
            num_steps=512,
            upsample_steps=0
        )
        
        # 计算语义分割
        if classes is not None and feature_transform is not None and feature_transform.text_features is not None:
            features = outputs['semantic_features']
            features = features / torch.norm(features, dim=-1, keepdim=True)
            text_features = torch.tensor(feature_transform.text_features).cuda()
            H, W, D = features.shape
            C = text_features.shape[0]
            similarities = torch.zeros((H, W, C), dtype=features.dtype)
            for i in range(H):
                similarities[i, :, :] = (features[i, :, None] * text_features).sum(dim=-1).cpu()
            p_semantic = similarities.argmax(dim=-1).numpy()
        else:
            p_semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
        
        # 组合输出帧 (2x2 网格)
        frame = np.zeros((self.config.size[1], self.config.size[0], 3), dtype=np.uint8)
        half_w, half_h = self.config.size[0] // 2, self.config.size[1] // 2
        
        # 左上: RGB
        p_rgb = (outputs['image'].cpu().numpy() * 255.0).astype(np.uint8)
        frame[:half_h, :half_w, :] = p_rgb
        
        # 右上: 深度
        p_depth = outputs['depth'].cpu().numpy()
        depth_vis = self._visualize_depth(p_depth)
        frame[:half_h, half_w:] = depth_vis
        
        # 左下: 语义分割
        frame[half_h:, :half_w] = COLORS[p_semantic % COLORS.shape[0]]
        
        # 右下: 特征可视化
        if feature_transform is not None and 'semantic_features' in outputs:
            p_features = feature_transform.transform_features(
                outputs['semantic_features'].cpu().numpy()
            )
            frame[half_h:, half_w:] = p_features
        
        return frame
    
    def render_video(
        self,
        scene_path: str,
        output_path: str,
        model_dir: str = None,
        classes: List[str] = None,
        feature_checkpoint: str = None
    ) -> RenderResult:
        """渲染视频
        
        Args:
            scene_path: 场景目录路径
            output_path: 输出视频文件路径 (.mp4)
            model_dir: NeRF 模型目录 (可选)
            classes: 语义类别列表 (用于开放词汇分割)
            feature_checkpoint: 特征提取器检查点
            
        Returns:
            渲染结果
        """
        if not self.is_available:
            return RenderResult(
                success=False,
                error_message="Rendering requires CUDA + OpenCV + autolabel"
            )
        
        try:
            from tqdm import tqdm
            import matplotlib.pyplot as plt
            
            # 加载模型
            print("Loading NeRF model...")
            params = self._load_model(scene_path, model_dir)
            
            # 创建特征转换器
            feature_transform = None
            if params.features is not None:
                feature_transform = FeatureTransformer(
                    scene_path,
                    params.features,
                    classes,
                    feature_checkpoint
                )
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                self.config.size
            )
            
            # 渲染帧
            print("Rendering frames...")
            frame_indices = self.dataset.indices[::self.config.stride]
            num_frames = 0
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    for frame_index in tqdm(frame_indices):
                        frame = self._render_frame(
                            frame_index,
                            feature_transform,
                            classes
                        )
                        # OpenCV 使用 BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
                        num_frames += 1
            
            writer.release()
            print(f"Video saved to: {output_path}")
            
            return RenderResult(
                success=True,
                output_path=output_path,
                num_frames=num_frames
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return RenderResult(
                success=False,
                error_message=str(e)
            )


# 为了兼容 matplotlib 的深度可视化
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
