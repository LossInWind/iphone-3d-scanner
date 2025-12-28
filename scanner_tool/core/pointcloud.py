"""
3D 点云语义分割模块

复用 autolabel/scripts/language/pointcloud.py 的功能。

功能:
- 从训练好的 NeRF 模型提取 3D 点云
- 使用文本提示进行开放词汇的 3D 语义分割
- 导出带语义标签的点云

Requirements: CUDA (NVIDIA GPU)
"""

import os
import sys
import math
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
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def _check_cuda():
    """检查 CUDA 是否可用"""
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


CUDA_AVAILABLE = _check_cuda()


@dataclass
class PointCloudConfig:
    """点云分割配置"""
    batch_size: int = 8192
    stride: int = 1  # 每 N 帧采样一次
    features: str = 'lseg'  # 特征类型
    variance_percentile: float = 50.0  # 方差过滤百分位


@dataclass
class PointCloudResult:
    """点云分割结果"""
    success: bool
    output_path: str = ""
    num_points: int = 0
    class_names: List[str] = field(default_factory=list)
    error_message: str = ""


def check_pointcloud_available() -> Dict[str, bool]:
    """检查点云分割功能可用性"""
    autolabel_available = False
    try:
        from autolabel.dataset import SceneDataset
        from autolabel import model_utils
        from autolabel.utils.feature_utils import get_feature_extractor
        autolabel_available = True
    except ImportError:
        pass
    
    return {
        'torch': TORCH_AVAILABLE,
        'cuda': CUDA_AVAILABLE,
        'open3d': OPEN3D_AVAILABLE,
        'autolabel': autolabel_available,
        'available': CUDA_AVAILABLE and OPEN3D_AVAILABLE and autolabel_available
    }


class PointCloudSegmenter:
    """3D 点云语义分割器
    
    使用训练好的 NeRF 模型提取 3D 点云，
    并使用文本提示进行开放词汇的语义分割。
    """
    
    def __init__(self, config: PointCloudConfig = None):
        self.config = config or PointCloudConfig()
        self.model = None
        self.dataset = None
        self.feature_extractor = None
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        availability = check_pointcloud_available()
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
    
    def _load_model(self, scene_path: str, workspace: str = None):
        """加载 NeRF 模型"""
        from autolabel.dataset import SceneDataset
        from autolabel import model_utils
        
        nerf_dir = self._get_nerf_dir(scene_path, workspace)
        model_name = self._find_model(nerf_dir)
        
        if model_name is None:
            raise FileNotFoundError(f"No trained model found in {nerf_dir}")
        
        model_path = os.path.join(nerf_dir, model_name)
        params = model_utils.read_params(model_path)
        
        # 创建数据集
        self.dataset = SceneDataset(
            'test',
            scene_path,
            factor=4.0,
            batch_size=self.config.batch_size,
            lazy=True
        )
        
        # 创建模型
        self.model = model_utils.create_model(
            self.dataset.min_bounds,
            self.dataset.max_bounds,
            606,  # 最大类别数
            params
        ).cuda()
        
        # 加载检查点
        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        model_utils.load_checkpoint(self.model, checkpoint_dir)
        self.model = self.model.eval()
        
        return params
    
    def _load_feature_extractor(self, checkpoint: str):
        """加载特征提取器"""
        from autolabel.utils.feature_utils import get_feature_extractor
        self.feature_extractor = get_feature_extractor(
            self.config.features,
            checkpoint
        )
    
    def _render_frame(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """渲染单帧，返回 3D 点和 RGB 颜色"""
        batch = self.dataset._get_test(frame_index)
        
        rays_o = torch.tensor(batch['rays_o']).cuda()
        rays_d = torch.tensor(batch['rays_d']).cuda()
        direction_norms = torch.tensor(batch['direction_norms']).cuda()
        
        output = self.model.render(
            rays_o,
            rays_d,
            direction_norms,
            staged=True,
            perturb=False,
            num_steps=512,
            upsample_steps=0
        )
        
        # 使用方差过滤不确定的点
        variance = output['depth_variance'].cpu().numpy()
        cutoff = np.percentile(variance, self.config.variance_percentile)
        mask = variance < cutoff
        
        # 提取 3D 坐标和颜色
        coordinates = output['coordinates_map'].cpu().numpy()[mask]
        rgb = output['image'].cpu().numpy()[mask]
        
        return coordinates[:, :3], rgb
    
    def extract_pointcloud(
        self,
        scene_path: str,
        output_path: str,
        workspace: str = None,
        feature_checkpoint: str = None,
        prompts: List[str] = None,
        visualize: bool = False
    ) -> PointCloudResult:
        """提取 3D 点云
        
        Args:
            scene_path: 场景目录路径
            output_path: 输出点云文件路径 (.ply)
            workspace: NeRF 模型工作目录
            feature_checkpoint: 特征提取器检查点 (用于语义分割)
            prompts: 文本提示列表 (用于语义分割)
            visualize: 是否可视化结果
            
        Returns:
            点云提取结果
        """
        if not self.is_available:
            return PointCloudResult(
                success=False,
                error_message="Point cloud segmentation requires CUDA + Open3D + autolabel"
            )
        
        try:
            from tqdm import tqdm
            
            # 加载模型
            print("Loading NeRF model...")
            params = self._load_model(scene_path, workspace)
            
            # 加载特征提取器 (如果需要语义分割)
            if feature_checkpoint and prompts:
                print("Loading feature extractor...")
                self._load_feature_extractor(feature_checkpoint)
            
            # 提取点云
            print("Extracting point cloud...")
            all_points = []
            all_colors = []
            
            frame_indices = self.dataset.indices[::self.config.stride]
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    for frame_index in tqdm(frame_indices):
                        points, colors = self._render_frame(frame_index)
                        all_points.append(points)
                        all_colors.append(colors)
            
            # 合并点云
            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            
            # 创建 Open3D 点云
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pc.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存点云
            o3d.io.write_point_cloud(output_path, pc)
            print(f"Point cloud saved to: {output_path}")
            
            # 可视化
            if visualize:
                o3d.visualization.draw_geometries([pc])
            
            return PointCloudResult(
                success=True,
                output_path=output_path,
                num_points=len(points),
                class_names=prompts or []
            )
            
        except Exception as e:
            return PointCloudResult(
                success=False,
                error_message=str(e)
            )
    
    def segment_pointcloud(
        self,
        scene_path: str,
        output_path: str,
        prompts: List[str],
        feature_checkpoint: str,
        workspace: str = None,
        visualize: bool = False
    ) -> PointCloudResult:
        """对点云进行语义分割
        
        使用文本提示对 3D 点云进行开放词汇的语义分割。
        
        Args:
            scene_path: 场景目录路径
            output_path: 输出点云文件路径 (.ply)
            prompts: 文本提示列表 (如 ["chair", "table", "floor"])
            feature_checkpoint: 特征提取器检查点
            workspace: NeRF 模型工作目录
            visualize: 是否可视化结果
            
        Returns:
            点云分割结果
        """
        if not self.is_available:
            return PointCloudResult(
                success=False,
                error_message="Point cloud segmentation requires CUDA + Open3D + autolabel"
            )
        
        try:
            from tqdm import tqdm
            from autolabel.constants import COLORS
            
            # 加载模型
            print("Loading NeRF model...")
            params = self._load_model(scene_path, workspace)
            
            # 加载特征提取器
            print("Loading feature extractor...")
            self._load_feature_extractor(feature_checkpoint)
            
            # 编码文本提示
            print("Encoding text prompts...")
            text_features = self.feature_extractor.encode_text(prompts)
            text_features = torch.tensor(text_features).cuda()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 提取点云和语义特征
            print("Extracting point cloud with semantic features...")
            all_points = []
            all_semantics = []
            
            frame_indices = self.dataset.indices[::self.config.stride]
            
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    for frame_index in tqdm(frame_indices):
                        batch = self.dataset._get_test(frame_index)
                        
                        rays_o = torch.tensor(batch['rays_o']).cuda()
                        rays_d = torch.tensor(batch['rays_d']).cuda()
                        direction_norms = torch.tensor(batch['direction_norms']).cuda()
                        
                        output = self.model.render(
                            rays_o,
                            rays_d,
                            direction_norms,
                            staged=True,
                            perturb=False,
                            num_steps=512,
                            upsample_steps=0
                        )
                        
                        # 方差过滤
                        variance = output['depth_variance'].cpu().numpy()
                        cutoff = np.percentile(variance, self.config.variance_percentile)
                        mask = variance < cutoff
                        
                        # 提取坐标
                        coordinates = output['coordinates_map'].cpu().numpy()[mask]
                        points = coordinates[:, :3]
                        
                        # 提取语义特征并计算相似度
                        if 'semantic_features' in output:
                            features = output['semantic_features'][mask]
                            features = features / features.norm(dim=-1, keepdim=True)
                            
                            # 计算与每个文本的相似度
                            similarities = torch.matmul(features, text_features.T)
                            semantics = similarities.argmax(dim=-1).cpu().numpy()
                        else:
                            # 如果没有语义特征，使用默认类别
                            semantics = np.zeros(len(points), dtype=np.int32)
                        
                        all_points.append(points)
                        all_semantics.append(semantics)
            
            # 合并
            points = np.concatenate(all_points, axis=0)
            semantics = np.concatenate(all_semantics, axis=0)
            
            # 根据语义类别着色
            colors = COLORS[semantics % len(COLORS)] / 255.0
            
            # 创建点云
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            pc.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存
            o3d.io.write_point_cloud(output_path, pc)
            print(f"Segmented point cloud saved to: {output_path}")
            
            # 打印统计
            print("\nClass distribution:")
            for i, prompt in enumerate(prompts):
                count = (semantics == i).sum()
                percentage = count / len(semantics) * 100
                print(f"  {prompt}: {count} points ({percentage:.1f}%)")
            
            # 可视化
            if visualize:
                o3d.visualization.draw_geometries([pc])
            
            return PointCloudResult(
                success=True,
                output_path=output_path,
                num_points=len(points),
                class_names=prompts
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return PointCloudResult(
                success=False,
                error_message=str(e)
            )
