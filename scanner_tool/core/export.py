"""
导出模块

支持:
- 语义分割图导出
- 视频渲染
- 格式转换 (Open3D, instant-ngp)

复用来源:
- autolabel/scripts/export.py
- autolabel/scripts/render.py
- autolabel/scripts/convert_to_instant_ngp.py

Requirements: 12.1, 12.2, 12.3, 13.1, 13.2, 13.3, 16.1, 16.2, 16.3
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

# 可选依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from skvideo.io.ffmpeg import FFmpegWriter
    SKVIDEO_AVAILABLE = True
except ImportError:
    SKVIDEO_AVAILABLE = False


# 默认颜色表
DEFAULT_COLORS = np.array([
    [0, 0, 0],        # 背景
    [255, 0, 0],      # 类别 1
    [0, 255, 0],      # 类别 2
    [0, 0, 255],      # 类别 3
    [255, 255, 0],    # 类别 4
    [255, 0, 255],    # 类别 5
    [0, 255, 255],    # 类别 6
    [128, 0, 0],      # 类别 7
    [0, 128, 0],      # 类别 8
    [0, 0, 128],      # 类别 9
    [128, 128, 0],    # 类别 10
], dtype=np.uint8)


@dataclass
class ExportConfig:
    """导出配置"""
    max_width: int = 640
    objects_per_class: Optional[int] = None  # 每类最大连通域数
    colors: np.ndarray = field(default_factory=lambda: DEFAULT_COLORS)


@dataclass
class ExportResult:
    """导出结果"""
    success: bool
    output_path: str
    num_frames: int = 0
    error_message: str = ""


def check_export_available() -> Dict[str, bool]:
    """检查导出功能可用性"""
    basic = CV2_AVAILABLE or PIL_AVAILABLE
    return {
        'cv2': CV2_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'skimage': SKIMAGE_AVAILABLE,
        'torch': TORCH_AVAILABLE,
        'skvideo': SKVIDEO_AVAILABLE,
        'basic': basic,
        'video': SKVIDEO_AVAILABLE,
        'available': basic,  # 基本功能可用
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


class SemanticExporter:
    """语义分割图导出器
    
    从训练好的 NeRF 模型导出语义分割图。
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.device = _get_device()
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return TORCH_AVAILABLE and (CV2_AVAILABLE or PIL_AVAILABLE)
    
    def find_largest_components(self, semantic_map: np.ndarray, 
                                class_id: int, 
                                object_count: int) -> List[np.ndarray]:
        """找到指定类别的最大连通域
        
        Args:
            semantic_map: 语义分割图 [H, W]
            class_id: 类别 ID
            object_count: 要保留的连通域数量
            
        Returns:
            连通域掩码列表
        """
        if not SKIMAGE_AVAILABLE:
            return [semantic_map == class_id]
        
        mask = semantic_map.copy()
        mask[mask != class_id] = 0
        labels = measure.label(mask)
        counts = np.bincount(labels.flat)[1:]
        
        if len(counts) == 0:
            return []
        
        largest = []
        sorted_counts = np.argsort(counts)[::-1]
        for i in range(min(object_count, len(sorted_counts))):
            nth_largest_label = sorted_counts[i] + 1
            largest.append(labels == nth_largest_label)
        return largest
    
    def post_process(self, semantic_map: np.ndarray) -> np.ndarray:
        """后处理语义图 - 保留每类最大连通域
        
        Args:
            semantic_map: 语义分割图 [H, W]
            
        Returns:
            处理后的语义图
        """
        if self.config.objects_per_class is None:
            return semantic_map
        
        out = np.zeros_like(semantic_map)
        class_ids = np.unique(semantic_map)
        
        for class_id in class_ids:
            if class_id == 0:
                continue  # 跳过背景
            components = self.find_largest_components(
                semantic_map, class_id, self.config.objects_per_class
            )
            for component in components:
                out[component] = class_id
        
        return out
    
    def export_from_model(self, scene_path: str, 
                          model_dir: str = None,
                          output_path: str = None) -> ExportResult:
        """从模型导出语义分割图
        
        Args:
            scene_path: 场景目录
            model_dir: 模型目录 (默认: <scene>/nerf/)
            output_path: 输出目录 (默认: <scene>/output/semantic/)
            
        Returns:
            导出结果
        """
        if not self.is_available:
            return ExportResult(
                success=False,
                output_path="",
                error_message="Export dependencies not available"
            )
        
        scene_path = Path(scene_path)
        
        # 确定模型目录
        if model_dir is None:
            model_dir = scene_path / 'nerf'
        else:
            model_dir = Path(model_dir)
        
        if not model_dir.exists():
            return ExportResult(
                success=False,
                output_path="",
                error_message=f"Model directory not found: {model_dir}"
            )
        
        # 确定输出目录
        if output_path is None:
            output_path = scene_path / 'output' / 'semantic'
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: 实现模型加载和推理
        # 这需要完整的 NeRF 模型支持
        return ExportResult(
            success=False,
            output_path=str(output_path),
            error_message="Model-based export not yet implemented for cross-platform"
        )
    
    def export_from_annotations(self, scene_path: str,
                                output_path: str = None,
                                resize: Tuple[int, int] = None) -> ExportResult:
        """从手动标注导出语义分割图
        
        Args:
            scene_path: 场景目录
            output_path: 输出目录 (默认: <scene>/output/semantic/)
            resize: 调整大小 (width, height)
            
        Returns:
            导出结果
        """
        scene_path = Path(scene_path)
        semantic_dir = scene_path / 'semantic'
        
        if not semantic_dir.exists():
            return ExportResult(
                success=False,
                output_path="",
                error_message=f"Semantic directory not found: {semantic_dir}"
            )
        
        # 确定输出目录
        if output_path is None:
            output_path = scene_path / 'output' / 'semantic'
        else:
            output_path = Path(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理所有标注
        num_frames = 0
        for semantic_file in sorted(semantic_dir.glob('*.png')):
            # 读取标注
            if PIL_AVAILABLE:
                img = np.array(Image.open(semantic_file))
            elif CV2_AVAILABLE:
                img = cv2.imread(str(semantic_file), cv2.IMREAD_GRAYSCALE)
            else:
                continue
            
            # 后处理
            if self.config.objects_per_class is not None:
                img = self.post_process(img)
            
            # 调整大小
            if resize is not None:
                if PIL_AVAILABLE:
                    img = np.array(Image.fromarray(img).resize(resize, Image.NEAREST))
                elif CV2_AVAILABLE:
                    img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
            
            # 保存
            out_file = output_path / semantic_file.name
            if PIL_AVAILABLE:
                Image.fromarray(img).save(out_file)
            elif CV2_AVAILABLE:
                cv2.imwrite(str(out_file), img)
            
            num_frames += 1
        
        return ExportResult(
            success=True,
            output_path=str(output_path),
            num_frames=num_frames
        )
    
    def colorize_semantic(self, semantic_map: np.ndarray) -> np.ndarray:
        """将语义图转换为彩色图像
        
        Args:
            semantic_map: 语义分割图 [H, W]
            
        Returns:
            彩色图像 [H, W, 3]
        """
        return self.config.colors[semantic_map % len(self.config.colors)]


class VideoRenderer:
    """视频渲染器
    
    渲染 NeRF 模型的视频输出。
    """
    
    def __init__(self, fps: int = 5, stride: int = 1):
        self.fps = fps
        self.stride = stride
        self.device = _get_device()
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return SKVIDEO_AVAILABLE and TORCH_AVAILABLE
    
    def render_video(self, scene_path: str,
                     model_dir: str,
                     output_path: str,
                     max_depth: float = 7.5) -> ExportResult:
        """渲染视频
        
        Args:
            scene_path: 场景目录
            model_dir: 模型目录
            output_path: 输出视频路径
            max_depth: 深度可视化最大值
            
        Returns:
            导出结果
        """
        if not self.is_available:
            return ExportResult(
                success=False,
                output_path="",
                error_message="Video rendering dependencies not available"
            )
        
        # TODO: 实现视频渲染
        return ExportResult(
            success=False,
            output_path=output_path,
            error_message="Video rendering not yet implemented for cross-platform"
        )


class FormatConverter:
    """格式转换器
    
    支持转换为:
    - Open3D 格式
    - instant-ngp 格式
    """
    
    def __init__(self):
        pass
    
    def to_instant_ngp(self, scene_path: str, output_path: str) -> ExportResult:
        """转换为 instant-ngp 格式
        
        Args:
            scene_path: 场景目录
            output_path: 输出目录
            
        Returns:
            导出结果
        """
        scene_path = Path(scene_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取相机参数
        intrinsics_file = scene_path / 'intrinsics.txt'
        if not intrinsics_file.exists():
            return ExportResult(
                success=False,
                output_path="",
                error_message="intrinsics.txt not found"
            )
        
        intrinsics = np.loadtxt(intrinsics_file)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # 读取图像尺寸
        rgb_dir = scene_path / 'rgb'
        if not rgb_dir.exists():
            return ExportResult(
                success=False,
                output_path="",
                error_message="rgb directory not found"
            )
        
        sample_image = next(rgb_dir.glob('*.png'), None) or next(rgb_dir.glob('*.jpg'), None)
        if sample_image is None:
            return ExportResult(
                success=False,
                output_path="",
                error_message="No RGB images found"
            )
        
        if PIL_AVAILABLE:
            img = Image.open(sample_image)
            width, height = img.size
        elif CV2_AVAILABLE:
            img = cv2.imread(str(sample_image))
            height, width = img.shape[:2]
        else:
            return ExportResult(
                success=False,
                output_path="",
                error_message="PIL or OpenCV required"
            )
        
        # 读取位姿
        poses_file = scene_path / 'odometry.csv'
        if not poses_file.exists():
            return ExportResult(
                success=False,
                output_path="",
                error_message="odometry.csv not found"
            )
        
        poses_data = np.loadtxt(poses_file, delimiter=',', skiprows=1)
        
        # 构建 transforms.json
        transforms = {
            "camera_angle_x": 2 * np.arctan(width / (2 * fx)),
            "camera_angle_y": 2 * np.arctan(height / (2 * fy)),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w": width,
            "h": height,
            "frames": []
        }
        
        # 复制图像并添加帧信息
        rgb_files = sorted(rgb_dir.glob('*.png')) + sorted(rgb_dir.glob('*.jpg'))
        
        for i, rgb_file in enumerate(rgb_files):
            if i >= len(poses_data):
                break
            
            # 复制图像
            import shutil
            dest_file = output_path / 'images' / rgb_file.name
            dest_file.parent.mkdir(exist_ok=True)
            shutil.copy(rgb_file, dest_file)
            
            # 构建变换矩阵
            pose = poses_data[i]
            # 假设 pose 格式: timestamp, tx, ty, tz, qx, qy, qz, qw
            if len(pose) >= 8:
                tx, ty, tz = pose[1:4]
                qx, qy, qz, qw = pose[4:8]
                
                # 四元数转旋转矩阵
                R = self._quat_to_rotation_matrix(qx, qy, qz, qw)
                
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = R
                transform_matrix[:3, 3] = [tx, ty, tz]
                
                transforms["frames"].append({
                    "file_path": f"images/{rgb_file.name}",
                    "transform_matrix": transform_matrix.tolist()
                })
        
        # 保存 transforms.json
        with open(output_path / 'transforms.json', 'w') as f:
            json.dump(transforms, f, indent=2)
        
        return ExportResult(
            success=True,
            output_path=str(output_path),
            num_frames=len(transforms["frames"])
        )
    
    def _quat_to_rotation_matrix(self, qx, qy, qz, qw) -> np.ndarray:
        """四元数转旋转矩阵"""
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
