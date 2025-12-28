"""
位姿估计模块 - 封装 SfM 管线和场景边界计算

复用自: autolabel/scripts/mapping.py
Requirements: 6.1-6.7, 7.1-7.3
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np

# 添加 autolabel 到路径
AUTOLABEL_PATH = Path(__file__).parent.parent.parent / 'autolabel'
if str(AUTOLABEL_PATH) not in sys.path:
    sys.path.insert(0, str(AUTOLABEL_PATH))


@dataclass
class BoundingBox:
    """场景边界框"""
    min_bounds: np.ndarray  # [min_x, min_y, min_z]
    max_bounds: np.ndarray  # [max_x, max_y, max_z]
    transform: np.ndarray   # 4x4 变换矩阵
    
    def to_file(self, path: str) -> None:
        """保存边界到文件"""
        with open(path, 'wt') as f:
            min_str = " ".join([str(x) for x in self.min_bounds])
            max_str = " ".join([str(x) for x in self.max_bounds])
            f.write(f"{min_str} {max_str} 0.01")
    
    @classmethod
    def from_file(cls, path: str) -> 'BoundingBox':
        """从文件加载边界"""
        data = np.loadtxt(path)
        bounds = data[:6].reshape(2, 3)
        return cls(
            min_bounds=bounds[0],
            max_bounds=bounds[1],
            transform=np.eye(4)
        )


@dataclass
class PoseEstimationResult:
    """位姿估计结果"""
    poses: Dict[str, np.ndarray]  # frame_name -> 4x4 pose matrix
    intrinsics: np.ndarray        # 3x3 相机内参
    distortion: np.ndarray        # 畸变参数
    bounds: Optional[BoundingBox] = None
    scale: float = 1.0


@dataclass
class PoseEstimationConfig:
    """位姿估计配置"""
    debug: bool = False           # 是否保存调试信息
    visualize: bool = False       # 是否可视化结果
    exhaustive_threshold: int = 250  # 超过此帧数使用 NetVLAD 加速
    ransac_iterations: int = 10000   # RANSAC 迭代次数


class PoseEstimator:
    """
    位姿估计器 - 封装 SfM 管线
    
    复用 autolabel/scripts/mapping.py 的功能:
    - SuperPoint 特征提取 (Req 6.1)
    - SuperGlue 特征匹配 (Req 6.2)
    - NetVLAD 图像检索 (Req 6.3)
    - COLMAP 重建 (Req 6.4)
    - 畸变校正 (Req 6.5)
    - 图像去畸变 (Req 6.6)
    - 深度尺度对齐 (Req 6.7)
    """
    
    def __init__(self, config: Optional[PoseEstimationConfig] = None):
        self.config = config or PoseEstimationConfig()
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """检查必要依赖是否可用"""
        self._hloc_available = False
        self._colmap_available = False
        
        try:
            import pycolmap
            self._colmap_available = True
        except ImportError:
            pass
        
        try:
            from hloc import extract_features
            self._hloc_available = True
        except ImportError:
            pass
        
        return self._hloc_available and self._colmap_available
    
    @property
    def is_available(self) -> bool:
        """检查 SfM 功能是否可用"""
        return self._hloc_available and self._colmap_available

    
    def run_sfm(self, scene_path: str) -> PoseEstimationResult:
        """
        运行完整 SfM 管线
        
        Args:
            scene_path: 场景目录路径
            
        Returns:
            PoseEstimationResult: 包含位姿、内参、畸变参数和边界
            
        Raises:
            RuntimeError: 如果依赖不可用
            ValueError: 如果场景路径无效
        """
        if not self.is_available:
            raise RuntimeError(
                "SfM dependencies not available. "
                "Please install hloc and pycolmap."
            )
        
        if not os.path.exists(scene_path):
            raise ValueError(f"Scene path does not exist: {scene_path}")
        
        # 导入 autolabel 模块
        from autolabel.utils import Scene
        from autolabel.scripts.mapping import HLoc, ScaleEstimation, PoseSaver
        
        # 创建临时目录
        tmp_dir = tempfile.mkdtemp()
        
        try:
            # 加载场景
            scene = Scene(scene_path)
            
            # 创建 flags 对象
            class Flags:
                def __init__(self, debug, vis):
                    self.debug = debug
                    self.vis = vis
            
            flags = Flags(self.config.debug, self.config.visualize)
            
            # 运行 HLoc SfM
            hloc = HLoc(tmp_dir, scene, flags)
            hloc.run()
            
            # 重新加载场景（内参可能已更新）
            scene = Scene(scene_path)
            
            # 估计尺度
            scale_estimation = ScaleEstimation(scene, tmp_dir)
            scaled_poses = scale_estimation.run()
            
            # 保存位姿和计算边界
            pose_saver = PoseSaver(scene, scaled_poses)
            pose_saver.run()
            
            # 读取结果
            intrinsics = np.loadtxt(os.path.join(scene_path, 'intrinsics.txt'))
            distortion = np.loadtxt(
                os.path.join(scene_path, 'distortion_parameters.txt')
            )
            
            # 读取边界
            bbox_path = os.path.join(scene_path, 'bbox.txt')
            bounds = None
            if os.path.exists(bbox_path):
                bounds = BoundingBox.from_file(bbox_path)
            
            return PoseEstimationResult(
                poses=scaled_poses,
                intrinsics=intrinsics,
                distortion=distortion,
                bounds=bounds
            )
            
        finally:
            # 清理临时目录
            if self.config.debug:
                debug_dir = "/tmp/sfm_debug"
                if os.path.exists(debug_dir):
                    shutil.rmtree(debug_dir)
                shutil.move(tmp_dir, debug_dir)
            else:
                shutil.rmtree(tmp_dir)

    
    def undistort_images(
        self, 
        scene_path: str,
        intrinsics: np.ndarray,
        distortion: np.ndarray
    ) -> None:
        """
        去畸变图像
        
        Args:
            scene_path: 场景目录路径
            intrinsics: 3x3 相机内参矩阵
            distortion: 畸变参数 [k1, k2, p1, p2]
        """
        import cv2
        from autolabel.undistort import ImageUndistorter
        from autolabel.utils import Scene, Camera
        
        scene = Scene(scene_path)
        
        # 创建输出目录
        undistorted_rgb_dir = os.path.join(scene_path, "rgb")
        undistorted_depth_dir = os.path.join(scene_path, "depth")
        os.makedirs(undistorted_rgb_dir, exist_ok=True)
        os.makedirs(undistorted_depth_dir, exist_ok=True)
        
        # RGB 去畸变器
        color_undistorter = ImageUndistorter(
            K=intrinsics,
            D=distortion,
            H=scene.camera.size[1],
            W=scene.camera.size[0]
        )
        
        # 深度图去畸变器
        depth_camera = Camera(intrinsics, scene.camera.size).scale(
            scene.depth_size()
        )
        depth_undistorter = ImageUndistorter(
            K=depth_camera.camera_matrix,
            D=distortion,
            H=depth_camera.size[1],
            W=depth_camera.size[0]
        )
        
        # 去畸变 RGB 图像
        for image_path in scene.raw_rgb_paths():
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            undistorted = color_undistorter.undistort_image(image)
            output_path = os.path.join(
                undistorted_rgb_dir, 
                os.path.basename(image_path)
            )
            cv2.imwrite(output_path, undistorted)
        
        # 去畸变深度图
        for depth_path in scene.raw_depth_paths():
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            undistorted = depth_undistorter.undistort_image(depth)
            output_path = os.path.join(
                undistorted_depth_dir,
                os.path.basename(depth_path)
            )
            cv2.imwrite(output_path, undistorted)



class SceneBoundsCalculator:
    """
    场景边界计算器
    
    复用 autolabel/scripts/mapping.py::PoseSaver.compute_bbox()
    Requirements: 7.1, 7.2, 7.3
    """
    
    def __init__(self, scene_path: str):
        """
        Args:
            scene_path: 场景目录路径
        """
        self.scene_path = scene_path
        self._check_dependencies()
    
    def _check_dependencies(self) -> bool:
        """检查依赖"""
        try:
            import open3d as o3d
            self._o3d_available = True
        except ImportError:
            self._o3d_available = False
        return self._o3d_available
    
    @property
    def is_available(self) -> bool:
        return self._o3d_available
    
    def compute_bounds(
        self,
        poses: Optional[Dict[str, np.ndarray]] = None,
        stride: int = 1,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> BoundingBox:
        """
        计算场景边界
        
        Args:
            poses: 位姿字典 {frame_name: 4x4 T_WC}，如果为 None 则从文件加载
            stride: 采样步长，用于减少计算量
            nb_neighbors: 统计滤波邻居数
            std_ratio: 统计滤波标准差比率
            
        Returns:
            BoundingBox: 场景边界框
        """
        if not self.is_available:
            raise RuntimeError("Open3D not available")
        
        import open3d as o3d
        from autolabel.utils import Scene, transform_points
        
        scene = Scene(self.scene_path)
        
        # 如果没有提供位姿，从文件加载
        if poses is None:
            poses = self._load_poses_from_files(scene)
        
        # 获取深度图尺寸和内参
        depth_frame = o3d.io.read_image(scene.depth_paths()[0])
        depth_size = np.asarray(depth_frame).shape[::-1]
        K = scene.camera.scale(depth_size).camera_matrix
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            int(depth_size[0]), int(depth_size[1]),
            K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        
        # 构建深度帧字典
        depth_frames = {
            os.path.basename(p).split('.')[0]: p 
            for p in scene.depth_paths()
        }
        
        # 累积点云
        pc = o3d.geometry.PointCloud()
        min_bounds = np.zeros(3)
        max_bounds = np.zeros(3)
        
        items = list(poses.items())
        actual_stride = max(len(items) // 100, stride)
        
        for key, T_WC in items[::actual_stride]:
            if key not in depth_frames:
                print(f"WARNING: Can't find depth image {key}.png")
                continue
            
            depth = o3d.io.read_image(depth_frames[key])
            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics
            )
            pc_C = np.asarray(pc_C.points)
            pc_W = transform_points(T_WC, pc_C)
            
            min_bounds = np.minimum(min_bounds, pc_W.min(axis=0))
            max_bounds = np.maximum(max_bounds, pc_W.max(axis=0))
            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)
            ).uniform_down_sample(50)
        
        # 统计滤波去除离群点 (Req 7.2)
        filtered, _ = pc.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        # 计算定向边界框
        bbox = filtered.get_oriented_bounding_box(robust=True)
        T = np.eye(4)
        T[:3, :3] = bbox.R.T
        
        # 转换到轴对齐边界框
        o3d_aabb = o3d.geometry.PointCloud(filtered).transform(T)\
            .get_axis_aligned_bounding_box()
        center = o3d_aabb.get_center()
        T[:3, 3] = -center
        
        aabb_min = o3d_aabb.get_min_bound() - center
        aabb_max = o3d_aabb.get_max_bound() - center
        
        return BoundingBox(
            min_bounds=aabb_min,
            max_bounds=aabb_max,
            transform=T
        )

    
    def _load_poses_from_files(self, scene) -> Dict[str, np.ndarray]:
        """从文件加载位姿"""
        poses = {}
        pose_dir = os.path.join(self.scene_path, 'pose')
        if not os.path.exists(pose_dir):
            raise ValueError(f"Pose directory not found: {pose_dir}")
        
        for pose_file in os.listdir(pose_dir):
            if pose_file.startswith('.'):
                continue
            frame_name = pose_file.split('.')[0]
            T_CW = np.loadtxt(os.path.join(pose_dir, pose_file))
            # 转换为 T_WC
            T_WC = np.linalg.inv(T_CW)
            poses[frame_name] = T_WC
        
        return poses
    
    def save_bounds(self, bounds: BoundingBox, output_path: Optional[str] = None):
        """
        保存边界到文件
        
        Args:
            bounds: 边界框
            output_path: 输出路径，默认为 scene_path/bbox.txt
        """
        if output_path is None:
            output_path = os.path.join(self.scene_path, 'bbox.txt')
        bounds.to_file(output_path)


# 便捷函数
def run_sfm_pipeline(
    scene_path: str,
    debug: bool = False,
    visualize: bool = False
) -> PoseEstimationResult:
    """
    运行完整 SfM 管线的便捷函数
    
    Args:
        scene_path: 场景目录路径
        debug: 是否保存调试信息
        visualize: 是否可视化结果
        
    Returns:
        PoseEstimationResult
    """
    config = PoseEstimationConfig(debug=debug, visualize=visualize)
    estimator = PoseEstimator(config)
    return estimator.run_sfm(scene_path)


def compute_scene_bounds(
    scene_path: str,
    poses: Optional[Dict[str, np.ndarray]] = None,
    save: bool = True
) -> BoundingBox:
    """
    计算场景边界的便捷函数
    
    Args:
        scene_path: 场景目录路径
        poses: 位姿字典，如果为 None 则从文件加载
        save: 是否保存到 bbox.txt
        
    Returns:
        BoundingBox
    """
    calculator = SceneBoundsCalculator(scene_path)
    bounds = calculator.compute_bounds(poses)
    if save:
        calculator.save_bounds(bounds)
    return bounds
