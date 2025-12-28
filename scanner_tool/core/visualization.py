"""
可视化模块 - 封装 StrayVisualizer 功能

提供点云可视化、轨迹显示、RGB-D 积分重建等功能。

复用自: StrayVisualizer/stray_visualize.py

Requirements:
- 3.1: 从 RGB-D 帧生成彩色点云
- 3.2: 在交互式 3D 查看器中显示点云
- 3.3: 支持按深度置信度过滤点
- 3.4: 支持每隔 N 帧显示以减少内存
- 3.5: 允许旋转、平移、缩放视图
- 4.1: 绘制连接相机位置的轨迹线
- 4.2: 可选显示每个位姿的相机坐标系
- 4.3: 允许配置显示频率
- 5.1: 使用 Open3D 进行 TSDF 体积积分
- 5.2: 允许配置体素大小
- 5.3: 提取并显示三角网格
- 5.4: 支持导出网格为 PLY 或 OBJ 格式
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
from typing import List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# 尝试导入 open3d，如果不可用则设置标志
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

# 视频读取后端检测：优先 OpenCV，回退 skvideo
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import skvideo.io
    SKVIDEO_AVAILABLE = True
except ImportError:
    SKVIDEO_AVAILABLE = False
    skvideo = None


def _check_open3d():
    """检查 open3d 是否可用"""
    if not OPEN3D_AVAILABLE:
        raise ImportError(
            "open3d is required for visualization features. "
            "Please install it with: pip install open3d"
        )

# 默认深度图尺寸 (Scanner App 输出)
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0


@dataclass
class SceneData:
    """场景数据结构"""
    poses: List[np.ndarray]  # 相机位姿列表 (4x4 矩阵)
    intrinsics: np.ndarray   # 相机内参 (3x3 矩阵)
    depth_frames: List[str]  # 深度图路径列表
    rgb_path: Optional[str] = None  # RGB 视频路径
    rgb_frames: Optional[List[str]] = None  # RGB 帧路径列表 (替代视频)
    confidence_dir: Optional[str] = None  # 置信度目录


@dataclass
class VisualizationConfig:
    """可视化配置"""
    every: int = 60  # 每隔 N 帧显示
    confidence_level: int = 1  # 置信度过滤阈值 (0, 1, 2)
    voxel_size: float = 0.015  # 体素大小 (米)
    sdf_trunc: float = 0.05  # TSDF 截断距离
    max_depth: float = MAX_DEPTH  # 最大深度
    frame_scale: float = 0.1  # 坐标系显示大小
    origin_scale: float = 0.25  # 原点坐标系大小


def _resize_camera_matrix(camera_matrix: np.ndarray, 
                          scale_x: float, scale_y: float) -> np.ndarray:
    """缩放相机内参矩阵"""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    return np.array([
        [fx * scale_x, 0.0, cx * scale_x],
        [0., fy * scale_y, cy * scale_y],
        [0., 0., 1.0]
    ])


def load_scene_data(scene_path: str) -> SceneData:
    """
    加载 Scanner App 数据集
    
    Args:
        scene_path: 数据集目录路径
        
    Returns:
        SceneData 对象
        
    Raises:
        FileNotFoundError: 如果必需文件不存在
    """
    scene_path = Path(scene_path)
    
    # 读取相机内参
    intrinsics_path = scene_path / 'camera_matrix.csv'
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Camera matrix not found: {intrinsics_path}")
    
    intrinsics = np.loadtxt(str(intrinsics_path), delimiter=',')
    
    # 读取位姿
    odometry_path = scene_path / 'odometry.csv'
    if not odometry_path.exists():
        raise FileNotFoundError(f"Odometry file not found: {odometry_path}")
    
    odometry = np.loadtxt(str(odometry_path), delimiter=',', skiprows=1)
    
    poses = []
    for line in odometry:
        # timestamp, frame, x, y, z, qx, qy, qz, qw
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        poses.append(T_WC)
    
    # 读取深度图列表
    depth_dir = scene_path / 'depth'
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    
    depth_frames = sorted([
        str(depth_dir / p) 
        for p in os.listdir(depth_dir)
        if p.endswith('.npy') or p.endswith('.png')
    ])
    
    # RGB 视频路径
    rgb_path = scene_path / 'rgb.mp4'
    rgb_path_str = str(rgb_path) if rgb_path.exists() else None
    
    # 置信度目录
    confidence_dir = scene_path / 'confidence'
    confidence_dir_str = str(confidence_dir) if confidence_dir.exists() else None
    
    return SceneData(
        poses=poses,
        intrinsics=intrinsics,
        depth_frames=depth_frames,
        rgb_path=rgb_path_str,
        confidence_dir=confidence_dir_str
    )


def load_depth(path: str, confidence: Optional[np.ndarray] = None, 
               filter_level: int = 0):
    """
    加载深度图
    
    Args:
        path: 深度图路径 (.npy 或 .png)
        confidence: 置信度图 (可选)
        filter_level: 置信度过滤阈值 (0, 1, 2)
        
    Returns:
        Open3D Image 对象
        
    Raises:
        ValueError: 如果深度格式不支持
    """
    _check_open3d()
    if path.endswith('.npy'):
        depth_mm = np.load(path)
    elif path.endswith('.png'):
        depth_mm = np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported depth format: {path}")
    
    depth_m = depth_mm.astype(np.float32) / 1000.0
    
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    
    return o3d.geometry.Image(depth_m)


def load_confidence(path: str) -> np.ndarray:
    """加载置信度图"""
    return np.array(Image.open(path))


def get_intrinsics(intrinsics: np.ndarray, 
                   width: int = DEPTH_WIDTH, 
                   height: int = DEPTH_HEIGHT,
                   original_width: int = 1920,
                   original_height: int = 1440):
    """
    获取缩放后的相机内参
    
    Args:
        intrinsics: 原始内参矩阵 (3x3)
        width: 目标宽度
        height: 目标高度
        original_width: 原始图像宽度
        original_height: 原始图像高度
        
    Returns:
        Open3D PinholeCameraIntrinsic 对象
    """
    _check_open3d()
    intrinsics_scaled = _resize_camera_matrix(
        intrinsics, 
        width / original_width, 
        height / original_height
    )
    return o3d.camera.PinholeCameraIntrinsic(
        width=width, 
        height=height, 
        fx=intrinsics_scaled[0, 0],
        fy=intrinsics_scaled[1, 1], 
        cx=intrinsics_scaled[0, 2], 
        cy=intrinsics_scaled[1, 2]
    )


class Visualizer:
    """
    可视化器类 - 提供统一的可视化接口
    
    支持:
    - 点云可视化 (Requirements 3.1-3.5)
    - 相机轨迹显示 (Requirements 4.1-4.3)
    - RGB-D 积分重建 (Requirements 5.1-5.4)
    """
    
    def __init__(self, scene_path: str, config: Optional[VisualizationConfig] = None):
        """
        初始化可视化器
        
        Args:
            scene_path: 数据集路径
            config: 可视化配置
        """
        _check_open3d()
        self.scene_path = Path(scene_path)
        self.config = config or VisualizationConfig()
        self._data: Optional[SceneData] = None
        self._geometries: List[Any] = []
    
    @property
    def data(self) -> SceneData:
        """懒加载场景数据"""
        if self._data is None:
            self._data = load_scene_data(str(self.scene_path))
        return self._data
    
    def show_point_clouds(self, every: Optional[int] = None,
                          confidence: Optional[int] = None) -> 'Visualizer':
        """
        显示点云
        
        Args:
            every: 每隔 N 帧采样 (覆盖配置)
            confidence: 置信度过滤阈值 (覆盖配置)
            
        Returns:
            self (支持链式调用)
        """
        every = every or self.config.every
        confidence_level = confidence if confidence is not None else self.config.confidence_level
        
        pc = create_point_cloud(
            str(self.scene_path),
            self.data,
            every=every,
            confidence_level=confidence_level
        )
        self._geometries.append(pc)
        return self
    
    def show_trajectory(self) -> 'Visualizer':
        """
        显示相机轨迹
        
        Returns:
            self (支持链式调用)
        """
        lines = create_trajectory(self.data)
        self._geometries.extend(lines)
        return self
    
    def show_frames(self, every: Optional[int] = None,
                    scale: Optional[float] = None) -> 'Visualizer':
        """
        显示相机坐标系
        
        Args:
            every: 每隔 N 帧显示 (覆盖配置)
            scale: 坐标系大小 (覆盖配置)
            
        Returns:
            self (支持链式调用)
        """
        every = every or self.config.every
        scale = scale or self.config.frame_scale
        
        frames = create_camera_frames(self.data, every=every, scale=scale)
        self._geometries.extend(frames)
        return self
    
    def integrate_mesh(self, voxel_size: Optional[float] = None) -> Any:
        """
        RGB-D 积分生成网格
        
        Args:
            voxel_size: 体素大小 (覆盖配置)
            
        Returns:
            重建的三角网格
        """
        voxel_size = voxel_size or self.config.voxel_size
        
        mesh = integrate_rgbd(
            str(self.scene_path),
            self.data,
            voxel_size=voxel_size,
            sdf_trunc=self.config.sdf_trunc,
            max_depth=self.config.max_depth
        )
        self._geometries.append(mesh)
        return mesh
    
    def add_geometry(self, geometry: Any) -> 'Visualizer':
        """
        添加自定义几何体
        
        Args:
            geometry: Open3D 几何体
            
        Returns:
            self (支持链式调用)
        """
        self._geometries.append(geometry)
        return self
    
    def clear(self) -> 'Visualizer':
        """
        清除所有几何体
        
        Returns:
            self (支持链式调用)
        """
        self._geometries.clear()
        return self
    
    def visualize(self, window_name: str = "Scanner Tool") -> None:
        """
        显示所有几何体
        
        Args:
            window_name: 窗口标题
        """
        if not self._geometries:
            print("No geometries to visualize. Add some first.")
            return
        
        o3d.visualization.draw_geometries(
            self._geometries, 
            window_name=window_name
        )


def create_trajectory(data: SceneData) -> List[Any]:
    """
    创建相机轨迹线
    
    Args:
        data: 场景数据
        
    Returns:
        LineSet 列表
    """
    _check_open3d()
    line_sets = []
    previous_pose = None
    
    for T_WC in data.poses:
        if previous_pose is not None:
            points = o3d.utility.Vector3dVector([
                previous_pose[:3, 3], 
                T_WC[:3, 3]
            ])
            lines = o3d.utility.Vector2iVector([[0, 1]])
            line = o3d.geometry.LineSet(points=points, lines=lines)
            line_sets.append(line)
        previous_pose = T_WC
    
    return line_sets


def create_camera_frames(data: SceneData, 
                         every: int = 60, 
                         scale: float = 0.1) -> List[Any]:
    """
    创建相机坐标系可视化
    
    Args:
        data: 场景数据
        every: 每隔 N 帧显示一个
        scale: 坐标系大小
        
    Returns:
        TriangleMesh 列表
    """
    _check_open3d()
    frames = [
        o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.25, np.zeros(3))
    ]
    
    for i, T_WC in enumerate(data.poses):
        if i % every != 0:
            continue
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(scale, np.zeros(3))
        frames.append(mesh.transform(T_WC))
    
    return frames


def _read_video_frames(video_path: str):
    """
    读取视频帧的生成器
    
    优先使用 OpenCV (cv2)，如果不可用则回退到 skvideo。
    OpenCV 更稳定且与新版 NumPy 兼容。
    
    Args:
        video_path: 视频文件路径
        
    Yields:
        RGB 格式的帧 (numpy array)
        
    Raises:
        ImportError: 如果 cv2 和 skvideo 都不可用
    """
    if CV2_AVAILABLE:
        # 优先使用 OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV 读取的是 BGR，转换为 RGB
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
    elif SKVIDEO_AVAILABLE:
        # 回退到 skvideo
        import skvideo.io
        for frame in skvideo.io.vreader(video_path):
            yield frame
    else:
        raise ImportError(
            "Video reading requires either opencv-python or scikit-video. "
            "Install with: pip install opencv-python  OR  pip install scikit-video"
        )


def create_point_cloud(scene_path: str, 
                       data: SceneData, 
                       every: int = 60,
                       confidence_level: int = 1) -> Any:
    """
    从深度图创建合并的点云
    
    Args:
        scene_path: 数据集路径
        data: 场景数据
        every: 每隔 N 帧采样
        confidence_level: 置信度过滤阈值
        
    Returns:
        合并的点云
    """
    _check_open3d()
    
    intrinsics = get_intrinsics(data.intrinsics)
    pc = o3d.geometry.PointCloud()
    
    if data.rgb_path is None:
        raise FileNotFoundError(f"RGB video not found in {scene_path}")
    
    video = _read_video_frames(data.rgb_path)
    
    for i, (T_WC, rgb) in enumerate(zip(data.poses, video)):
        if i % every != 0:
            continue
        
        print(f"Processing point cloud {i}", end="\r")
        
        T_CW = np.linalg.inv(T_WC)
        
        # 加载置信度 (如果可用)
        confidence = None
        if data.confidence_dir:
            confidence_path = os.path.join(data.confidence_dir, f'{i:06}.png')
            if os.path.exists(confidence_path):
                confidence = load_confidence(confidence_path)
        
        if i < len(data.depth_frames):
            depth_path = data.depth_frames[i]
            depth = load_depth(depth_path, confidence, filter_level=confidence_level)
        else:
            continue
        
        rgb_img = Image.fromarray(rgb)
        rgb_img = rgb_img.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb_arr = np.array(rgb_img)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_arr), 
            depth,
            depth_scale=1.0, 
            depth_trunc=MAX_DEPTH, 
            convert_rgb_to_intensity=False
        )
        pc += o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsics, extrinsic=T_CW
        )
    
    print()  # 换行
    return pc


def integrate_rgbd(scene_path: str, 
                   data: SceneData,
                   voxel_size: float = 0.015,
                   sdf_trunc: float = 0.05,
                   max_depth: float = MAX_DEPTH) -> Any:
    """
    RGB-D 积分重建
    
    Args:
        scene_path: 数据集路径
        data: 场景数据
        voxel_size: 体素大小 (米)
        sdf_trunc: TSDF 截断距离
        max_depth: 最大深度
        
    Returns:
        重建的三角网格
    """
    _check_open3d()
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    intrinsics = get_intrinsics(data.intrinsics)
    
    if data.rgb_path is None:
        raise FileNotFoundError(f"RGB video not found in {scene_path}")
    
    video = _read_video_frames(data.rgb_path)
    
    for i, (T_WC, rgb) in enumerate(zip(data.poses, video)):
        print(f"Integrating frame {i:06}", end='\r')
        
        if i < len(data.depth_frames):
            depth_path = data.depth_frames[i]
            depth = load_depth(depth_path)
        else:
            continue
        
        rgb_img = Image.fromarray(rgb)
        rgb_img = rgb_img.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb_arr = np.array(rgb_img)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_arr), 
            depth,
            depth_scale=1.0, 
            depth_trunc=max_depth, 
            convert_rgb_to_intensity=False
        )
        
        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_WC))
    
    print()  # 换行
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def export_mesh(mesh: Any, 
                output_path: str,
                write_vertex_normals: bool = True,
                write_vertex_colors: bool = True) -> bool:
    """
    导出网格到文件
    
    支持格式: PLY, OBJ, STL, OFF, GLTF
    
    Args:
        mesh: 三角网格
        output_path: 输出文件路径 (扩展名决定格式)
        write_vertex_normals: 是否写入顶点法线
        write_vertex_colors: 是否写入顶点颜色
        
    Returns:
        是否成功
    """
    _check_open3d()
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return o3d.io.write_triangle_mesh(
        output_path, 
        mesh,
        write_vertex_normals=write_vertex_normals,
        write_vertex_colors=write_vertex_colors
    )


def export_point_cloud(point_cloud: Any,
                       output_path: str,
                       write_ascii: bool = False) -> bool:
    """
    导出点云到文件
    
    支持格式: PLY, PCD, XYZ, XYZN, XYZRGB, PTS
    
    Args:
        point_cloud: 点云
        output_path: 输出文件路径
        write_ascii: 是否使用 ASCII 格式
        
    Returns:
        是否成功
    """
    _check_open3d()
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    return o3d.io.write_point_cloud(
        output_path, 
        point_cloud,
        write_ascii=write_ascii
    )


def visualize(geometries: List[Any], window_name: str = "Scanner Tool") -> None:
    """
    显示 3D 几何体
    
    Args:
        geometries: Open3D 几何体列表
        window_name: 窗口标题
    """
    _check_open3d()
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


def validate_scene(scene_path: str) -> Tuple[bool, str]:
    """
    验证数据集是否有效
    
    Args:
        scene_path: 数据集路径
        
    Returns:
        (是否有效, 错误信息)
    """
    required_files = ['rgb.mp4', 'camera_matrix.csv', 'odometry.csv']
    required_dirs = ['depth']
    
    for f in required_files:
        if not os.path.exists(os.path.join(scene_path, f)):
            return False, f"Missing required file: {f}"
    
    for d in required_dirs:
        if not os.path.isdir(os.path.join(scene_path, d)):
            return False, f"Missing required directory: {d}"
    
    return True, ""
