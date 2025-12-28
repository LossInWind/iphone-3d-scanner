"""Replica 数据导入器

复用自: autolabel/scripts/convert_replica.py

支持从 SemanticNeRF Replica 渲染数据转换为统一内部格式。
"""

import os
import math
import json
import shutil
import numpy as np
import cv2
from tqdm import tqdm

from .base import BaseImporter, ImportResult


class ReplicaImporter(BaseImporter):
    """Replica 数据导入器
    
    将 SemanticNeRF Replica 渲染数据转换为统一的内部格式。
    
    输入格式:
        replica_scene/
        ├── rgb/              # RGB 图像
        ├── depth/            # 深度图像
        ├── semantic_class/   # 语义标签
        └── traj_w_c.txt      # 轨迹文件
    
    输出格式:
        output_dir/
        ├── rgb/           # RGB 帧 (PNG)
        ├── depth/         # 深度帧 (PNG, 16-bit)
        ├── pose/          # 相机位姿 (TXT)
        ├── semantic/      # 语义标签 (PNG)
        ├── intrinsics.txt # 相机内参
        ├── bbox.txt       # 场景边界
        └── metadata.json  # 元数据
    """
    
    # Replica 默认相机参数
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_HFOV = 90.0
    
    def validate_input(self, input_path: str) -> bool:
        """验证 Replica 数据集是否有效"""
        required_items = [
            'rgb',
            'depth',
            'traj_w_c.txt'
        ]
        
        for item in required_items:
            path = os.path.join(input_path, item)
            if not os.path.exists(path):
                return False
        
        return True
    
    def import_data(self, input_path: str, output_path: str,
                   compute_bounds: bool = True) -> ImportResult:
        """导入 Replica 数据
        
        Args:
            input_path: Replica 场景目录路径
            output_path: 输出目录路径
            compute_bounds: 是否计算场景边界
            
        Returns:
            ImportResult 对象
        """
        if not self.validate_input(input_path):
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=f"Invalid Replica dataset: {input_path}"
            )
        
        try:
            # 收集路径
            rgb_frames, depth_frames, semantic_frames = self._collect_paths(input_path)
            
            # 创建输出目录
            dirs = ['rgb', 'depth', 'pose']
            if semantic_frames:
                dirs.append('semantic')
            self._ensure_output_dirs(output_path, dirs)
            
            # 复制帧
            rgb_count, semantic_classes = self._copy_frames(
                rgb_frames, depth_frames, semantic_frames, output_path
            )
            
            # 复制轨迹
            pose_count = self._copy_trajectory(input_path, output_path)
            
            # 写入内参
            self._write_intrinsics(output_path)
            
            # 写入元数据
            if semantic_frames:
                self._write_metadata(output_path, len(semantic_classes))
            
            # 计算边界
            if compute_bounds:
                self._compute_bounds(output_path)
            
            return ImportResult(
                success=True,
                output_path=output_path,
                rgb_count=rgb_count,
                depth_count=rgb_count,
                pose_count=pose_count
            )
            
        except Exception as e:
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=str(e)
            )
    
    def _collect_paths(self, input_path: str):
        """收集输入文件路径"""
        rgb_path = os.path.join(input_path, 'rgb')
        depth_path = os.path.join(input_path, 'depth')
        semantic_path = os.path.join(input_path, 'semantic_class')
        
        rgb_frames = [f for f in os.listdir(rgb_path) if f[0] != '.']
        depth_frames = [f for f in os.listdir(depth_path) if f[0] != '.']
        
        # 语义标签可选
        semantic_frames = []
        if os.path.exists(semantic_path):
            semantic_frames = [
                f for f in os.listdir(semantic_path)
                if f[0] != '.' and 'semantic' in f
            ]
        
        # 排序
        def sort_key(x):
            return int(x.split('_')[-1].split('.')[0])
        
        rgb_frames = sorted(rgb_frames, key=sort_key)
        depth_frames = sorted(depth_frames, key=sort_key)
        if semantic_frames:
            semantic_frames = sorted(semantic_frames, key=sort_key)
        
        # 转换为完整路径
        rgb_frames = [os.path.join(rgb_path, f) for f in rgb_frames]
        depth_frames = [os.path.join(depth_path, f) for f in depth_frames]
        if semantic_frames:
            semantic_frames = [os.path.join(semantic_path, f) for f in semantic_frames]
        
        return rgb_frames, depth_frames, semantic_frames
    
    def _copy_frames(self, rgb_frames, depth_frames, semantic_frames, output_path):
        """复制帧数据"""
        rgb_out = os.path.join(output_path, 'rgb')
        depth_out = os.path.join(output_path, 'depth')
        semantic_out = os.path.join(output_path, 'semantic')
        
        semantic_classes = set()
        semantic_data = []
        
        # 复制 RGB 和深度
        for i, (rgb, depth) in enumerate(
            tqdm(zip(rgb_frames, depth_frames), desc="Copying frames", total=len(rgb_frames))
        ):
            rgb_out_path = os.path.join(rgb_out, f"{i:06}.png")
            depth_out_path = os.path.join(depth_out, f"{i:06}.png")
            shutil.copy(rgb, rgb_out_path)
            shutil.copy(depth, depth_out_path)
            
            # 处理语义标签
            if semantic_frames and i < len(semantic_frames):
                sem_frame = cv2.imread(semantic_frames[i], -1)
                semantic_data.append(sem_frame)
                classes = np.unique(sem_frame)
                semantic_classes = semantic_classes.union(classes)
        
        # 重映射语义类别
        if semantic_data:
            semantic_classes = sorted(semantic_classes)
            for i, frame in enumerate(
                tqdm(semantic_data, desc="Writing semantic")
            ):
                new_semantic = np.zeros_like(frame)
                for new_id, old_id in enumerate(semantic_classes):
                    new_semantic[frame == old_id] = new_id
                semantic_path = os.path.join(semantic_out, f"{i:06}.png")
                cv2.imwrite(semantic_path, new_semantic)
        
        return len(rgb_frames), semantic_classes
    
    def _copy_trajectory(self, input_path: str, output_path: str) -> int:
        """复制轨迹数据"""
        pose_dir = os.path.join(output_path, 'pose')
        trajectory = np.loadtxt(
            os.path.join(input_path, 'traj_w_c.txt'),
            delimiter=' '
        ).reshape(-1, 4, 4)
        
        for i, T_CW in enumerate(trajectory):
            pose_out = os.path.join(pose_dir, f"{i:06}.txt")
            np.savetxt(pose_out, np.linalg.inv(T_CW))
        
        return len(trajectory)
    
    def _write_intrinsics(self, output_path: str):
        """写入相机内参"""
        fx = self.DEFAULT_WIDTH / 2.0 / math.tan(math.radians(self.DEFAULT_HFOV / 2.0))
        cx = (self.DEFAULT_WIDTH - 1.0) / 2.0
        cy = (self.DEFAULT_HEIGHT - 1.0) / 2.0
        
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fx
        camera_matrix[0, 2] = cx
        camera_matrix[1, 2] = cy
        
        np.savetxt(os.path.join(output_path, 'intrinsics.txt'), camera_matrix)
    
    def _write_metadata(self, output_path: str, n_classes: int):
        """写入元数据"""
        metadata = {'n_classes': n_classes}
        metadata_path = os.path.join(output_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata, indent=2))
    
    def _compute_bounds(self, output_path: str):
        """计算场景边界"""
        try:
            import open3d as o3d
        except ImportError:
            # 如果没有 Open3D，跳过边界计算
            return
        
        # 读取深度和位姿
        depth_dir = os.path.join(output_path, 'depth')
        pose_dir = os.path.join(output_path, 'pose')
        intrinsics_path = os.path.join(output_path, 'intrinsics.txt')
        
        if not os.path.exists(pose_dir):
            return
        
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
        pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
        
        if not depth_files or not pose_files:
            return
        
        # 读取内参
        K = np.loadtxt(intrinsics_path)
        
        # 读取第一个深度图获取尺寸
        depth_sample = o3d.io.read_image(os.path.join(depth_dir, depth_files[0]))
        depth_size = np.asarray(depth_sample).shape[::-1]
        
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            int(depth_size[0]), int(depth_size[1]),
            K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        
        pc = o3d.geometry.PointCloud()
        
        # 每隔 10 帧采样
        for depth_file, pose_file in zip(depth_files[::10], pose_files[::10]):
            T_CW = np.loadtxt(os.path.join(pose_dir, pose_file))
            T_WC = np.linalg.inv(T_CW)
            depth = o3d.io.read_image(os.path.join(depth_dir, depth_file))
            
            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics
            )
            pc_C_points = np.asarray(pc_C.points)
            
            # 变换到世界坐标系
            pc_W = (T_WC[:3, :3] @ pc_C_points.T).T + T_WC[:3, 3]
            
            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)
            ).uniform_down_sample(50)
        
        # 移除离群点
        filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        aabb = filtered.get_axis_aligned_bounding_box()
        
        # 写入边界
        with open(os.path.join(output_path, 'bbox.txt'), 'wt') as f:
            min_str = " ".join([str(x) for x in aabb.get_min_bound()])
            max_str = " ".join([str(x) for x in aabb.get_max_bound()])
            f.write(f"{min_str} {max_str} 0.01")


# 便捷函数
def import_replica(input_path: str, output_path: str,
                  compute_bounds: bool = True) -> ImportResult:
    """导入 Replica 数据的便捷函数"""
    importer = ReplicaImporter()
    return importer.import_data(
        input_path, output_path,
        compute_bounds=compute_bounds
    )
