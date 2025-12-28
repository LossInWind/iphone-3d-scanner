"""ARKitScenes 数据导入器

复用自: autolabel/scripts/convert_arkitscenes.py

支持从 ARKitScenes 数据集转换为统一内部格式。
"""

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .base import BaseImporter, ImportResult


class ARKitScenesImporter(BaseImporter):
    """ARKitScenes 数据导入器
    
    将 ARKitScenes 数据集转换为统一的内部格式。
    
    输入格式:
        arkit_scenes_dir/
        └── <scene_id>/
            ├── lowres_wide/           # RGB 图像
            ├── lowres_depth/          # 深度图像
            ├── confidence/            # 置信度图像
            ├── lowres_wide.traj       # 轨迹文件
            └── lowres_wide_intrinsics/ # 相机内参
    
    输出格式:
        output_dir/
        └── <scene_id>/
            ├── rgb/           # RGB 帧 (PNG)
            ├── depth/         # 深度帧 (PNG, 16-bit)
            ├── pose/          # 相机位姿 (TXT)
            └── intrinsics.txt # 相机内参
    """
    
    def validate_input(self, input_path: str) -> bool:
        """验证 ARKitScenes 数据集是否有效"""
        required_items = [
            'lowres_wide.traj',
            'confidence',
            'lowres_depth',
            'lowres_wide',
            'lowres_wide_intrinsics'
        ]
        
        for item in required_items:
            path = os.path.join(input_path, item)
            if not os.path.exists(path):
                return False
        
        return True
    
    def import_data(self, input_path: str, output_path: str,
                   confidence_threshold: int = 2,
                   time_tolerance: float = 1.0/90.0) -> ImportResult:
        """导入 ARKitScenes 数据
        
        Args:
            input_path: ARKitScenes 场景目录路径
            output_path: 输出目录路径
            confidence_threshold: 深度置信度阈值
            time_tolerance: 时间戳匹配容差 (秒)
            
        Returns:
            ImportResult 对象
        """
        if not self.validate_input(input_path):
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=f"Invalid ARKitScenes dataset: {input_path}"
            )
        
        try:
            # 读取轨迹
            traj_file = os.path.join(input_path, 'lowres_wide.traj')
            trajectory = np.loadtxt(traj_file)
            
            # 收集图像路径
            rgb_dir = os.path.join(input_path, 'lowres_wide')
            depth_dir = os.path.join(input_path, 'lowres_depth')
            confidence_dir = os.path.join(input_path, 'confidence')
            intrinsics_dir = os.path.join(input_path, 'lowres_wide_intrinsics')
            
            rgb_images = self._collect_images(rgb_dir)
            depth_images = self._collect_images(depth_dir)
            confidence_images = self._collect_images(confidence_dir)
            intrinsics = self._read_intrinsics(intrinsics_dir)
            
            # 创建输出目录
            rgb_out = os.path.join(output_path, 'rgb')
            depth_out = os.path.join(output_path, 'depth')
            pose_out = os.path.join(output_path, 'pose')
            self._ensure_output_dirs(output_path, ['rgb', 'depth', 'pose'])
            
            # 写入场景
            rgb_count, depth_count, pose_count = self._write_scene(
                trajectory, rgb_images, depth_images, confidence_images,
                rgb_out, depth_out, pose_out,
                confidence_threshold, time_tolerance
            )
            
            # 写入内参
            np.savetxt(os.path.join(output_path, 'intrinsics.txt'), intrinsics)
            
            return ImportResult(
                success=True,
                output_path=output_path,
                rgb_count=rgb_count,
                depth_count=depth_count,
                pose_count=pose_count
            )
            
        except Exception as e:
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=str(e)
            )
    
    def _collect_images(self, dir_path: str) -> dict:
        """收集目录中的图像"""
        filenames = os.listdir(dir_path)
        out = {}
        for filename in filenames:
            name = filename.replace('.png', '')
            out[name] = os.path.join(dir_path, filename)
        return out
    
    def _read_intrinsics(self, dir_path: str) -> np.ndarray:
        """读取相机内参"""
        intrinsic_files = os.listdir(dir_path)
        intrinsic_path = os.path.join(dir_path, intrinsic_files[0])
        _, _, fx, fy, cx, cy = np.loadtxt(intrinsic_path)
        C = np.eye(3)
        C[0, 0] = fx
        C[1, 1] = fy
        C[0, 2] = cx
        C[1, 2] = cy
        return C
    
    def _to_timestamp(self, filename: str) -> float:
        """从文件名提取时间戳"""
        _, ts = filename.split('_')
        seconds, ms = [int(v) for v in ts.split('.')]
        return seconds + ms * 1e-3
    
    def _find_pose(self, trajectory: np.ndarray, rgb_name: str):
        """查找最近的位姿"""
        timestamp = self._to_timestamp(rgb_name)
        errors = np.abs(trajectory[:, 0] - timestamp)
        closest = errors.argmin()
        return trajectory[closest], errors[closest]
    
    def _to_transform(self, pose: np.ndarray) -> np.ndarray:
        """将位姿转换为变换矩阵"""
        rotvec = pose[1:4]
        translation = pose[4:]
        T_CW = np.eye(4)
        R_CW = Rotation.from_rotvec(rotvec)
        T_CW[:3, :3] = R_CW.as_matrix()
        T_CW[:3, 3] = translation
        return T_CW
    
    def _write_scene(self, trajectory, rgb_images, depth_images, 
                    confidence_images, rgb_out, depth_out, pose_out,
                    confidence_threshold, time_tolerance):
        """写入场景数据"""
        images = [(n, p) for n, p in rgb_images.items()]
        images.sort(key=lambda x: self._to_timestamp(x[0]))
        
        rgb_count = 0
        depth_count = 0
        pose_count = 0
        
        for i, (rgb_name, rgb_path_in) in enumerate(tqdm(images, desc="Writing frames")):
            if rgb_name not in depth_images or rgb_name not in confidence_images:
                continue
            
            pose, time_diff = self._find_pose(trajectory, rgb_name)
            if time_diff > time_tolerance:
                continue
            
            T_CW = self._to_transform(pose)
            
            image_name = f"{i:06}"
            pose_path = os.path.join(pose_out, image_name + '.txt')
            rgb_path = os.path.join(rgb_out, image_name + '.png')
            depth_path = os.path.join(depth_out, image_name + '.png')
            
            rgb = cv2.imread(rgb_path_in, -1)
            depth = cv2.imread(depth_images[rgb_name], -1)
            confidence = cv2.imread(confidence_images[rgb_name], -1)
            depth[confidence < confidence_threshold] = 0
            
            cv2.imwrite(depth_path, depth)
            cv2.imwrite(rgb_path, rgb)
            np.savetxt(pose_path, T_CW)
            
            rgb_count += 1
            depth_count += 1
            pose_count += 1
        
        return rgb_count, depth_count, pose_count


# 便捷函数
def import_arkitscenes(input_path: str, output_path: str,
                      confidence_threshold: int = 2) -> ImportResult:
    """导入 ARKitScenes 数据的便捷函数"""
    importer = ARKitScenesImporter()
    return importer.import_data(
        input_path, output_path,
        confidence_threshold=confidence_threshold
    )
