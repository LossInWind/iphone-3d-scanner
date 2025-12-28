"""Scanner App 数据导入器

复用自: autolabel/scripts/convert_scanner.py

支持从 iOS Scanner App 导出的数据集转换为统一内部格式。
"""

import os
import math
import numpy as np
import cv2
from tqdm import tqdm

from .base import BaseImporter, ImportResult


class ScannerImporter(BaseImporter):
    """Scanner App 数据导入器
    
    将 iOS Scanner App 导出的数据转换为统一的内部格式。
    
    输入格式:
        scan_dir/
        ├── rgb.mp4          # RGB 视频
        ├── depth/           # 深度帧 (PNG)
        ├── confidence/      # 置信度帧 (PNG)
        └── camera_matrix.csv # 相机内参
    
    输出格式:
        output_dir/
        ├── raw_rgb/         # RGB 帧 (JPEG)
        ├── raw_depth/       # 深度帧 (PNG, 16-bit)
        └── intrinsics.txt   # 相机内参
    """
    
    def validate_input(self, input_path: str) -> bool:
        """验证 Scanner App 数据集是否有效
        
        Args:
            input_path: Scanner App 数据集路径
            
        Returns:
            True 如果数据集包含必需文件
        """
        required_files = [
            'rgb.mp4',
            'camera_matrix.csv'
        ]
        required_dirs = [
            'depth',
            'confidence'
        ]
        
        for f in required_files:
            if not os.path.exists(os.path.join(input_path, f)):
                return False
        
        for d in required_dirs:
            dir_path = os.path.join(input_path, d)
            if not os.path.isdir(dir_path):
                return False
            # 检查目录非空
            if not os.listdir(dir_path):
                return False
        
        return True
    
    def import_data(self, input_path: str, output_path: str,
                   subsample: int = 1, rotate: bool = False,
                   confidence_threshold: int = 2) -> ImportResult:
        """导入 Scanner App 数据
        
        Args:
            input_path: Scanner App 数据集路径
            output_path: 输出目录路径
            subsample: 帧采样间隔 (每 N 帧取一帧)
            rotate: 是否旋转 90 度
            confidence_threshold: 深度置信度阈值 (0-2)
            
        Returns:
            ImportResult 对象
        """
        if not self.validate_input(input_path):
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=f"Invalid Scanner App dataset: {input_path}"
            )
        
        try:
            # 创建输出目录
            rgb_out = os.path.join(output_path, 'raw_rgb')
            depth_out = os.path.join(output_path, 'raw_depth')
            self._ensure_output_dirs(output_path, ['raw_rgb', 'raw_depth'])
            
            # 写入内参
            self._write_intrinsics(input_path, output_path, rotate)
            
            # 写入深度帧
            depth_count = self._write_depth(
                input_path, depth_out, 
                rotate=rotate, 
                subsample=subsample,
                confidence_threshold=confidence_threshold
            )
            
            # 写入 RGB 帧
            rgb_count = self._write_frames(
                input_path, rgb_out,
                rotate=rotate,
                subsample=subsample
            )
            
            return ImportResult(
                success=True,
                output_path=output_path,
                rgb_count=rgb_count,
                depth_count=depth_count,
                pose_count=0  # Scanner App 原始数据不包含位姿
            )
            
        except Exception as e:
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=str(e)
            )
    
    def _write_frames(self, scan_dir: str, rgb_out_dir: str,
                     rotate: bool = False, subsample: int = 1) -> int:
        """写入 RGB 帧
        
        Args:
            scan_dir: 输入目录
            rgb_out_dir: RGB 输出目录
            rotate: 是否旋转
            subsample: 采样间隔
            
        Returns:
            写入的帧数
        """
        try:
            from skvideo import io as skvideo_io
        except ImportError:
            raise ImportError(
                "skvideo is required for video processing. "
                "Install with: pip install scikit-video"
            )
        
        rgb_video = os.path.join(scan_dir, 'rgb.mp4')
        video = skvideo_io.vreader(rgb_video)
        img_idx = 0
        
        for i, frame in tqdm(enumerate(video), desc="Writing RGB"):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if i % subsample != 0:
                continue
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            frame_path = os.path.join(rgb_out_dir, f"{img_idx:05}.jpg")
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            cv2.imwrite(frame_path, frame, params)
            img_idx += 1
        
        return img_idx
    
    def _write_depth(self, scan_dir: str, depth_out_dir: str,
                    rotate: bool = False, subsample: int = 1,
                    confidence_threshold: int = 2) -> int:
        """写入深度帧
        
        Args:
            scan_dir: 输入目录
            depth_out_dir: 深度输出目录
            rotate: 是否旋转
            subsample: 采样间隔
            confidence_threshold: 置信度阈值
            
        Returns:
            写入的帧数
        """
        depth_dir_in = os.path.join(scan_dir, 'depth')
        confidence_dir = os.path.join(scan_dir, 'confidence')
        files = sorted(os.listdir(depth_dir_in))
        img_idx = 0
        
        for i, filename in tqdm(enumerate(files), desc="Writing Depth"):
            if '.png' not in filename:
                continue
            number, _ = filename.split('.')
            
            if i % subsample != 0:
                continue
            
            depth = cv2.imread(os.path.join(depth_dir_in, filename), -1)
            confidence = cv2.imread(
                os.path.join(confidence_dir, number + '.png')
            )[:, :, 0]
            
            if rotate:
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
                confidence = cv2.rotate(confidence, cv2.ROTATE_90_CLOCKWISE)
            
            # 根据置信度过滤深度值
            depth[confidence < confidence_threshold] = 0
            
            cv2.imwrite(
                os.path.join(depth_out_dir, f"{img_idx:05}.png"),
                depth
            )
            img_idx += 1
        
        return img_idx
    
    def _write_intrinsics(self, scan_dir: str, out_dir: str,
                         rotate: bool = False) -> None:
        """写入相机内参
        
        Args:
            scan_dir: 输入目录
            out_dir: 输出目录
            rotate: 是否旋转 (会交换 fx/fy 和 cx/cy)
        """
        intrinsics = np.loadtxt(
            os.path.join(scan_dir, 'camera_matrix.csv'),
            delimiter=','
        )
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        if rotate:
            out_intrinsics = np.array([
                [fy, 0, cy],
                [0, fx, cx],
                [0, 0, 1]
            ])
        else:
            out_intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        
        np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), out_intrinsics)


# 便捷函数
def import_scanner(input_path: str, output_path: str,
                  subsample: int = 1, rotate: bool = False,
                  confidence_threshold: int = 2) -> ImportResult:
    """导入 Scanner App 数据的便捷函数
    
    Args:
        input_path: Scanner App 数据集路径
        output_path: 输出目录路径
        subsample: 帧采样间隔
        rotate: 是否旋转 90 度
        confidence_threshold: 深度置信度阈值
        
    Returns:
        ImportResult 对象
    """
    importer = ScannerImporter()
    return importer.import_data(
        input_path, output_path,
        subsample=subsample,
        rotate=rotate,
        confidence_threshold=confidence_threshold
    )
