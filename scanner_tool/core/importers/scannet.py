"""ScanNet 数据导入器

复用自: autolabel/scripts/convert_scannet.py

支持从 ScanNet 数据集转换为统一内部格式。

注意: 此模块需要 imageio 库。如果未安装，导入功能将不可用。
"""

import os
import math
import struct
import zlib
import json
import numpy as np
import cv2

from .base import BaseImporter, ImportResult

# 尝试导入 imageio，如果不可用则设置标志
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    imageio = None


def _check_imageio():
    """检查 imageio 是否可用"""
    if not IMAGEIO_AVAILABLE:
        raise ImportError(
            "imageio is required for ScanNet import. "
            "Please install it with: pip install imageio"
        )


class RGBDFrame:
    """RGB-D 帧数据结构"""
    
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                         file_handle.read(self.color_size_bytes))
        )
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                         file_handle.read(self.depth_size_bytes))
        )


class SensReader:
    """ScanNet .sens 文件读取器"""
    
    def __init__(self, sens_file: str):
        self.file = sens_file
        self.file_handle = None
        self.num_frames = None
        self.rgb_size = None
        self.depth_size = None
    
    def __enter__(self):
        self.file_handle = open(self.file, 'rb')
        f = self.file_handle
        version = struct.unpack('I', f.read(4))[0]
        assert version == 4, f"Unsupported .sens version: {version}"
        
        strlen = struct.unpack('Q', f.read(8))[0]
        self.sensor_name = ''.join([
            c.decode('utf-8')
            for c in struct.unpack('c' * strlen, f.read(strlen))
        ])
        
        self.intrinsic_color = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_color = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.intrinsic_depth = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.extrinsic_depth = np.asarray(
            struct.unpack('f' * 16, f.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        
        color_compression_type = struct.unpack('i', f.read(4))[0]
        depth_compression_type = struct.unpack('i', f.read(4))[0]
        color_width = struct.unpack('I', f.read(4))[0]
        color_height = struct.unpack('I', f.read(4))[0]
        self.rgb_size = (color_width, color_height)
        depth_width = struct.unpack('I', f.read(4))[0]
        depth_height = struct.unpack('I', f.read(4))[0]
        self.depth_size = (depth_width, depth_height)
        depth_shift = struct.unpack('f', f.read(4))[0]
        self.num_frames = struct.unpack('Q', f.read(8))[0]
        
        return self
    
    def __exit__(self, *args):
        self.file_handle.close()
    
    def read(self):
        """迭代读取帧"""
        _check_imageio()
        for i in range(self.num_frames):
            frame = RGBDFrame()
            frame.load(self.file_handle)
            rgb_frame = imageio.v3.imread(frame.color_data)
            depth_frame = zlib.decompress(frame.depth_data)
            depth_frame = np.frombuffer(depth_frame, dtype=np.uint16).reshape(
                self.depth_size[1], self.depth_size[0]
            )
            yield frame.camera_to_world, rgb_frame, depth_frame


class ScanNetImporter(BaseImporter):
    """ScanNet 数据导入器
    
    将 ScanNet 数据集转换为统一的内部格式。
    
    输入格式:
        scannet_dir/
        └── <scene_id>/
            ├── <scene_id>.sens     # 传感器数据
            └── label-filt/         # 语义标签 (可选)
    
    输出格式:
        output_dir/
        └── <scene_id>/
            ├── rgb/           # RGB 帧 (JPEG)
            ├── depth/         # 深度帧 (PNG, 16-bit)
            ├── pose/          # 相机位姿 (TXT)
            ├── gt_semantic/   # 语义标签 (可选)
            └── intrinsics.txt # 相机内参
    """
    
    def validate_input(self, input_path: str) -> bool:
        """验证 ScanNet 数据集是否有效"""
        # 检查是否存在 .sens 文件
        scene_name = os.path.basename(input_path)
        sens_file = os.path.join(input_path, f"{scene_name}.sens")
        return os.path.exists(sens_file)
    
    def import_data(self, input_path: str, output_path: str,
                   stride: int = 5, max_frames: int = 750) -> ImportResult:
        """导入 ScanNet 数据
        
        Args:
            input_path: ScanNet 场景目录路径
            output_path: 输出目录路径
            stride: 帧采样步长
            max_frames: 最大帧数
            
        Returns:
            ImportResult 对象
        """
        _check_imageio()
        
        if not self.validate_input(input_path):
            return ImportResult(
                success=False,
                output_path=output_path,
                error_message=f"Invalid ScanNet dataset: {input_path}"
            )
        
        try:
            scene_name = os.path.basename(input_path)
            sens_file = os.path.join(input_path, f"{scene_name}.sens")
            
            # 创建输出目录
            rgb_dir = os.path.join(output_path, 'rgb')
            depth_dir = os.path.join(output_path, 'depth')
            pose_dir = os.path.join(output_path, 'pose')
            self._ensure_output_dirs(output_path, ['rgb', 'depth', 'pose'])
            
            rgb_count = 0
            depth_count = 0
            pose_count = 0
            
            with SensReader(sens_file) as reader:
                # 写入内参
                intrinsics = reader.intrinsic_color
                np.savetxt(os.path.join(output_path, 'intrinsics.txt'), intrinsics)
                
                # 计算实际步长
                actual_stride = max(
                    math.ceil(reader.num_frames / max_frames),
                    stride
                )
                
                for i, (T_WC, rgb, depth) in enumerate(reader.read()):
                    if i % actual_stride != 0:
                        continue
                    
                    # 跳过无效位姿
                    if np.isnan(T_WC).any() or np.isinf(T_WC).any():
                        continue
                    
                    T_CW = np.linalg.inv(T_WC)
                    number = f"{i:06}"
                    
                    rgb_path = os.path.join(rgb_dir, f"{number}.jpg")
                    depth_path = os.path.join(depth_dir, f"{number}.png")
                    pose_path = os.path.join(pose_dir, f"{number}.txt")
                    
                    imageio.imwrite(rgb_path, rgb)
                    cv2.imwrite(depth_path, depth)
                    np.savetxt(pose_path, T_CW)
                    
                    rgb_count += 1
                    depth_count += 1
                    pose_count += 1
            
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


# 便捷函数
def import_scannet(input_path: str, output_path: str,
                  stride: int = 5, max_frames: int = 750) -> ImportResult:
    """导入 ScanNet 数据的便捷函数"""
    importer = ScanNetImporter()
    return importer.import_data(
        input_path, output_path,
        stride=stride,
        max_frames=max_frames
    )
