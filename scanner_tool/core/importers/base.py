"""基础导入器接口定义"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import os


@dataclass
class ImportResult:
    """导入结果"""
    success: bool
    output_path: str
    rgb_count: int = 0
    depth_count: int = 0
    pose_count: int = 0
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        if self.success:
            return (f"Import successful: {self.rgb_count} RGB frames, "
                    f"{self.depth_count} depth frames, {self.pose_count} poses")
        return f"Import failed: {self.error_message}"


class BaseImporter(ABC):
    """数据导入器基类"""
    
    @abstractmethod
    def validate_input(self, input_path: str) -> bool:
        """验证输入数据集是否有效
        
        Args:
            input_path: 输入数据集路径
            
        Returns:
            True 如果数据集有效，否则 False
        """
        pass
    
    @abstractmethod
    def import_data(self, input_path: str, output_path: str, **kwargs) -> ImportResult:
        """导入数据到统一格式
        
        Args:
            input_path: 输入数据集路径
            output_path: 输出目录路径
            **kwargs: 额外参数
            
        Returns:
            ImportResult 对象
        """
        pass
    
    def _ensure_output_dirs(self, output_path: str, 
                           dirs: List[str] = None) -> None:
        """确保输出目录存在
        
        Args:
            output_path: 输出根目录
            dirs: 需要创建的子目录列表
        """
        os.makedirs(output_path, exist_ok=True)
        if dirs:
            for d in dirs:
                os.makedirs(os.path.join(output_path, d), exist_ok=True)
    
    def _count_files(self, directory: str, extension: str = None) -> int:
        """统计目录中的文件数量
        
        Args:
            directory: 目录路径
            extension: 文件扩展名过滤 (如 '.png')
            
        Returns:
            文件数量
        """
        if not os.path.exists(directory):
            return 0
        files = os.listdir(directory)
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return len(files)
