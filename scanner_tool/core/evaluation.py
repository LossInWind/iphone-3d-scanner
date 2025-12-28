"""
评估模块

支持:
- IoU (Intersection over Union) 计算
- 语义分割评估
- 结果可视化

复用来源:
- autolabel/scripts/evaluate.py
- autolabel/autolabel/evaluation.py

Requirements: 14.1, 14.2, 14.3
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np

# 可选依赖
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """评估配置"""
    ignore_background: bool = True
    num_classes: int = 10


@dataclass
class EvaluationResult:
    """评估结果"""
    success: bool
    iou_per_class: Dict[int, float] = field(default_factory=dict)
    mean_iou: float = 0.0
    pixel_accuracy: float = 0.0
    num_frames: int = 0
    error_message: str = ""
    
    def __str__(self):
        if not self.success:
            return f"Evaluation failed: {self.error_message}"
        
        lines = [
            f"Evaluation Results:",
            f"  Mean IoU: {self.mean_iou:.4f}",
            f"  Pixel Accuracy: {self.pixel_accuracy:.4f}",
            f"  Frames evaluated: {self.num_frames}",
            f"  IoU per class:"
        ]
        for class_id, iou in sorted(self.iou_per_class.items()):
            lines.append(f"    Class {class_id}: {iou:.4f}")
        return "\n".join(lines)


def check_evaluation_available() -> Dict[str, bool]:
    """检查评估功能可用性"""
    return {
        'pil': PIL_AVAILABLE,
        'cv2': CV2_AVAILABLE,
        'available': PIL_AVAILABLE or CV2_AVAILABLE
    }


class SemanticEvaluator:
    """语义分割评估器
    
    计算预测结果与真值之间的 IoU 和像素准确率。
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return PIL_AVAILABLE or CV2_AVAILABLE
    
    def compute_iou(self, pred: np.ndarray, gt: np.ndarray, 
                    class_id: int) -> float:
        """计算单个类别的 IoU
        
        Args:
            pred: 预测分割图 [H, W]
            gt: 真值分割图 [H, W]
            class_id: 类别 ID
            
        Returns:
            IoU 值
        """
        pred_mask = pred == class_id
        gt_mask = gt == class_id
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_pixel_accuracy(self, pred: np.ndarray, 
                               gt: np.ndarray) -> float:
        """计算像素准确率
        
        Args:
            pred: 预测分割图 [H, W]
            gt: 真值分割图 [H, W]
            
        Returns:
            像素准确率
        """
        return (pred == gt).mean()
    
    def evaluate_frame(self, pred: np.ndarray, 
                       gt: np.ndarray) -> Dict[int, float]:
        """评估单帧
        
        Args:
            pred: 预测分割图 [H, W]
            gt: 真值分割图 [H, W]
            
        Returns:
            每个类别的 IoU
        """
        # 获取所有类别
        all_classes = np.unique(np.concatenate([pred.flatten(), gt.flatten()]))
        
        iou_per_class = {}
        for class_id in all_classes:
            if self.config.ignore_background and class_id == 0:
                continue
            iou_per_class[int(class_id)] = self.compute_iou(pred, gt, class_id)
        
        return iou_per_class
    
    def evaluate_scene(self, pred_dir: str, 
                       gt_dir: str) -> EvaluationResult:
        """评估整个场景
        
        Args:
            pred_dir: 预测结果目录
            gt_dir: 真值目录
            
        Returns:
            评估结果
        """
        if not self.is_available:
            return EvaluationResult(
                success=False,
                error_message="Evaluation dependencies not available"
            )
        
        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)
        
        if not pred_dir.exists():
            return EvaluationResult(
                success=False,
                error_message=f"Prediction directory not found: {pred_dir}"
            )
        
        if not gt_dir.exists():
            return EvaluationResult(
                success=False,
                error_message=f"Ground truth directory not found: {gt_dir}"
            )
        
        # 收集所有帧的 IoU
        all_ious: Dict[int, List[float]] = {}
        all_accuracies = []
        num_frames = 0
        
        for pred_file in sorted(pred_dir.glob('*.png')):
            gt_file = gt_dir / pred_file.name
            if not gt_file.exists():
                continue
            
            # 读取图像
            pred = self._load_image(pred_file)
            gt = self._load_image(gt_file)
            
            if pred is None or gt is None:
                continue
            
            # 确保尺寸一致
            if pred.shape != gt.shape:
                if PIL_AVAILABLE:
                    pred = np.array(Image.fromarray(pred).resize(
                        (gt.shape[1], gt.shape[0]), Image.NEAREST
                    ))
                elif CV2_AVAILABLE:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            
            # 计算 IoU
            frame_ious = self.evaluate_frame(pred, gt)
            for class_id, iou in frame_ious.items():
                if class_id not in all_ious:
                    all_ious[class_id] = []
                all_ious[class_id].append(iou)
            
            # 计算像素准确率
            all_accuracies.append(self.compute_pixel_accuracy(pred, gt))
            num_frames += 1
        
        if num_frames == 0:
            return EvaluationResult(
                success=False,
                error_message="No matching frames found"
            )
        
        # 计算平均 IoU
        iou_per_class = {
            class_id: np.mean(ious) 
            for class_id, ious in all_ious.items()
        }
        mean_iou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0
        pixel_accuracy = np.mean(all_accuracies)
        
        return EvaluationResult(
            success=True,
            iou_per_class=iou_per_class,
            mean_iou=mean_iou,
            pixel_accuracy=pixel_accuracy,
            num_frames=num_frames
        )
    
    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """加载图像"""
        if PIL_AVAILABLE:
            return np.array(Image.open(path))
        elif CV2_AVAILABLE:
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return None
    
    def save_results(self, result: EvaluationResult, 
                     output_path: str):
        """保存评估结果
        
        Args:
            result: 评估结果
            output_path: 输出文件路径 (JSON)
        """
        data = {
            'success': result.success,
            'mean_iou': result.mean_iou,
            'pixel_accuracy': result.pixel_accuracy,
            'num_frames': result.num_frames,
            'iou_per_class': result.iou_per_class,
            'error_message': result.error_message
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


class ConfusionMatrix:
    """混淆矩阵
    
    用于详细分析分割结果。
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """更新混淆矩阵
        
        Args:
            pred: 预测分割图 [H, W]
            gt: 真值分割图 [H, W]
        """
        mask = (gt >= 0) & (gt < self.num_classes)
        self.matrix += np.bincount(
            self.num_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
    
    def get_iou(self) -> np.ndarray:
        """获取每个类别的 IoU"""
        intersection = np.diag(self.matrix)
        union = self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - intersection
        iou = intersection / np.maximum(union, 1)
        return iou
    
    def get_accuracy(self) -> float:
        """获取总体准确率"""
        return np.diag(self.matrix).sum() / max(self.matrix.sum(), 1)
    
    def get_class_accuracy(self) -> np.ndarray:
        """获取每个类别的准确率"""
        return np.diag(self.matrix) / np.maximum(self.matrix.sum(axis=1), 1)
    
    def reset(self):
        """重置混淆矩阵"""
        self.matrix.fill(0)
