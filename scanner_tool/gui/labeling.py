"""
语义标注 GUI 模块

基于 PyQt6 的交互式语义标注界面。
支持:
- 手动绘制语义标签
- 实时 NeRF 推理预览 (可选)
- 多类别标注
- 保存/加载标注

复用来源:
- autolabel/scripts/gui.py
- autolabel/autolabel/ui/canvas.py

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable, Any

import numpy as np

# 检查 PyQt6 是否可用
try:
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QPushButton, 
        QVBoxLayout, QHBoxLayout, QLabel, QSlider,
        QGraphicsView, QGraphicsScene, QSizePolicy,
        QFileDialog, QMessageBox, QStatusBar, QToolBar,
        QSpinBox, QComboBox, QGroupBox
    )
    from PyQt6 import QtCore, QtGui
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QAction
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

try:
    from PIL import Image
    from PIL.ImageQt import ImageQt, fromqimage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# 默认颜色表 (与 autolabel 兼容)
DEFAULT_COLORS = np.array([
    [255, 0, 0],      # 类别 1: 红色
    [0, 255, 0],      # 类别 2: 绿色
    [0, 0, 255],      # 类别 3: 蓝色
    [255, 255, 0],    # 类别 4: 黄色
    [255, 0, 255],    # 类别 5: 品红
    [0, 255, 255],    # 类别 6: 青色
    [128, 0, 0],      # 类别 7: 深红
    [0, 128, 0],      # 类别 8: 深绿
    [0, 0, 128],      # 类别 9: 深蓝
    [128, 128, 0],    # 类别 10: 橄榄
], dtype=np.uint8)

ALPHA = 175  # 标注透明度


@dataclass
class LabelingConfig:
    """标注配置"""
    brush_size: int = 5
    alpha: int = ALPHA
    colors: np.ndarray = field(default_factory=lambda: DEFAULT_COLORS)
    canvas_width: int = 720
    auto_save: bool = True
    inference_interval: int = 5000  # 推理更新间隔 (ms)


def check_gui_available() -> Dict[str, bool]:
    """检查 GUI 功能可用性"""
    return {
        'pyqt6': PYQT_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'available': PYQT_AVAILABLE and PIL_AVAILABLE
    }


if PYQT_AVAILABLE and PIL_AVAILABLE:
    
    def _bitmap_to_color(array: np.ndarray, colors: np.ndarray = DEFAULT_COLORS, alpha: int = ALPHA) -> np.ndarray:
        """将位图转换为彩色图像
        
        Args:
            array: 类别位图 [H, W]
            colors: 颜色表 [N, 3]
            alpha: 透明度
            
        Returns:
            RGBA 图像 [H, W, 4]
        """
        alpha_colors = np.zeros((colors.shape[0] + 1, 4), dtype=np.uint8)
        alpha_colors[0] = np.array([0., 0., 0., 0.])
        alpha_colors[1:, :3] = colors
        alpha_colors[1:, 3] = alpha
        return alpha_colors[array]
    
    
    class LabelingCanvas(QWidget):
        """标注画布组件
        
        支持:
        - 鼠标绘制标注
        - 多类别切换
        - 推理结果叠加显示
        """
        
        def __init__(self, width: int, height: int, callback: Callable = None,
                     config: LabelingConfig = None):
            super().__init__()
            self.config = config or LabelingConfig()
            self.canvas_width = int(width)
            self.canvas_height = int(height)
            self.brush_size = self.config.brush_size
            self.active = False
            self.callback = callback
            
            # 创建图形视图
            self.g_view = QGraphicsView(self)
            self.g_view.setSceneRect(0, 0, self.canvas_width, self.canvas_height)
            self.g_view.setBackgroundBrush(
                QtGui.QBrush(QColor(52, 52, 52), Qt.BrushStyle.SolidPattern)
            )
            self.g_scene = QGraphicsScene(0, 0, width, height)
            self.g_view.setScene(self.g_scene)
            
            # 绑定鼠标事件
            self.g_view.mousePressEvent = self._mouse_down
            self.g_view.mouseReleaseEvent = self._mouse_up
            self.g_view.mouseMoveEvent = self._mouse_move
            
            # 初始化状态
            self.drawing = None
            self.canvas = None
            self.canvas_pixmap = None
            self.scene_image = None
            self.active_class = 1
            self.bitmap_painter = None
            self.color_painter = None
            self.inferred_image = None
            
            # 颜色
            self.qt_colors = [
                QColor(c[0], c[1], c[2], self.config.alpha) 
                for c in self.config.colors
            ]
        
        @property
        def color(self):
            """当前绘制颜色"""
            return self.qt_colors[self.active_class - 1] if self.active_class > 0 else QColor(0, 0, 0, 0)
        
        def _mouse_down(self, event):
            """鼠标按下"""
            self.active = True
            self.lastpoint = self._scale(event.pos())
            self._draw_point(self.lastpoint)
            self._changed()
        
        def _mouse_up(self, event):
            """鼠标释放"""
            self.active = False
            if self.callback:
                self.callback()
        
        def _mouse_move(self, event):
            """鼠标移动"""
            if event.buttons() & Qt.MouseButton.LeftButton and self.active:
                self._draw_line(self.lastpoint, self._scale(event.pos()))
                self.lastpoint = self._scale(event.pos())
                self._changed()
        
        def set_image(self, image: Image.Image, drawing: QImage):
            """设置背景图像和绘制层
            
            Args:
                image: PIL 图像
                drawing: Qt 绘制图像
            """
            self.bitmap_painter = None
            self.color_painter = None
            self.drawing = drawing
            self.image = ImageQt(image)
            
            # 转换绘制层为彩色
            array = np.asarray(fromqimage(drawing))[:, :, 0]
            color_array = _bitmap_to_color(array, self.config.colors, self.config.alpha)
            self.canvas = QPixmap.fromImage(ImageQt(Image.fromarray(color_array)))
            
            self.image_width = image.width
            self.image_height = image.height
            self._image_changed()
        
        def _image_changed(self):
            """图像更新"""
            if self.scene_image is not None:
                self.g_scene.removeItem(self.scene_image)
            if self.canvas_pixmap is not None:
                self.g_scene.removeItem(self.canvas_pixmap)
            if self.inferred_image is not None:
                self.g_scene.removeItem(self.inferred_image)
                self.inferred_image = None
            
            self.scene_image = self.g_scene.addPixmap(QPixmap.fromImage(self.image))
            self.canvas_pixmap = self.g_scene.addPixmap(self.canvas)
            self.canvas_pixmap.setZValue(2.0)
            self.scene_image.setScale(self.canvas_width / self.image_width)
            self.update()
            self.set_class(self.active_class)
        
        def _changed(self):
            """画布更新"""
            self.canvas_pixmap.update()
            self.canvas_pixmap.setPixmap(self.canvas)
            self.g_view.update()
            self.update()
        
        def _scale(self, point):
            """将视图坐标转换为画布坐标"""
            return self.g_view.mapToScene(point)
        
        def _draw_point(self, point):
            """绘制点"""
            if self.bitmap_painter and self.color_painter:
                self.bitmap_painter.drawPoint(point)
                self.color_painter.drawPoint(point)
        
        def _draw_line(self, start, end):
            """绘制线"""
            if self.bitmap_painter and self.color_painter:
                self.bitmap_painter.drawLine(start, end)
                self.color_painter.drawLine(start, end)
        
        def set_class(self, class_index: int):
            """设置当前绘制类别
            
            Args:
                class_index: 类别索引 (1-based)
            """
            self.active_class = class_index
            
            # 清理旧的画笔
            self.bitmap_painter = None
            self.color_painter = None
            
            if self.drawing is None or self.canvas is None:
                return
            
            self.bitmap_painter = QPainter(self.drawing)
            self.color_painter = QPainter(self.canvas)
            
            # 位图画笔 (存储类别索引)
            bitpen = QPen(
                QColor(self.active_class, self.active_class, self.active_class),
                self.brush_size,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
            
            # 颜色画笔 (显示用)
            color_pen = QPen(
                self.color,
                self.brush_size,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin
            )
            
            self.bitmap_painter.setPen(bitpen)
            self.bitmap_painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Source
            )
            self.color_painter.setPen(color_pen)
            self.color_painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_Source
            )
        
        def set_inferred(self, image: np.ndarray):
            """设置推理结果叠加显示
            
            Args:
                image: 类别图 [H, W]
            """
            colored = self.config.colors[image]
            alpha = np.ones_like(colored[:, :, :1]) * 120
            colored = np.concatenate([colored, alpha], axis=-1)
            pil_image = Image.fromarray(colored).resize(
                (self.canvas_width, self.canvas_height), Image.NEAREST
            )
            pixmap = QPixmap.fromImage(ImageQt(pil_image))
            
            if self.inferred_image is not None:
                self.g_scene.removeItem(self.inferred_image)
            self.inferred_image = self.g_scene.addPixmap(pixmap)
            self.inferred_image.setZValue(1.0)
        
        def minimumSizeHint(self):
            return QtCore.QSize(self.canvas_width, self.canvas_height)
        
        def resizeEvent(self, event):
            self._size_changed(event.size())
        
        def showEvent(self, event):
            self._size_changed(self.size())
        
        def _size_changed(self, size):
            self.g_view.setFixedWidth(size.width())
            self.g_view.setFixedHeight(size.height())
            self.g_view.fitInView(
                0, 0, self.canvas_width, self.canvas_height,
                Qt.AspectRatioMode.KeepAspectRatio
            )
    
    
    class LabelingWindow(QMainWindow):
        """标注主窗口
        
        提供完整的标注界面，包括:
        - 图像浏览
        - 标注绘制
        - 类别切换
        - 保存/加载
        """
        
        def __init__(self, scene_path: str, config: LabelingConfig = None):
            super().__init__()
            self.scene_path = Path(scene_path)
            self.config = config or LabelingConfig()
            
            self.setWindowTitle(f"Scanner Tool - Labeling: {self.scene_path.name}")
            
            # 加载场景数据
            self._load_scene()
            
            # 创建 UI
            self._setup_ui()
            
            # 加载已有标注
            self._load_annotations()
            
            # 显示第一张图像
            if len(self.image_names) > 0:
                self._set_image(0)
        
        def _load_scene(self):
            """加载场景数据"""
            rgb_dir = self.scene_path / 'rgb'
            if not rgb_dir.exists():
                raise ValueError(f"RGB directory not found: {rgb_dir}")
            
            self.rgb_paths = sorted(
                [f for f in rgb_dir.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']],
                key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
            )
            self.image_names = [p.stem for p in self.rgb_paths]
            
            if len(self.rgb_paths) == 0:
                raise ValueError("No RGB images found")
            
            # 获取图像尺寸
            sample_image = Image.open(self.rgb_paths[0])
            self.image_size = sample_image.size  # (W, H)
            
            # 缓存
            self._image_cache = {}
            self._drawings = {}
        
        def _setup_ui(self):
            """设置 UI"""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # 计算画布尺寸
            W, H = self.image_size
            canvas_width = self.config.canvas_width
            canvas_height = int(canvas_width / W * H)
            
            # 创建画布
            self.canvas = LabelingCanvas(
                canvas_width, canvas_height,
                callback=self._on_canvas_changed,
                config=self.config
            )
            
            # 创建滑块
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.rgb_paths) - 1)
            self.slider.valueChanged.connect(self._on_slider_changed)
            
            # 创建类别标签
            self.class_label = QLabel(f"Current class: {self.canvas.active_class}")
            
            # 创建状态栏
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage(f"Loaded {len(self.rgb_paths)} images")
            
            # 布局
            bottom_bar = QHBoxLayout()
            bottom_bar.addWidget(self.slider)
            bottom_bar.addWidget(self.class_label)
            
            main_layout = QVBoxLayout()
            main_layout.addWidget(self.canvas)
            main_layout.addLayout(bottom_bar)
            
            central_widget.setLayout(main_layout)
        
        def _load_annotations(self):
            """加载已有标注"""
            semantic_dir = self.scene_path / 'semantic'
            if not semantic_dir.exists():
                return
            
            for image_file in semantic_dir.iterdir():
                if image_file.suffix == '.png':
                    image_name = image_file.stem
                    array = np.array(Image.open(image_file)).astype(np.uint8)
                    array = np.repeat(array[:, :, None], 3, axis=2)
                    self._drawings[image_name] = ImageQt(Image.fromarray(array))
        
        def _set_image(self, index: int):
            """设置当前图像
            
            Args:
                index: 图像索引
            """
            self.current_index = index
            self.current_image_name = self.image_names[index]
            
            # 加载图像
            if self.current_image_name not in self._image_cache:
                self._image_cache[self.current_image_name] = Image.open(
                    self.rgb_paths[index]
                )
            
            # 获取或创建绘制层
            if self.current_image_name not in self._drawings:
                drawing = QImage(
                    self.canvas.canvas_width,
                    self.canvas.canvas_height,
                    QImage.Format.Format_RGB888
                )
                drawing.fill(0)
                self._drawings[self.current_image_name] = drawing
            
            image = self._image_cache[self.current_image_name]
            drawing = self._drawings[self.current_image_name]
            self.canvas.set_image(image, drawing)
            
            self.status_bar.showMessage(
                f"Image {index + 1}/{len(self.rgb_paths)}: {self.current_image_name}"
            )
        
        def _on_slider_changed(self):
            """滑块值改变"""
            self._set_image(self.slider.value())
        
        def _on_canvas_changed(self):
            """画布内容改变"""
            if self.config.auto_save:
                self._save_current_annotation()
        
        def _save_current_annotation(self):
            """保存当前标注"""
            semantic_dir = self.scene_path / 'semantic'
            semantic_dir.mkdir(exist_ok=True)
            
            drawing = self._drawings.get(self.current_image_name)
            if drawing is None:
                return
            
            array = np.asarray(fromqimage(drawing))[:, :, 0]
            if array.max() == 0:
                return  # 空标注，跳过
            
            path = semantic_dir / f"{self.current_image_name}.png"
            Image.fromarray(array).save(path)
        
        def save_all(self):
            """保存所有标注"""
            for image_name in self._drawings.keys():
                self._save_annotation(image_name)
        
        def _save_annotation(self, image_name: str):
            """保存指定图像的标注"""
            semantic_dir = self.scene_path / 'semantic'
            semantic_dir.mkdir(exist_ok=True)
            
            drawing = self._drawings.get(image_name)
            if drawing is None:
                return
            
            array = np.asarray(fromqimage(drawing))[:, :, 0]
            if array.max() == 0:
                return
            
            path = semantic_dir / f"{image_name}.png"
            Image.fromarray(array).save(path)
        
        def clear_current(self):
            """清除当前图像的标注"""
            drawing = QImage(
                self.canvas.canvas_width,
                self.canvas.canvas_height,
                QImage.Format.Format_RGB888
            )
            drawing.fill(0)
            self._drawings[self.current_image_name] = drawing
            self._set_image(self.current_index)
        
        def set_class(self, class_index: int):
            """设置当前类别"""
            self.canvas.set_class(class_index)
            self.class_label.setText(f"Current class: {class_index}")
        
        def keyPressEvent(self, event):
            """键盘事件"""
            key = event.key()
            modifiers = QApplication.keyboardModifiers()
            
            # 退出
            if key == Qt.Key.Key_Escape or key == Qt.Key.Key_Q:
                self.close()
            
            # 数字键切换类别
            elif key >= Qt.Key.Key_0 and key <= Qt.Key.Key_9:
                class_index = key - Qt.Key.Key_0
                if class_index == 0:
                    class_index = 10  # 0 键对应类别 10
                self.set_class(class_index)
            
            # Ctrl+S 保存
            elif key == Qt.Key.Key_S and modifiers == Qt.KeyboardModifier.ControlModifier:
                self.save_all()
                self.status_bar.showMessage("All annotations saved")
            
            # C 清除当前
            elif key == Qt.Key.Key_C:
                self.clear_current()
            
            # 左右箭头切换图像
            elif key == Qt.Key.Key_Left:
                if self.current_index > 0:
                    self.slider.setValue(self.current_index - 1)
            elif key == Qt.Key.Key_Right:
                if self.current_index < len(self.rgb_paths) - 1:
                    self.slider.setValue(self.current_index + 1)
        
        def closeEvent(self, event):
            """关闭事件"""
            self.save_all()
            event.accept()


def run_labeling_gui(scene_path: str, config: LabelingConfig = None) -> bool:
    """运行标注 GUI
    
    Args:
        scene_path: 场景目录路径
        config: 标注配置
        
    Returns:
        是否成功运行
    """
    availability = check_gui_available()
    if not availability['available']:
        print("Error: GUI dependencies not available")
        if not availability['pyqt6']:
            print("  - PyQt6 not installed: pip install PyQt6")
        if not availability['pil']:
            print("  - PIL not installed: pip install Pillow")
        return False
    
    app = QApplication(sys.argv)
    
    try:
        window = LabelingWindow(scene_path, config)
        window.show()
        return app.exec() == 0
    except Exception as e:
        print(f"Error: {e}")
        return False
