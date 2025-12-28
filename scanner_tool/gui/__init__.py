"""GUI 模块"""

from .labeling import (
    LabelingConfig,
    check_gui_available,
    run_labeling_gui,
)

# 条件导入 (仅当 PyQt6 可用时)
try:
    from .labeling import LabelingCanvas, LabelingWindow
except ImportError:
    LabelingCanvas = None
    LabelingWindow = None

__all__ = [
    'LabelingConfig',
    'check_gui_available',
    'run_labeling_gui',
    'LabelingCanvas',
    'LabelingWindow',
]
