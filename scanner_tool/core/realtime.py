"""
实时语义分割服务

基于目录监控的实时分割服务，无 ROS 依赖。

支持:
- 监控输入目录的新帧
- 实时特征提取和分割
- 结果保存到输出目录
- 动态类别配置

复用来源:
- autolabel/scripts/ros/node.py (核心逻辑)

Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6
"""

import os
import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from queue import Queue

import numpy as np

# 可选依赖
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# 默认颜色表
DEFAULT_COLORS = np.array([
    [0, 0, 0],        # 背景
    [255, 0, 0],      # 类别 1
    [0, 255, 0],      # 类别 2
    [0, 0, 255],      # 类别 3
    [255, 255, 0],    # 类别 4
    [255, 0, 255],    # 类别 5
    [0, 255, 255],    # 类别 6
], dtype=np.uint8)


@dataclass
class RealtimeConfig:
    """实时服务配置"""
    input_dir: str = ""
    output_dir: str = ""
    feature_type: str = "clip"  # 'clip' or 'lseg'
    checkpoint: Optional[str] = None
    prompts: List[str] = field(default_factory=lambda: ["background", "object"])
    config_file: Optional[str] = None  # 动态配置文件
    save_features: bool = False
    colors: np.ndarray = field(default_factory=lambda: DEFAULT_COLORS)
    poll_interval: float = 0.1  # 轮询间隔 (秒)


@dataclass
class Frame:
    """帧数据"""
    name: str
    rgb: np.ndarray
    depth: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    timestamp: float = 0.0


def check_realtime_available() -> Dict[str, bool]:
    """检查实时服务可用性"""
    return {
        'torch': TORCH_AVAILABLE,
        'pil': PIL_AVAILABLE,
        'cv2': CV2_AVAILABLE,
        'watchdog': WATCHDOG_AVAILABLE,
        'available': TORCH_AVAILABLE and (PIL_AVAILABLE or CV2_AVAILABLE)
    }


def _get_device():
    """获取计算设备"""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FrameProcessor:
    """帧处理器
    
    处理单帧的特征提取和分割。
    """
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.device = _get_device()
        self.text_encoder = None
        self.text_features = None
        self._load_encoder()
    
    def _load_encoder(self):
        """加载文本编码器"""
        if self.config.feature_type == 'clip':
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load(
                    "ViT-B/32", device=self.device
                )
                self.clip_model.eval()
                self._update_prompts(self.config.prompts)
            except ImportError:
                print("Warning: CLIP not available")
        elif self.config.feature_type == 'lseg':
            if not torch.cuda.is_available():
                print("Warning: LSeg requires CUDA")
                return
            try:
                from autolabel.lseg import LSegNet
                self.lseg_model = LSegNet(
                    backbone='clip_vitl16_384',
                    features=256,
                    crop_size=480,
                    arch_option=0,
                    block_depth=0,
                    activation='lrelu'
                )
                if self.config.checkpoint:
                    self.lseg_model.load_state_dict(
                        torch.load(self.config.checkpoint, map_location='cpu')
                    )
                self.lseg_model = self.lseg_model.cuda().eval()
                self._update_prompts(self.config.prompts)
            except ImportError:
                print("Warning: LSeg not available")
    
    def _update_prompts(self, prompts: List[str]):
        """更新文本提示"""
        self.config.prompts = prompts
        
        if self.config.feature_type == 'clip' and hasattr(self, 'clip_model'):
            import clip
            with torch.no_grad():
                text_tokens = clip.tokenize(prompts).to(self.device)
                self.text_features = self.clip_model.encode_text(text_tokens)
                self.text_features = self.text_features / self.text_features.norm(
                    dim=-1, keepdim=True
                )
        elif self.config.feature_type == 'lseg' and hasattr(self, 'lseg_model'):
            with torch.no_grad():
                self.text_features = self.lseg_model.encode_text(prompts)
                self.text_features = self.text_features / self.text_features.norm(
                    dim=-1, keepdim=True
                )
    
    def extract_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """提取图像特征
        
        Args:
            image: RGB 图像 [H, W, 3]
            
        Returns:
            特征图 [H, W, D] 或 None
        """
        if self.config.feature_type == 'clip' and hasattr(self, 'clip_model'):
            return self._extract_clip_features(image)
        elif self.config.feature_type == 'lseg' and hasattr(self, 'lseg_model'):
            return self._extract_lseg_features(image)
        return None
    
    def _extract_clip_features(self, image: np.ndarray) -> np.ndarray:
        """提取 CLIP 特征"""
        # CLIP 需要特殊处理，这里简化为直接使用图像编码
        # 实际应用中可能需要更复杂的特征提取
        H, W = image.shape[:2]
        
        # 使用 CLIP 预处理
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        else:
            return np.zeros((H, W, 512), dtype=np.float32)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 扩展到图像尺寸 (简化版本)
        features = image_features.cpu().numpy()
        features = np.broadcast_to(features, (H, W, features.shape[-1]))
        return features.astype(np.float32)
    
    def _extract_lseg_features(self, image: np.ndarray) -> np.ndarray:
        """提取 LSeg 特征"""
        image_tensor = torch.tensor(
            image.transpose(2, 0, 1) / 255.0,
            device='cuda',
            dtype=torch.float32
        ).unsqueeze(0)
        
        with torch.no_grad():
            features = self.lseg_model(image_tensor)[0]
            features = F.normalize(features, dim=0)
        
        return features.permute(1, 2, 0).cpu().numpy()
    
    def segment(self, features: np.ndarray) -> np.ndarray:
        """基于特征进行分割
        
        Args:
            features: 特征图 [H, W, D]
            
        Returns:
            分割图 [H, W]
        """
        if self.text_features is None:
            return np.zeros(features.shape[:2], dtype=np.uint8)
        
        H, W, D = features.shape
        C = self.text_features.shape[0]
        
        # 归一化特征
        features_norm = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-8)
        
        # 计算相似度
        text_features_np = self.text_features.cpu().numpy()
        similarity = np.zeros((H, W, C), dtype=np.float32)
        for i in range(C):
            similarity[:, :, i] = (features_norm * text_features_np[i]).sum(axis=-1)
        
        return similarity.argmax(axis=-1).astype(np.uint8)
    
    def process_frame(self, frame: Frame) -> Dict[str, Any]:
        """处理单帧
        
        Args:
            frame: 帧数据
            
        Returns:
            处理结果
        """
        features = self.extract_features(frame.rgb)
        if features is None:
            return {'success': False, 'error': 'Feature extraction failed'}
        
        segmentation = self.segment(features)
        
        return {
            'success': True,
            'segmentation': segmentation,
            'features': features if self.config.save_features else None
        }


class DirectoryWatcher:
    """目录监控器
    
    监控输入目录的新文件。
    """
    
    def __init__(self, input_dir: str, callback: Callable[[str], None]):
        self.input_dir = Path(input_dir)
        self.callback = callback
        self.processed_files = set()
        self.running = False
        self.observer = None
    
    def start(self):
        """启动监控"""
        self.running = True
        
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            self._start_polling()
    
    def _start_watchdog(self):
        """使用 watchdog 监控"""
        class Handler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher
            
            def on_created(self, event):
                if not event.is_directory:
                    self.watcher._on_new_file(event.src_path)
        
        self.observer = Observer()
        self.observer.schedule(Handler(self), str(self.input_dir / 'rgb'), recursive=False)
        self.observer.start()
    
    def _start_polling(self):
        """使用轮询监控"""
        def poll():
            while self.running:
                rgb_dir = self.input_dir / 'rgb'
                if rgb_dir.exists():
                    for f in rgb_dir.iterdir():
                        if f.suffix in ['.png', '.jpg', '.jpeg']:
                            if str(f) not in self.processed_files:
                                self._on_new_file(str(f))
                time.sleep(0.1)
        
        self.poll_thread = threading.Thread(target=poll, daemon=True)
        self.poll_thread.start()
    
    def _on_new_file(self, path: str):
        """处理新文件"""
        if path in self.processed_files:
            return
        self.processed_files.add(path)
        self.callback(path)
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.observer:
            self.observer.stop()
            self.observer.join()


class RealtimeSegmentationService:
    """实时语义分割服务
    
    监控输入目录，处理新帧，保存结果。
    """
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        self.processor = FrameProcessor(config)
        self.watcher = None
        self.running = False
        self.frame_queue = Queue()
        self.process_thread = None
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        (self.output_dir / 'semantic').mkdir(parents=True, exist_ok=True)
        if config.save_features:
            (self.output_dir / 'features').mkdir(parents=True, exist_ok=True)
    
    @property
    def is_available(self) -> bool:
        """检查是否可用"""
        return TORCH_AVAILABLE and (PIL_AVAILABLE or CV2_AVAILABLE)
    
    def start(self):
        """启动服务"""
        if not self.is_available:
            raise RuntimeError("Service dependencies not available")
        
        self.running = True
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # 启动目录监控
        self.watcher = DirectoryWatcher(
            self.config.input_dir,
            self._on_new_frame
        )
        self.watcher.start()
        
        # 启动配置监控
        if self.config.config_file:
            self._start_config_watcher()
        
        print(f"Realtime segmentation service started")
        print(f"  Input: {self.config.input_dir}")
        print(f"  Output: {self.config.output_dir}")
        print(f"  Prompts: {self.config.prompts}")
    
    def _on_new_frame(self, rgb_path: str):
        """处理新帧"""
        self.frame_queue.put(rgb_path)
    
    def _process_loop(self):
        """处理循环"""
        while self.running:
            try:
                rgb_path = self.frame_queue.get(timeout=0.1)
                self._process_frame(rgb_path)
            except:
                continue
    
    def _process_frame(self, rgb_path: str):
        """处理单帧"""
        rgb_path = Path(rgb_path)
        frame_name = rgb_path.stem
        
        # 加载图像
        if PIL_AVAILABLE:
            rgb = np.array(Image.open(rgb_path))
        elif CV2_AVAILABLE:
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        else:
            return
        
        # 创建帧
        frame = Frame(name=frame_name, rgb=rgb)
        
        # 处理
        result = self.processor.process_frame(frame)
        
        if result['success']:
            # 保存分割结果
            seg_path = self.output_dir / 'semantic' / f"{frame_name}.png"
            if PIL_AVAILABLE:
                Image.fromarray(result['segmentation']).save(seg_path)
            elif CV2_AVAILABLE:
                cv2.imwrite(str(seg_path), result['segmentation'])
            
            # 保存彩色分割图
            colored = self.config.colors[result['segmentation'] % len(self.config.colors)]
            colored_path = self.output_dir / 'semantic' / f"{frame_name}_colored.png"
            if PIL_AVAILABLE:
                Image.fromarray(colored).save(colored_path)
            
            # 保存特征
            if self.config.save_features and result['features'] is not None:
                feat_path = self.output_dir / 'features' / f"{frame_name}.npy"
                np.save(feat_path, result['features'])
            
            print(f"Processed: {frame_name}")
    
    def _start_config_watcher(self):
        """启动配置文件监控"""
        def watch_config():
            last_mtime = 0
            while self.running:
                try:
                    config_path = Path(self.config.config_file)
                    if config_path.exists():
                        mtime = config_path.stat().st_mtime
                        if mtime > last_mtime:
                            last_mtime = mtime
                            self._reload_config()
                except:
                    pass
                time.sleep(1.0)
        
        self.config_thread = threading.Thread(target=watch_config, daemon=True)
        self.config_thread.start()
    
    def _reload_config(self):
        """重新加载配置"""
        try:
            with open(self.config.config_file, 'r') as f:
                data = json.load(f)
            
            if 'prompts' in data:
                new_prompts = data['prompts']
                if new_prompts != self.config.prompts:
                    print(f"Updating prompts: {new_prompts}")
                    self.processor._update_prompts(new_prompts)
        except Exception as e:
            print(f"Failed to reload config: {e}")
    
    def update_prompts(self, prompts: List[str]):
        """更新分割类别"""
        self.processor._update_prompts(prompts)
        print(f"Updated prompts: {prompts}")
    
    def stop(self):
        """停止服务"""
        self.running = False
        if self.watcher:
            self.watcher.stop()
        print("Realtime segmentation service stopped")
    
    def run(self):
        """运行服务 (阻塞)"""
        self.start()
        try:
            while self.running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def run_realtime_service(config: RealtimeConfig) -> bool:
    """运行实时分割服务
    
    Args:
        config: 服务配置
        
    Returns:
        是否成功运行
    """
    availability = check_realtime_available()
    if not availability['available']:
        print("Error: Realtime service dependencies not available")
        return False
    
    service = RealtimeSegmentationService(config)
    service.run()
    return True
