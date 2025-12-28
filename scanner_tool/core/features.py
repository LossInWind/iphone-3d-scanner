"""
特征提取模块 - 封装 DINO 和 LSeg 特征提取功能

跨平台支持:
- NVIDIA GPU: 使用 CUDA
- Apple Silicon: 使用 MPS
- CPU: 回退到 CPU (慢)

复用来源:
- autolabel/scripts/compute_feature_maps.py
- autolabel/autolabel/features/dino.py
- autolabel/autolabel/features/lseg.py

Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3
"""

import os
import math
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from pathlib import Path

import numpy as np
import h5py


def _get_device():
    """获取最佳可用设备"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    except ImportError:
        return None


def _get_device_string():
    """获取设备字符串"""
    device = _get_device()
    return str(device) if device else 'cpu'


@dataclass
class FeatureExtractionConfig:
    """特征提取配置"""
    feature_type: str = 'dino'  # 'dino' or 'lseg'
    output_dim: int = 64  # 输出特征维度
    autoencode: bool = True  # 是否使用自编码器压缩
    checkpoint: Optional[str] = None  # LSeg 模型检查点路径
    batch_size: int = 2  # 批处理大小
    target_size_dino: int = 720  # DINO 目标短边尺寸
    target_size_lseg: int = 242  # LSeg 目标短边尺寸


@dataclass
class FeatureExtractionResult:
    """特征提取结果"""
    success: bool
    output_path: str
    feature_type: str
    num_frames: int
    feature_shape: Tuple[int, ...]
    pca_available: bool = False
    error_message: Optional[str] = None


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""
    
    @abstractmethod
    def shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """计算输出特征图尺寸"""
        pass
    
    @abstractmethod
    def __call__(self, images):
        """提取特征
        
        Args:
            images: 图像张量 [B, C, H, W]
            
        Returns:
            特征张量 [B, H', W', C']
        """
        pass


class DinoFeatureExtractor(BaseFeatureExtractor):
    """DINO 特征提取器
    
    使用 Facebook 的 DINO 自监督视觉模型提取特征。
    支持 CUDA、MPS 和 CPU。
    
    复用: autolabel/autolabel/features/dino.py
    
    Requirements: 8.1, 8.2
    """
    
    def __init__(self):
        """初始化 DINO 特征提取器"""
        self._available = False
        self._error = None
        self.device = _get_device()
        
        if self.device is None:
            self._error = "PyTorch not available"
            return
        
        try:
            import torch
            from torchvision import transforms
            
            self.torch = torch
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ).to(self.device)
            
            # 加载预训练 DINO 模型
            self.model = torch.hub.load(
                'facebookresearch/dino:main',
                'dino_vits8'
            )
            
            # MPS 不支持 half()，只在 CUDA 上使用
            if self.device.type == 'cuda':
                self.model = self.model.half()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._available = True
        except Exception as e:
            self._error = str(e)
    
    @property
    def is_available(self) -> bool:
        """检查 DINO 是否可用"""
        return self._available
    
    def shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """计算输出特征图尺寸
        
        DINO ViT-S/8 使用 8x8 patch，输出尺寸为输入的 1/8
        """
        return (90, 120)  # 固定输出尺寸
    
    def __call__(self, x):
        """提取 DINO 特征
        
        Args:
            x: 图像张量 [B, C, H, W]，值范围 [0, 1]
            
        Returns:
            特征张量 [B, H', W', 384]
        """
        if not self._available:
            raise RuntimeError(f"DINO not available: {self._error}")
        
        B, C, H, W = x.shape
        x = x.to(self.device)
        x = self.normalize(x)
        
        # MPS 不支持 half()
        if self.device.type == 'cuda':
            x = x.half()
        
        x = self.model.get_intermediate_layers(x)
        
        width_out = W // 8
        height_out = H // 8
        
        return x[0][:, 1:, :].reshape(B, height_out, width_out, 384).detach().cpu()


class LSegFeatureExtractor(BaseFeatureExtractor):
    """LSeg 特征提取器
    
    使用 LSeg 视觉-语言模型提取特征，支持文本查询。
    注意: LSeg 目前仅支持 CUDA，MPS 和 CPU 不支持。
    
    复用: autolabel/autolabel/features/lseg.py
    
    Requirements: 9.1, 9.2, 9.3
    """
    
    def __init__(self, checkpoint: str):
        """初始化 LSeg 特征提取器
        
        Args:
            checkpoint: LSeg 模型检查点路径
        """
        self.checkpoint = checkpoint
        self._available = False
        self._error = None
        self.device = _get_device()
        
        if self.device is None:
            self._error = "PyTorch not available"
            return
        
        # LSeg 需要 CUDA
        if self.device.type != 'cuda':
            self._error = "LSeg requires CUDA. MPS and CPU are not supported."
            return
        
        try:
            import torch
            import clip
            from torchvision import transforms
            
            self.torch = torch
            self.clip = clip
            
            # 尝试导入 LSeg 模块
            try:
                from modules.lseg_module import LSegModule
                from additional_utils.models import LSeg_MultiEvalModule
                
                module = LSegModule.load_from_checkpoint(
                    checkpoint_path=checkpoint,
                    backbone='clip_vitl16_384',
                    data_path=None,
                    num_features=256,
                    batch_size=1,
                    base_lr=1e-3,
                    max_epochs=100,
                    augment=False,
                    aux=True,
                    aux_weight=0,
                    ignore_index=255,
                    dataset='ade20k',
                    se_loss=False,
                    se_weight=0,
                    arch_option=0,
                    block_depth=0,
                    activation='lrelu'
                )
                
                # 跳过 ToTensor 操作
                self.transform = transforms.Compose(
                    module.val_transform.transforms[1:]
                )
                
                net = module.net.to(self.device)
                scales = [1.0]
                self.evaluator = LSeg_MultiEvalModule(
                    module, scales=scales, flip=False
                ).half().to(self.device).eval()
                
                self.text_encoder = module.net.clip_pretrained.to(
                    torch.float32
                ).to(self.device)
                
                self._available = True
                
            except ImportError as e:
                self._error = f"LSeg modules not found: {e}"
                
        except Exception as e:
            self._error = str(e)
    
    @property
    def is_available(self) -> bool:
        """检查 LSeg 是否可用"""
        return self._available
    
    def shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """计算输出特征图尺寸
        
        LSeg 输出为输入的 1/2
        """
        return (input_shape[0] // 2, input_shape[1] // 2)
    
    def encode_text(self, text: List[str]):
        """编码文本查询
        
        Args:
            text: 文本字符串列表
            
        Returns:
            归一化的文本特征张量 [N, 512]
        """
        if not self._available:
            raise RuntimeError(f"LSeg not available: {self._error}")
        
        with self.torch.inference_mode():
            tokenized = self.clip.tokenize(text).to(self.device)
            features = []
            for item in tokenized:
                f = self.text_encoder.encode_text(item[None])[0]
                features.append(f)
            features = self.torch.stack(features, dim=0)
            return features / self.torch.norm(features, dim=-1, keepdim=True)
    
    def __call__(self, x):
        """提取 LSeg 特征
        
        Args:
            x: 图像张量 [B, C, H, W]
            
        Returns:
            特征张量 [B, H', W', 512]
        """
        if not self._available:
            raise RuntimeError(f"LSeg not available: {self._error}")
        
        from torch.nn import functional as F
        
        x = self.transform(x)
        _, _, H, W = x.shape
        H_out, W_out = H // 2, W // 2
        
        out = []
        x = [F.interpolate(image[None], [H_out, W_out]) for image in x]
        for image in x:
            out.append(self.evaluator.compute_features(image.half()))
        
        out = self.torch.cat(out, dim=0)
        return out.permute(0, 2, 3, 1)


def get_feature_extractor(
    feature_type: str,
    checkpoint: Optional[str] = None
) -> BaseFeatureExtractor:
    """获取特征提取器
    
    Args:
        feature_type: 特征类型 ('dino' 或 'lseg')
        checkpoint: LSeg 模型检查点路径 (仅 LSeg 需要)
        
    Returns:
        特征提取器实例
        
    Raises:
        ValueError: 不支持的特征类型
    """
    if feature_type == 'dino':
        return DinoFeatureExtractor()
    elif feature_type == 'lseg':
        if checkpoint is None:
            raise ValueError("LSeg requires a checkpoint path")
        return LSegFeatureExtractor(checkpoint)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")


class FeatureCompressor:
    """特征压缩器 - 使用自编码器压缩特征维度
    
    跨平台支持: CUDA、MPS、CPU
    
    复用: autolabel/scripts/compute_feature_maps.py::compress_features
    
    Requirements: 8.2
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """初始化特征压缩器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._model = None
        self._available = False
        self.device = _get_device()
        
        if self.device is None:
            self._error = "PyTorch not available"
            return
        
        try:
            import torch
            from torch.nn import functional as F
            
            self.torch = torch
            self.F = F
            
            # 尝试使用 tinycudann 的自编码器 (仅 CUDA)
            if self.device.type == 'cuda':
                try:
                    import tinycudann as tcnn
                    self._model = self._create_tcnn_autoencoder(input_dim, output_dim)
                    self._available = True
                    self._backend = 'tcnn'
                except ImportError:
                    # 回退到纯 PyTorch 实现
                    self._model = self._create_torch_autoencoder(input_dim, output_dim)
                    self._available = True
                    self._backend = 'torch'
            else:
                # MPS 和 CPU 使用纯 PyTorch
                self._model = self._create_torch_autoencoder(input_dim, output_dim)
                self._available = True
                self._backend = 'torch'
                
        except ImportError as e:
            self._error = str(e)
    
    def _create_tcnn_autoencoder(self, input_dim: int, output_dim: int):
        """创建 tinycudann 自编码器"""
        import torch.nn as nn
        import tinycudann as tcnn
        
        class TcnnAutoencoder(nn.Module):
            def __init__(self, in_features, bottleneck):
                super().__init__()
                self.encoder = tcnn.Network(
                    n_input_dims=in_features,
                    n_output_dims=bottleneck,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "ReLU",
                        "n_neurons": 128,
                        "n_hidden_layers": 1
                    }
                )
                self.decoder = tcnn.Network(
                    n_input_dims=bottleneck,
                    n_output_dims=in_features,
                    network_config={
                        "otype": "CutlassMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 128,
                        "n_hidden_layers": 1
                    }
                )
            
            def forward(self, x, p=0.1):
                code = self.encoder(x)
                out = self.decoder(self.F.dropout(code, 0.1))
                return out, code
        
        return TcnnAutoencoder(input_dim, output_dim).to(self.device)
    
    def _create_torch_autoencoder(self, input_dim: int, output_dim: int):
        """创建纯 PyTorch 自编码器"""
        import torch.nn as nn
        
        class TorchAutoencoder(nn.Module):
            def __init__(self, in_features, bottleneck):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.ReLU(),
                    nn.Linear(128, bottleneck),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck, 128),
                    nn.ReLU(),
                    nn.Linear(128, in_features)
                )
            
            def forward(self, x, p=0.1):
                code = self.encoder(x)
                out = self.decoder(nn.functional.dropout(code, p))
                return out, code
        
        return TorchAutoencoder(input_dim, output_dim).to(self.device)
    
    @property
    def is_available(self) -> bool:
        """检查压缩器是否可用"""
        return self._available
    
    def compress(
        self,
        features: np.ndarray,
        epochs: int = 5,
        batch_size: int = 2048
    ) -> np.ndarray:
        """压缩特征
        
        Args:
            features: 输入特征 [N, H, W, C]
            epochs: 训练轮数
            batch_size: 批处理大小
            
        Returns:
            压缩后的特征 [N, H, W, output_dim]
        """
        if not self._available:
            raise RuntimeError("Feature compressor not available")
        
        from tqdm import tqdm
        
        N, H, W, C = features.shape
        
        # 训练自编码器
        optimizer = self.torch.optim.Adam(self._model.parameters(), lr=1e-3)
        dataset = self.torch.utils.data.TensorDataset(
            self.torch.tensor(features.reshape(N * H * W, C))
        )
        loader = self.torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for _ in range(epochs):
            bar = tqdm(loader, desc="Training autoencoder")
            for batch in bar:
                batch = batch[0].to(self.device)
                reconstructed, code = self._model(batch)
                loss = self.F.mse_loss(reconstructed, batch) + \
                       0.01 * self.torch.abs(code).mean()
                bar.set_description(f"Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        # 压缩所有特征
        with self.torch.inference_mode():
            features_out = np.zeros((N, H, W, self.output_dim), dtype=np.float16)
            for i, feature in enumerate(features):
                feature = self.torch.tensor(feature).view(H * W, C).to(self.device)
                _, out = self._model(feature)
                features_out[i] = out.detach().cpu().numpy().reshape(
                    H, W, self.output_dim
                )
        
        return features_out


class FeatureExtractor:
    """统一特征提取接口
    
    封装 DINO 和 LSeg 特征提取，提供统一的接口。
    
    复用: autolabel/scripts/compute_feature_maps.py
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3
    """
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        """初始化特征提取器
        
        Args:
            config: 特征提取配置
        """
        self.config = config or FeatureExtractionConfig()
        self._extractor = None
        self._compressor = None
    
    def _compute_size(
        self,
        image_path: str
    ) -> Tuple[int, int]:
        """计算目标图像尺寸
        
        Args:
            image_path: 图像路径
            
        Returns:
            目标尺寸 (H, W)
        """
        try:
            from torchvision.io.image import read_image
            image = read_image(image_path)
            _, H, W = image.shape
        except:
            from PIL import Image
            img = Image.open(image_path)
            W, H = img.size
        
        return self._compute_size_from_dims(H, W)
    
    def _compute_size_from_dims(
        self,
        H: int,
        W: int
    ) -> Tuple[int, int]:
        """从尺寸计算目标图像尺寸
        
        Args:
            H: 图像高度
            W: 图像宽度
            
        Returns:
            目标尺寸 (H, W)
        """
        short_side = min(H, W)
        
        if self.config.feature_type in ['dino']:
            target_size = self.config.target_size_dino
        elif self.config.feature_type == 'lseg':
            target_size = self.config.target_size_lseg
        else:
            target_size = 720
        
        scale_factor = target_size / short_side
        return int(H * scale_factor), int(W * scale_factor)
    
    def extract(
        self,
        scene_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False,
        video_path: Optional[str] = None
    ) -> FeatureExtractionResult:
        """提取场景特征
        
        Args:
            scene_path: 场景目录路径
            output_path: 输出 HDF5 文件路径 (默认为 scene_path/features.hdf)
            visualize: 是否可视化特征
            video_path: 特征可视化视频输出路径
            
        Returns:
            特征提取结果
        """
        import torch
        from torch.nn import functional as F
        from tqdm import tqdm
        
        # 设置随机种子
        np.random.seed(0)
        torch.manual_seed(0)
        
        # 确定输出路径
        if output_path is None:
            output_path = os.path.join(scene_path, 'features.hdf')
        
        # 获取 RGB 图像路径或视频
        rgb_dir = os.path.join(scene_path, 'rgb')
        rgb_video = os.path.join(scene_path, 'rgb.mp4')
        use_video = False
        rgb_paths = []
        
        if os.path.exists(rgb_dir):
            # 使用 rgb/ 目录中的图片
            rgb_files = sorted(
                [f for f in os.listdir(rgb_dir) if not f.startswith('.')],
                key=lambda x: int(x.split('.')[0])
            )
            rgb_paths = [os.path.join(rgb_dir, f) for f in rgb_files]
        elif os.path.exists(rgb_video):
            # 使用 rgb.mp4 视频文件
            use_video = True
            # 获取视频帧数
            try:
                import cv2
                cap = cv2.VideoCapture(rgb_video)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                rgb_paths = list(range(frame_count))  # 用帧索引代替路径
            except ImportError:
                return FeatureExtractionResult(
                    success=False,
                    output_path=output_path,
                    feature_type=self.config.feature_type,
                    num_frames=0,
                    feature_shape=(0,),
                    error_message="OpenCV required for video processing. Install with: pip install opencv-python"
                )
        else:
            return FeatureExtractionResult(
                success=False,
                output_path=output_path,
                feature_type=self.config.feature_type,
                num_frames=0,
                feature_shape=(0,),
                error_message=f"RGB data not found. Expected: {rgb_dir} or {rgb_video}"
            )
        
        if len(rgb_paths) == 0:
            return FeatureExtractionResult(
                success=False,
                output_path=output_path,
                feature_type=self.config.feature_type,
                num_frames=0,
                feature_shape=(0,),
                error_message="No RGB frames found"
            )
        
        try:
            # 初始化特征提取器
            self._extractor = get_feature_extractor(
                self.config.feature_type,
                self.config.checkpoint
            )
            
            if not self._extractor.is_available:
                return FeatureExtractionResult(
                    success=False,
                    output_path=output_path,
                    feature_type=self.config.feature_type,
                    num_frames=len(rgb_paths),
                    feature_shape=(0,),
                    error_message=f"Feature extractor not available"
                )
            
            # 计算目标尺寸
            if use_video:
                import cv2
                cap = cv2.VideoCapture(rgb_video)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return FeatureExtractionResult(
                        success=False,
                        output_path=output_path,
                        feature_type=self.config.feature_type,
                        num_frames=len(rgb_paths),
                        feature_shape=(0,),
                        error_message="Failed to read video frame"
                    )
                frame_h, frame_w = frame.shape[:2]
                H, W = self._compute_size_from_dims(frame_h, frame_w)
            else:
                H, W = self._compute_size(rgb_paths[0])
            
            feature_shape = self._extractor.shape((H, W))
            
            # 创建输出文件
            output_file = h5py.File(output_path, 'w', libver='latest')
            group = output_file.create_group('features')
            
            dataset = group.create_dataset(
                self.config.feature_type,
                (len(rgb_paths), *feature_shape, self.config.output_dim),
                dtype=np.float16,
                compression='lzf'
            )
            
            # 提取特征
            extracted = []
            
            # 获取设备
            device = _get_device()
            
            with torch.inference_mode():
                batch_size = self.config.batch_size
                
                if use_video:
                    # 从视频读取帧
                    import cv2
                    cap = cv2.VideoCapture(rgb_video)
                    frame_idx = 0
                    batch_frames = []
                    
                    pbar = tqdm(total=len(rgb_paths), desc=f"Extracting {self.config.feature_type} features")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # BGR -> RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # HWC -> CHW
                        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
                        batch_frames.append(frame_tensor)
                        
                        if len(batch_frames) >= batch_size:
                            # 处理批次
                            image = torch.stack(batch_frames).to(device)
                            image = F.interpolate(image.float(), [H, W])
                            features = self._extractor(image / 255.).numpy()
                            
                            start_idx = frame_idx - len(batch_frames) + 1
                            if self.config.autoencode:
                                extracted += [f for f in features]
                            else:
                                dataset[start_idx:frame_idx+1] = features[..., :self.config.output_dim]
                            
                            batch_frames = []
                            pbar.update(batch_size)
                        
                        frame_idx += 1
                    
                    # 处理剩余帧
                    if batch_frames:
                        image = torch.stack(batch_frames).to(device)
                        image = F.interpolate(image.float(), [H, W])
                        features = self._extractor(image / 255.).numpy()
                        
                        start_idx = frame_idx - len(batch_frames)
                        if self.config.autoencode:
                            extracted += [f for f in features]
                        else:
                            dataset[start_idx:frame_idx] = features[..., :self.config.output_dim]
                        pbar.update(len(batch_frames))
                    
                    cap.release()
                    pbar.close()
                else:
                    # 从图片文件读取
                    from torchvision.io.image import read_image
                    
                    for i in tqdm(
                        range(math.ceil(len(rgb_paths) / batch_size)),
                        desc=f"Extracting {self.config.feature_type} features"
                    ):
                        index = slice(i * batch_size, (i + 1) * batch_size)
                        batch = rgb_paths[index]
                        image = torch.stack([read_image(p) for p in batch]).to(device)
                        image = F.interpolate(image.float(), [H, W])
                        features = self._extractor(image / 255.).numpy()
                        
                        if self.config.autoencode:
                            extracted += [f for f in features]
                        else:
                            dataset[index] = features[..., :self.config.output_dim]
            
            # 压缩特征
            if self.config.autoencode and len(extracted) > 0:
                features_array = np.stack(extracted)
                _, _, _, C = features_array.shape
                
                self._compressor = FeatureCompressor(C, self.config.output_dim)
                if self._compressor.is_available:
                    compressed = self._compressor.compress(features_array)
                    dataset[:] = compressed
                else:
                    # 如果压缩器不可用，直接截断维度
                    dataset[:] = features_array[..., :self.config.output_dim]
            
            # 计算 PCA 用于可视化
            pca_available = self._compute_pca(dataset)
            
            output_file.close()
            
            # 可视化
            if visualize or video_path:
                self._visualize_features(
                    output_path,
                    self.config.feature_type,
                    visualize,
                    video_path
                )
            
            return FeatureExtractionResult(
                success=True,
                output_path=output_path,
                feature_type=self.config.feature_type,
                num_frames=len(rgb_paths),
                feature_shape=feature_shape,
                pca_available=pca_available
            )
            
        except Exception as e:
            return FeatureExtractionResult(
                success=False,
                output_path=output_path,
                feature_type=self.config.feature_type,
                num_frames=len(rgb_paths),
                feature_shape=(0,),
                error_message=str(e)
            )
    
    def _compute_pca(self, dataset) -> bool:
        """计算 PCA 用于特征可视化
        
        Args:
            dataset: HDF5 数据集
            
        Returns:
            是否成功计算 PCA
        """
        try:
            from sklearn import decomposition
            
            N, H, W, C = dataset[:].shape
            X = dataset[:].reshape(N * H * W, C)
            
            pca = decomposition.PCA(n_components=3)
            indices = np.random.randint(0, X.shape[0], size=min(50000, X.shape[0]))
            subset = X[indices]
            transformed = pca.fit_transform(subset)
            
            minimum = transformed.min(axis=0)
            maximum = transformed.max(axis=0)
            diff = maximum - minimum
            
            dataset.attrs['pca'] = np.void(pickle.dumps(pca))
            dataset.attrs['min'] = minimum
            dataset.attrs['range'] = diff
            
            return True
        except Exception:
            return False
    
    def _visualize_features(
        self,
        hdf_path: str,
        feature_type: str,
        show: bool = False,
        video_path: Optional[str] = None
    ):
        """可视化特征
        
        Args:
            hdf_path: HDF5 文件路径
            feature_type: 特征类型
            show: 是否显示
            video_path: 视频输出路径
        """
        with h5py.File(hdf_path, 'r') as f:
            features = f['features'][feature_type]
            
            if 'pca' not in features.attrs:
                print("PCA not available for visualization")
                return
            
            pca = pickle.loads(features.attrs['pca'].tobytes())
            N, H, W, C = features[:].shape
            
            if show:
                from matplotlib import pyplot
                feature_maps = features[:]
                for fm in feature_maps[::10]:
                    mapped = pca.transform(fm.reshape(H * W, C)).reshape(H, W, 3)
                    normalized = np.clip(
                        (mapped - features.attrs['min']) / features.attrs['range'],
                        0, 1
                    )
                    pyplot.imshow(normalized)
                    pyplot.show()
            
            if video_path:
                self._write_video(features, pca, video_path)
    
    def _write_video(self, features, pca, output_path: str):
        """写入特征可视化视频
        
        Args:
            features: HDF5 特征数据集
            pca: PCA 模型
            output_path: 输出路径
        """
        try:
            from skvideo.io.ffmpeg import FFmpegWriter
            from tqdm import tqdm
            
            N, H, W, C = features[:].shape
            writer = FFmpegWriter(
                output_path,
                inputdict={'-framerate': '5'},
                outputdict={
                    '-c:v': 'libx264',
                    '-r': '5',
                    '-pix_fmt': 'yuv420p'
                }
            )
            
            for feature in tqdm(features, desc="Encoding frames"):
                mapped = pca.transform(feature.reshape(H * W, C)).reshape(H, W, 3)
                normalized = np.clip(
                    (mapped - features.attrs['min']) / features.attrs['range'],
                    0, 1
                )
                frame = (normalized * 255.0).astype(np.uint8)
                writer.writeFrame(frame)
            
            writer.close()
        except ImportError:
            print("skvideo not available for video export")


def extract_dino_features(
    scene_path: str,
    output_dim: int = 64,
    autoencode: bool = True,
    output_path: Optional[str] = None
) -> FeatureExtractionResult:
    """提取 DINO 特征的便捷函数
    
    Args:
        scene_path: 场景目录路径
        output_dim: 输出特征维度
        autoencode: 是否使用自编码器压缩
        output_path: 输出文件路径
        
    Returns:
        特征提取结果
        
    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    config = FeatureExtractionConfig(
        feature_type='dino',
        output_dim=output_dim,
        autoencode=autoencode
    )
    extractor = FeatureExtractor(config)
    return extractor.extract(scene_path, output_path)


def extract_lseg_features(
    scene_path: str,
    checkpoint: str,
    output_dim: int = 512,
    output_path: Optional[str] = None
) -> FeatureExtractionResult:
    """提取 LSeg 特征的便捷函数
    
    Args:
        scene_path: 场景目录路径
        checkpoint: LSeg 模型检查点路径
        output_dim: 输出特征维度
        output_path: 输出文件路径
        
    Returns:
        特征提取结果
        
    Requirements: 9.1, 9.2, 9.3
    """
    config = FeatureExtractionConfig(
        feature_type='lseg',
        output_dim=output_dim,
        autoencode=False,  # LSeg 通常不需要压缩
        checkpoint=checkpoint
    )
    extractor = FeatureExtractor(config)
    return extractor.extract(scene_path, output_path)


def load_features(
    hdf_path: str,
    feature_type: str = 'dino'
) -> Tuple[np.ndarray, Optional[object]]:
    """加载已提取的特征
    
    Args:
        hdf_path: HDF5 文件路径
        feature_type: 特征类型
        
    Returns:
        (特征数组, PCA 模型)
    """
    with h5py.File(hdf_path, 'r') as f:
        features = f['features'][feature_type][:]
        
        pca = None
        if 'pca' in f['features'][feature_type].attrs:
            pca = pickle.loads(
                f['features'][feature_type].attrs['pca'].tobytes()
            )
        
        return features, pca


def visualize_feature_map(
    feature: np.ndarray,
    pca,
    pca_min: np.ndarray,
    pca_range: np.ndarray
) -> np.ndarray:
    """可视化单个特征图
    
    Args:
        feature: 特征图 [H, W, C]
        pca: PCA 模型
        pca_min: PCA 最小值
        pca_range: PCA 范围
        
    Returns:
        RGB 图像 [H, W, 3]
    """
    H, W, C = feature.shape
    mapped = pca.transform(feature.reshape(H * W, C)).reshape(H, W, 3)
    normalized = np.clip((mapped - pca_min) / pca_range, 0, 1)
    return (normalized * 255).astype(np.uint8)


def check_feature_extraction_available() -> dict:
    """检查特征提取功能是否可用
    
    Returns:
        各功能可用性字典
    """
    result = {
        'torch': False,
        'cuda': False,
        'mps': False,
        'device': 'cpu',
        'dino': False,
        'lseg': False,
        'autoencoder_tcnn': False,
        'autoencoder_torch': False,
    }
    
    try:
        import torch
        result['torch'] = True
        result['cuda'] = torch.cuda.is_available()
        result['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # 确定设备
        if result['cuda']:
            result['device'] = 'cuda'
        elif result['mps']:
            result['device'] = 'mps'
        else:
            result['device'] = 'cpu'
    except ImportError:
        return result
    
    # 检查 DINO (支持 CUDA、MPS、CPU)
    result['dino'] = result['torch']  # DINO 在所有设备上都可用
    
    # 检查 tinycudann (仅 CUDA)
    if result['cuda']:
        try:
            import tinycudann
            result['autoencoder_tcnn'] = True
        except ImportError:
            pass
    
    # PyTorch 自编码器总是可用（如果有 torch）
    result['autoencoder_torch'] = result['torch']
    
    # LSeg 需要 CUDA
    if result['cuda']:
        try:
            import clip
            result['lseg'] = True
        except ImportError:
            pass
    
    return result


# 导出的公共接口
__all__ = [
    'FeatureExtractionConfig',
    'FeatureExtractionResult',
    'BaseFeatureExtractor',
    'DinoFeatureExtractor',
    'LSegFeatureExtractor',
    'FeatureExtractor',
    'FeatureCompressor',
    'get_feature_extractor',
    'extract_dino_features',
    'extract_lseg_features',
    'load_features',
    'visualize_feature_map',
    'check_feature_extraction_available',
]
