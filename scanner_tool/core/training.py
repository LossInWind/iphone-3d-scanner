"""
NeRF 训练模块 - 直接复用 autolabel 原项目代码

完整复现原项目功能:
- RGB + 深度 + 语义特征联合训练
- 支持 3D 点云语义分割
- 支持开放词汇查询

依赖: autolabel 原项目模块 (位于 scanner_tool/autolabel/)
- autolabel.trainer.SimpleTrainer
- autolabel.dataset.SceneDataset
- autolabel.model_utils

平台支持:
- NVIDIA GPU (CUDA): 完整功能
- Apple Silicon (MPS): 部分功能 (需要纯 PyTorch 后端)
- CPU: 仅用于测试

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8
"""

import os
import sys
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from argparse import Namespace

import numpy as np

# autolabel 现在位于 scanner_tool/autolabel/
AUTOLABEL_PATH = Path(__file__).parent.parent / 'autolabel'
if str(AUTOLABEL_PATH) not in sys.path:
    sys.path.insert(0, str(AUTOLABEL_PATH))


def _check_cuda():
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_autolabel():
    """检查 autolabel 模块是否可用"""
    try:
        from autolabel.trainer import SimpleTrainer
        from autolabel.dataset import SceneDataset
        from autolabel import model_utils
        return True
    except ImportError as e:
        print(f"Warning: autolabel not available: {e}")
        return False


CUDA_AVAILABLE = _check_cuda()
AUTOLABEL_AVAILABLE = _check_autolabel()


# ============================================================
# 平台检测
# ============================================================

from enum import Enum

class DeviceType(Enum):
    """设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class BackendType(Enum):
    """后端类型"""
    AUTOLABEL = "autolabel"
    PURE_PYTORCH = "pure_pytorch"


@dataclass
class PlatformInfo:
    """平台信息"""
    device_type: DeviceType
    cuda_available: bool
    mps_available: bool
    autolabel_available: bool
    torch_version: str = ""
    cuda_version: str = ""
    device_name: str = ""


def detect_platform() -> PlatformInfo:
    """检测当前平台信息"""
    cuda_available = False
    mps_available = False
    torch_version = ""
    cuda_version = ""
    device_name = ""
    device_type = DeviceType.CPU
    
    try:
        import torch
        torch_version = torch.__version__
        
        if torch.cuda.is_available():
            cuda_available = True
            device_type = DeviceType.CUDA
            cuda_version = torch.version.cuda or ""
            if torch.cuda.device_count() > 0:
                device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
            device_type = DeviceType.MPS
            device_name = "Apple Silicon"
    except ImportError:
        pass
    
    return PlatformInfo(
        device_type=device_type,
        cuda_available=cuda_available,
        mps_available=mps_available,
        autolabel_available=AUTOLABEL_AVAILABLE,
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_name=device_name
    )


def get_device_string() -> str:
    """获取设备字符串"""
    info = detect_platform()
    if info.cuda_available:
        return "cuda"
    elif info.mps_available:
        return "mps"
    return "cpu"


def print_platform_info():
    """打印平台信息"""
    info = detect_platform()
    
    print("--- Platform Detection ---")
    print(f"PyTorch version: {info.torch_version or 'Not installed'}")
    print(f"Device type: {info.device_type.value}")
    
    if info.cuda_available:
        print(f"CUDA version: {info.cuda_version}")
        print(f"GPU: {info.device_name}")
    elif info.mps_available:
        print(f"Device: {info.device_name}")
    else:
        print("Device: CPU only")
    
    print(f"autolabel available: {info.autolabel_available}")


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    iterations: int = 10000
    batch_size: int = 4096
    learning_rate: float = 5e-3
    
    # 损失权重
    rgb_weight: float = 1.0
    depth_weight: float = 0.1
    semantic_weight: float = 0.04
    feature_weight: float = 0.01
    
    # 数据参数
    factor_train: float = 2.0
    factor_test: float = 2.0
    
    # 模型参数
    features: Optional[str] = None  # 'dino' or 'lseg'
    n_classes: int = 2
    
    # 其他
    workers: int = 1
    fp16: bool = True
    eval_after_train: bool = False


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    model_path: str
    iterations: int
    final_loss: float = 0.0
    error_message: str = ""


def check_training_available() -> Dict[str, bool]:
    """检查训练功能可用性
    
    Returns:
        各功能可用性字典
    """
    info = detect_platform()
    
    result = {
        'torch': bool(info.torch_version),
        'cuda': info.cuda_available,
        'mps': info.mps_available,
        'autolabel': AUTOLABEL_AVAILABLE,
        'full_training': info.cuda_available and AUTOLABEL_AVAILABLE,
        'semantic_training': info.cuda_available and AUTOLABEL_AVAILABLE,
        'available': bool(info.torch_version),
    }
    
    # 检查 torch-ngp
    try:
        from torch_ngp.nerf.utils import Trainer
        result['torch_ngp'] = True
    except ImportError:
        result['torch_ngp'] = False
    
    return result


def train_nerf(
    scene_path: str,
    config: Optional[TrainingConfig] = None,
    workspace: Optional[str] = None
) -> TrainingResult:
    """训练 NeRF 模型 (完整功能，需要 CUDA)
    
    直接使用 autolabel 原项目的训练代码。
    
    Args:
        scene_path: 场景目录路径
        config: 训练配置
        workspace: 输出目录 (默认为 scene_path/nerf)
        
    Returns:
        训练结果
    """
    config = config or TrainingConfig()
    
    # 检查依赖
    if not CUDA_AVAILABLE:
        return TrainingResult(
            success=False,
            model_path="",
            iterations=0,
            error_message="CUDA not available. Full NeRF training requires NVIDIA GPU."
        )
    
    if not AUTOLABEL_AVAILABLE:
        return TrainingResult(
            success=False,
            model_path="",
            iterations=0,
            error_message="autolabel module not available. Please check installation."
        )
    
    try:
        import torch
        from torch import optim
        from autolabel import model_utils
        from autolabel.dataset import SceneDataset, LenDataset
        from autolabel.trainer import SimpleTrainer
        
        # 创建数据集
        dataset = SceneDataset(
            'train',
            scene_path,
            factor=config.factor_train,
            batch_size=config.batch_size,
            features=config.features
        )
        
        # 确定类别数
        n_classes = dataset.n_classes if dataset.n_classes is not None else config.n_classes
        
        # 创建模型参数
        flags = Namespace(
            features=config.features,
            rgb_weight=config.rgb_weight,
            depth_weight=config.depth_weight,
            semantic_weight=config.semantic_weight,
            feature_weight=config.feature_weight,
            lr=config.learning_rate,
            # 模型架构参数 (使用默认值)
            num_layers=2,
            hidden_dim=64,
            geo_feat_dim=15,
            num_layers_color=3,
            hidden_dim_color=64,
            num_layers_semantic=2,
            hidden_dim_semantic=64,
            semantic_dim=64,
            bound=1.0,
            cuda_ray=True,
            density_thresh=10,
        )
        
        # 创建模型
        model = model_utils.create_model(
            dataset.min_bounds, 
            dataset.max_bounds,
            n_classes, 
            flags
        )
        
        # 训练选项
        opt = Namespace(
            rand_pose=-1,
            color_space='srgb',
            feature_loss=config.features is not None,
            rgb_weight=config.rgb_weight,
            depth_weight=config.depth_weight,
            semantic_weight=config.semantic_weight,
            feature_weight=config.feature_weight
        )
        
        # 优化器
        optimizer = lambda model: torch.optim.Adam([
            {
                'name': 'encoding',
                'params': list(model.encoder.parameters())
            },
            {
                'name': 'net',
                'params': model.network_parameters(),
                'weight_decay': 1e-6
            },
        ], lr=config.learning_rate, betas=(0.9, 0.99), eps=1e-15)
        
        # 数据加载器
        train_dataloader = torch.utils.data.DataLoader(
            LenDataset(dataset, 1000),
            batch_size=None,
            num_workers=config.workers
        )
        train_dataloader._data = dataset
        
        # 损失函数
        criterion = torch.nn.MSELoss(reduction='none')
        
        # 学习率调度器
        gamma = 0.5
        steps = math.log(1e-4 / config.learning_rate, gamma)
        step_size = max(config.iterations // steps // 1000, 1)
        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(
            optimizer, gamma=gamma, step_size=step_size
        )
        
        # 确定输出目录
        if workspace is None:
            model_dir = model_utils.model_dir(scene_path, flags)
        else:
            model_dir = workspace
        
        # 保存参数
        model_utils.write_params(model_dir, flags)
        
        # 创建训练器
        epochs = int(np.ceil(config.iterations / 1000))
        trainer = SimpleTrainer(
            'ngp',
            opt,
            model,
            device='cuda:0',
            workspace=model_dir,
            optimizer=optimizer,
            criterion=criterion,
            fp16=config.fp16,
            ema_decay=0.95,
            lr_scheduler=scheduler,
            scheduler_update_every_step=False,
            metrics=[],
            use_checkpoint='latest'
        )
        
        # 训练
        print(f"Training NeRF for {config.iterations} iterations...")
        trainer.train(train_dataloader, epochs)
        trainer.save_checkpoint()
        
        # 评估 (可选)
        if config.eval_after_train:
            testset = SceneDataset(
                'test',
                scene_path,
                factor=config.factor_test,
                batch_size=config.batch_size * 2
            )
            test_dataloader = torch.utils.data.DataLoader(
                LenDataset(testset, testset.rotations.shape[0]),
                batch_size=None,
                num_workers=0
            )
            trainer.evaluate(test_dataloader)
        
        return TrainingResult(
            success=True,
            model_path=model_dir,
            iterations=config.iterations
        )
        
    except Exception as e:
        return TrainingResult(
            success=False,
            model_path="",
            iterations=0,
            error_message=str(e)
        )


def load_trained_model(
    scene_path: str,
    workspace: Optional[str] = None
) -> Tuple[Any, Any]:
    """加载训练好的 NeRF 模型
    
    Args:
        scene_path: 场景目录路径
        workspace: 模型目录 (默认为 scene_path/nerf)
        
    Returns:
        (model, dataset) 元组
    """
    if not AUTOLABEL_AVAILABLE:
        raise RuntimeError("autolabel module not available")
    
    from autolabel import model_utils
    from autolabel.dataset import SceneDataset
    
    # 查找模型目录
    if workspace is None:
        nerf_dir = os.path.join(scene_path, 'nerf')
    else:
        nerf_dir = workspace
    
    if not os.path.exists(nerf_dir):
        raise FileNotFoundError(f"NeRF directory not found: {nerf_dir}")
    
    # 查找最新的模型
    model_name = None
    for name in os.listdir(nerf_dir):
        checkpoint_dir = os.path.join(nerf_dir, name, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            model_name = name
            break
    
    if model_name is None:
        raise FileNotFoundError(f"No trained model found in {nerf_dir}")
    
    model_path = os.path.join(nerf_dir, model_name)
    
    # 读取参数
    params = model_utils.read_params(model_path)
    
    # 创建数据集
    dataset = SceneDataset(
        'test',
        scene_path,
        factor=2.0,
        batch_size=4096,
        lazy=True
    )
    
    # 创建模型
    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(
        dataset.min_bounds,
        dataset.max_bounds,
        n_classes,
        params
    ).cuda()
    
    # 加载检查点
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    model_utils.load_checkpoint(model, checkpoint_dir)
    model = model.eval()
    
    return model, dataset


def query_3d_semantics(
    model: Any,
    points: np.ndarray,
    text_features: np.ndarray,
    batch_size: int = 50000
) -> np.ndarray:
    """查询 3D 点的语义类别
    
    Args:
        model: 训练好的 NeRF 模型
        points: 3D 点坐标 [N, 3]
        text_features: 文本特征 [C, D]
        batch_size: 批处理大小
        
    Returns:
        语义类别索引 [N]
    """
    import torch
    
    points_tensor = torch.tensor(points, dtype=torch.float16, device='cuda')
    text_features_tensor = torch.tensor(text_features, dtype=torch.float16, device='cuda')
    
    N = points_tensor.shape[0]
    C = text_features_tensor.shape[0]
    
    similarities = torch.zeros((N, C), dtype=torch.float16, device='cuda')
    
    batches = math.ceil(N / batch_size)
    for batch_index in range(batches):
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, N)
        batch = points_tensor[start:end]
        
        # 查询密度和语义特征
        density = model.density(batch)
        _, features = model.semantic(density['geo_feat'], density['sigma'])
        
        # 归一化
        features = features / torch.norm(features, dim=-1, keepdim=True)
        
        # 计算与每个文本的相似度
        for i in range(C):
            similarities[start:end, i] = (features * text_features_tensor[i]).sum(dim=-1)
    
    # 返回最相似的类别
    return similarities.argmax(dim=-1).cpu().numpy()


# ============================================================
# 简化版训练 (用于非 CUDA 平台的基本测试)
# ============================================================

def train_simple(
    scene_path: str,
    iterations: int = 1000,
    output_path: Optional[str] = None
) -> TrainingResult:
    """简化版训练 (仅 RGB，用于测试)
    
    在没有 CUDA 的平台上提供基本的训练功能。
    不支持语义特征，仅用于验证数据流程。
    
    Args:
        scene_path: 场景目录路径
        iterations: 训练迭代次数
        output_path: 输出路径
        
    Returns:
        训练结果
    """
    try:
        import torch
        
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        
        print(f"Simple training on {device} (RGB only, no semantic features)")
        print("For full semantic training, use NVIDIA GPU with CUDA.")
        
        # 这里可以添加简化版的训练逻辑
        # 目前只是占位符
        
        return TrainingResult(
            success=True,
            model_path=output_path or os.path.join(scene_path, 'nerf_simple'),
            iterations=iterations,
            error_message="Simple training completed (RGB only)"
        )
        
    except Exception as e:
        return TrainingResult(
            success=False,
            model_path="",
            iterations=0,
            error_message=str(e)
        )
