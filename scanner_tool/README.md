# Scanner Tool 使用指南

PC 端 3D 场景处理工具，配合 iOS Scanner App 使用。

## ⚠️ 重要提示

**必须使用 conda 环境运行本工具！**

```bash
# 激活环境后再运行命令
conda activate scanner_tool
python -m scanner_tool.cli.main <command>
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境 (推荐 Python 3.10)
conda create -n scanner_tool python=3.10
conda activate scanner_tool

# 安装基础依赖
pip install -r requirements.txt

# 或手动安装
pip install numpy scipy pillow open3d opencv-python torch torchvision h5py PyQt6 tqdm scikit-image

# macOS 额外安装 ffmpeg (用于视频处理)
conda install -c conda-forge ffmpeg

# (可选) 安装 CLIP 用于文本查询
pip install git+https://github.com/openai/CLIP.git
```

### 2. 验证安装

```bash
# 激活环境
conda activate scanner_tool

# 查看平台信息和功能可用性
python -m scanner_tool.cli.main platform

# 查看所有命令
python -m scanner_tool.cli.main --help
```

### 3. 从 iOS 传输数据

**PC 端启动接收服务器：**
```bash
# 确保已激活 conda 环境
conda activate scanner_tool

python -m scanner_tool.cli.main serve --port 8080 --output ./datasets
```

**iOS 端发送数据：**
1. 打开 Scanner App → 选择数据集 → 点击 "WiFi 传输到电脑"
2. 输入 PC 显示的 IP:端口 (如 `192.168.1.100:8080`)
3. 点击发送

---

## 运行命令

**所有命令都需要先激活 conda 环境：**

```bash
# 方式 1: 先激活环境
conda activate scanner_tool
python -m scanner_tool.cli.main <command>

# 方式 2: 直接使用完整路径 (不推荐)
/opt/miniconda3/envs/scanner_tool/bin/python -m scanner_tool.cli.main <command>
```

---

## 功能一览

| 功能 | 命令 | 平台支持 |
|------|------|----------|
| WiFi 数据传输 | `serve` | 全平台 |
| 点云可视化 | `visualize` | 全平台 |
| 位姿估计 | `map` | 全平台 |
| 场景边界 | `bounds` | 全平台 |
| 特征提取 | `features` | DINO 全平台 / LSeg 仅 CUDA |
| NeRF 训练 | `train` | 全平台 (CUDA/MPS/CPU) |
| 语义标注 GUI | `label` | 全平台 |
| 文本查询分割 | `query` | CLIP 全平台 / LSeg 仅 CUDA |
| 3D 点云分割 | `pointcloud` | 仅 CUDA |
| NeRF 渲染视频 | `render` | 仅 CUDA |
| 导出结果 | `export` | 全平台 |
| 评估指标 | `evaluate` | 全平台 |
| 数据集导入 | `import` | 全平台 |
| 格式转换 | `convert` | 全平台 |
| 平台信息 | `platform` | 全平台 |
| 实时分割服务 | `realtime` | 全平台 |

---

## 平台支持详情

| 功能 | macOS (MPS) | Linux/Windows (CUDA) |
|------|-------------|----------------------|
| WiFi 数据传输 | ✅ | ✅ |
| 点云可视化 | ✅ | ✅ |
| DINO 特征提取 | ✅ | ✅ |
| LSeg 特征提取 | ❌ | ✅ |
| NeRF 训练 (RGB) | ⚠️ 简化版 | ✅ 完整版 |
| NeRF 训练 (语义) | ❌ | ✅ |
| 3D 点云语义分割 | ❌ | ✅ |
| NeRF 渲染视频 | ❌ | ✅ |
| CLIP 文本查询 | ✅ | ✅ |
| LSeg 文本查询 | ❌ | ✅ |
| 语义标注 GUI | ✅ | ✅ |
| 导出/评估 | ✅ | ✅ |

**完整的 3D 语义分割功能需要 NVIDIA GPU (CUDA)**

---

## 命令详解

### 🖥️ 平台信息 (platform)

查看当前平台和功能可用性。

```bash
python -m scanner_tool.cli.main platform
```

输出示例：
```
=== Scanner Tool Platform Info ===

--- Platform Detection ---
PyTorch version: 2.9.1
Device type: mps
Device: Apple Silicon
autolabel available: ✗ (需要 CUDA)

--- Feature Availability ---
PyTorch: ✓
CUDA: ✗
MPS (Apple Silicon): ✓
...
```

---

### 📡 WiFi 传输 (serve)

启动服务器接收 iOS Scanner App 的数据。

```bash
# 默认端口 8080，保存到 ./datasets
python -m scanner_tool.cli.main serve

# 自定义端口和输出目录
python -m scanner_tool.cli.main serve --port 9000 --output /path/to/datasets
```

---

### 📊 可视化 (visualize)

查看扫描的点云、相机轨迹。

```bash
# 基本可视化
python -m scanner_tool.cli.main visualize datasets/xxx

# 调整采样间隔 (每 30 帧取一帧，减少内存)
python -m scanner_tool.cli.main visualize datasets/xxx --every 30

# 调整置信度过滤 (0=全部, 1=中等, 2=高置信度)
python -m scanner_tool.cli.main visualize datasets/xxx --confidence 2

# 导出点云文件
python -m scanner_tool.cli.main visualize datasets/xxx --pointcloud-output output.ply

# RGB-D 积分重建并导出网格
python -m scanner_tool.cli.main visualize datasets/xxx --integrate --mesh-output mesh.ply
```

**交互操作：**
- 鼠标左键拖动：旋转视角
- 鼠标右键拖动：平移
- 滚轮：缩放
- `Q` 键：退出

---

### 🗺️ 位姿估计 (map)

使用 SfM (Structure from Motion) 重新估计相机位姿。

```bash
python -m scanner_tool.cli.main map datasets/xxx

# 启用调试模式
python -m scanner_tool.cli.main map datasets/xxx --debug
```

适用场景：当 ARKit 位姿不准确时，可以重新计算。

---

### 📐 场景边界 (bounds)

计算场景的 3D 边界框。

```bash
python -m scanner_tool.cli.main bounds datasets/xxx

# 自定义输出路径
python -m scanner_tool.cli.main bounds datasets/xxx --output bbox.txt
```

---

### 🧠 特征提取 (features)

提取视觉特征，为语义分割做准备。

```bash
# DINO 特征 (推荐，全平台可用)
python -m scanner_tool.cli.main features datasets/xxx --type dino

# LSeg 特征 (需要 NVIDIA GPU + 模型检查点)
python -m scanner_tool.cli.main features datasets/xxx --type lseg --checkpoint /path/to/lseg.ckpt

# 可视化特征图
python -m scanner_tool.cli.main features datasets/xxx --type dino --visualize

# 生成特征可视化视频
python -m scanner_tool.cli.main features datasets/xxx --type dino --video features.mp4
```

**特征类型说明：**
- **DINO**: 自监督视觉特征，用于相似性匹配和聚类
- **LSeg**: 语言驱动特征，支持开放词汇的语义分割

---

### 🎯 NeRF 训练 (train)

训练神经辐射场模型，实现 3D 场景重建。

```bash
# 基本训练
python -m scanner_tool.cli.main train datasets/xxx

# 指定迭代次数
python -m scanner_tool.cli.main train datasets/xxx --iters 10000

# 带语义特征训练 (需要先提取特征)
python -m scanner_tool.cli.main train datasets/xxx --features dino

# 完整参数
python -m scanner_tool.cli.main train datasets/xxx \
    --iters 15000 \
    --batch-size 4096 \
    --lr 5e-3 \
    --features lseg \
    --eval
```

**硬件加速：**
- NVIDIA GPU: 自动使用 CUDA (完整功能)
- Apple Silicon: 自动使用 MPS (简化版)
- 其他: 使用 CPU (较慢)

---

### 🎨 语义标注 (label)

打开图形界面，手动标注物体类别。

```bash
python -m scanner_tool.cli.main label datasets/xxx

# 自定义画笔大小
python -m scanner_tool.cli.main label datasets/xxx --brush-size 10
```

**标注工具：**
- 画笔工具：涂抹标注
- 多边形工具：精确边界
- 橡皮擦：修正错误
- 类别选择：切换标注类别

---

### 🔍 文本查询分割 (query)

用自然语言描述查找并分割物体。

```bash
# 使用 CLIP (全平台)
python -m scanner_tool.cli.main query datasets/xxx --prompts "chair" "table" "floor"

# 使用 LSeg (需要 CUDA)
python -m scanner_tool.cli.main query datasets/xxx \
    --type lseg \
    --prompts "chair" "table" \
    --checkpoint /path/to/lseg.ckpt
```

**前置条件：** 需要先运行 `features` 提取特征。

---

### 🌐 3D 点云语义分割 (pointcloud) [仅 CUDA]

从训练好的 NeRF 模型提取 3D 点云，并进行开放词汇的语义分割。

```bash
# 仅提取点云 (RGB 着色)
python -m scanner_tool.cli.main pointcloud datasets/xxx

# 语义分割点云
python -m scanner_tool.cli.main pointcloud datasets/xxx \
    --prompts "chair" "table" "floor" "wall" \
    --checkpoint /path/to/lseg.ckpt

# 可视化结果
python -m scanner_tool.cli.main pointcloud datasets/xxx \
    --prompts "chair" "table" \
    --checkpoint /path/to/lseg.ckpt \
    --visualize
```

**前置条件：**
1. 需要先运行 `train` 训练 NeRF 模型
2. 语义分割需要 LSeg 检查点文件

---

### 🎬 NeRF 渲染视频 (render) [仅 CUDA]

从训练好的 NeRF 模型渲染视频。

```bash
# 基本渲染
python -m scanner_tool.cli.main render datasets/xxx

# 指定输出路径和帧率
python -m scanner_tool.cli.main render datasets/xxx \
    --output output.mp4 \
    --fps 10

# 开放词汇语义渲染
python -m scanner_tool.cli.main render datasets/xxx \
    --classes "chair" "table" "floor" \
    --checkpoint /path/to/lseg.ckpt
```

**输出视频布局 (2x2 网格)：**
- 左上: RGB 渲染
- 右上: 深度图
- 左下: 语义分割
- 右下: 特征可视化

---

### 📤 导出结果 (export)

导出语义分割图或转换数据格式。

```bash
# 从手动标注导出语义分割图
python -m scanner_tool.cli.main export datasets/xxx --format semantic

# 从训练模型导出
python -m scanner_tool.cli.main export datasets/xxx --format semantic --from-model

# 导出为 instant-ngp 格式
python -m scanner_tool.cli.main export datasets/xxx --format instant-ngp --output ngp_data/
```

---

### 📈 评估指标 (evaluate)

评估语义分割的质量。

```bash
python -m scanner_tool.cli.main evaluate predictions/ groundtruth/

# 保存结果到 JSON
python -m scanner_tool.cli.main evaluate predictions/ groundtruth/ --output results.json
```

**输出指标：**
- mIoU: 平均交并比
- Pixel Accuracy: 像素准确率
- Per-class IoU: 各类别 IoU

---

### 📥 数据集导入 (import)

从其他格式导入数据集。

```bash
# 导入 Scanner App 数据
python -m scanner_tool.cli.main import scanner /path/to/input /path/to/output

# 导入 ARKitScenes 数据集
python -m scanner_tool.cli.main import arkitscenes /path/to/input /path/to/output

# 导入 ScanNet 数据集
python -m scanner_tool.cli.main import scannet /path/to/input /path/to/output

# 导入 Replica 数据集
python -m scanner_tool.cli.main import replica /path/to/input /path/to/output
```

---

### 🔄 格式转换 (convert)

转换数据格式。

```bash
# 转换为 Open3D 格式
python -m scanner_tool.cli.main convert input/ output/ --format open3d
```

---

### ⚡ 实时分割服务 (realtime)

启动后台服务，监控输入目录，有新图像时自动进行 2D 语义分割。

**适用场景：** 流式处理、与其他程序配合使用

```bash
python -m scanner_tool.cli.main realtime \
    --input /path/to/input \
    --output /path/to/output \
    --prompts "object" "background"
```

服务启动后会持续监控 `input/rgb/` 目录，发现新图像后自动分割并输出到 `output/` 目录。

---

## 语义分割方式对比

本工具提供 **4 种语义分割方式**，适用于不同场景：

| 方式 | 命令 | 数据来源 | 输出 | 平台 | 特点 |
|------|------|----------|------|------|------|
| 手动标注 | `label` | 数据集 RGB 帧 | 2D 分割图 | 全平台 | 精确但耗时 |
| 2D 文本查询 | `query` | 数据集 RGB 帧 | 2D 分割图 | 全平台 | 快速自动 |
| 3D 点云分割 | `pointcloud` | NeRF 模型 | 3D 点云文件 | 仅 CUDA | 真正的 3D 分割 |
| 实时分割 | `realtime` | 监控目录 | 2D 分割图 | 全平台 | 流式处理 |

**详细说明：**

1. **手动标注 (`label`)** 
   - 数据来源：从数据集的 `rgb.mp4` 视频中提取的帧
   - 打开 GUI 界面，用画笔工具逐帧标注物体边界
   - 适合需要高精度标注的场景

2. **2D 文本查询 (`query`)** 
   - 数据来源：数据集中的 RGB 帧（从 `rgb.mp4` 或 `rgb/` 目录）
   - 输入文本描述（如 "chair", "table"），自动在每帧图像上分割对应物体
   - 基于 CLIP/LSeg 特征匹配，输出每帧的 2D 分割掩码

3. **3D 点云分割 (`pointcloud`)** [仅 CUDA]
   - 数据来源：训练好的 NeRF 模型（需要先运行 `train` 命令）
   - 从 NeRF 的 3D 空间中提取点云，每个 3D 点都有语义标签
   - 输出 `.ply` 格式的带颜色点云文件，可用 MeshLab/CloudCompare 查看
   - 这是真正的 3D 语义分割，不是 2D 分割的投影

4. **实时分割 (`realtime`)** 
   - 数据来源：监控指定目录，处理新出现的图像文件
   - 启动后台服务，持续监控 `input/rgb/` 目录
   - 适合与其他程序配合使用，或流式处理场景

---

## 典型工作流程

### 流程 1: 快速查看扫描结果

最简单的使用方式，仅查看点云。

```bash
# 1. 传输数据
python -m scanner_tool.cli.main serve --port 8080

# 2. 可视化点云
python -m scanner_tool.cli.main visualize datasets/xxx
```

### 流程 2: 手动标注 (高精度)

适合需要精确标注的场景，如制作训练数据集。

```bash
# 1. 传输数据
python -m scanner_tool.cli.main serve --port 8080

# 2. 打开标注工具，手动标注每帧
python -m scanner_tool.cli.main label datasets/xxx

# 3. 导出分割结果
python -m scanner_tool.cli.main export datasets/xxx --format semantic
```

### 流程 3: 2D 文本查询分割 (快速)

用自然语言快速分割 2D 图像，无需手动标注。

```bash
# 1. 传输数据
python -m scanner_tool.cli.main serve --port 8080

# 2. 提取特征 (DINO 全平台可用)
python -m scanner_tool.cli.main features datasets/xxx --type dino

# 3. 文本查询分割
python -m scanner_tool.cli.main query datasets/xxx --prompts "chair" "table" "floor"
```

### 流程 4: 3D 点云语义分割 [仅 CUDA]

完整的 3D 语义分割流程，输出带语义标签的 3D 点云。这是原项目 autolabel 的核心功能。

**前置条件：** 需要 NVIDIA GPU (CUDA)

```bash
# 1. 传输数据
python -m scanner_tool.cli.main serve --port 8080

# 2. 计算场景边界
python -m scanner_tool.cli.main bounds datasets/xxx

# 3. 提取 LSeg 特征 (需要 CUDA)
python -m scanner_tool.cli.main features datasets/xxx \
    --type lseg \
    --checkpoint /path/to/lseg.ckpt

# 4. 训练带语义特征的 NeRF
python -m scanner_tool.cli.main train datasets/xxx \
    --features lseg \
    --iters 15000

# 5. 提取 3D 语义点云
python -m scanner_tool.cli.main pointcloud datasets/xxx \
    --prompts "chair" "table" "floor" "wall" \
    --checkpoint /path/to/lseg.ckpt \
    --visualize

# 6. (可选) 渲染视频
python -m scanner_tool.cli.main render datasets/xxx \
    --classes "chair" "table" "floor" \
    --checkpoint /path/to/lseg.ckpt
```

### 流程 5: NeRF 3D 重建 (无语义)

仅进行 3D 重建，不需要语义分割。

```bash
# 1. 传输数据
python -m scanner_tool.cli.main serve --port 8080

# 2. (可选) 重新估计位姿
python -m scanner_tool.cli.main map datasets/xxx

# 3. 计算边界
python -m scanner_tool.cli.main bounds datasets/xxx

# 4. 训练 NeRF (仅 RGB + 深度)
python -m scanner_tool.cli.main train datasets/xxx --iters 10000

# 5. 可视化结果
python -m scanner_tool.cli.main visualize datasets/xxx
```

---

## 数据集结构

传输后的数据集目录结构：

```
datasets/
└── 2025-12-28_22-11-40_B7999588/
    ├── rgb.mp4              # RGB 视频
    ├── depth/               # 深度图 (PNG)
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── confidence/          # 置信度图 (PNG)
    │   ├── 000000.png
    │   └── ...
    ├── odometry.csv         # 相机位姿
    ├── camera_matrix.csv    # 相机内参
    └── imu.csv              # IMU 数据
```

---

## 项目结构

```
scanner_tool/
├── cli/
│   └── main.py          # CLI 入口 (16 个命令)
├── core/
│   ├── visualization.py # 点云可视化
│   ├── features.py      # DINO/LSeg 特征提取
│   ├── training.py      # NeRF 训练 + 平台检测
│   ├── language.py      # 文本查询分割
│   ├── pointcloud.py    # 3D 点云语义分割
│   ├── render.py        # NeRF 渲染视频
│   ├── transfer.py      # WiFi 数据传输
│   ├── export.py        # 导出功能
│   ├── evaluation.py    # 评估指标
│   ├── realtime.py      # 实时分割服务
│   ├── pose_estimation.py # SfM 位姿估计
│   └── importers/       # 数据集导入器
├── gui/
│   └── labeling.py      # 语义标注 GUI
├── autolabel/           # 原项目代码 (子模块，需要 CUDA)
├── tests/
├── README.md
└── requirements.txt
```

---

## 常见问题

### Q: 命令报错 "ModuleNotFoundError: No module named 'scanner_tool'"
**原因：** 没有激活 conda 环境
```bash
# 解决方法：激活环境
conda activate scanner_tool
python -m scanner_tool.cli.main <command>
```

### Q: 可视化报错 "Cannot find installation of ffmpeg"
```bash
# 使用 conda 安装 ffmpeg
conda activate scanner_tool
conda install -c conda-forge ffmpeg
```

### Q: 特征提取报错 "CUDA not available"
DINO 特征支持 CPU/MPS，LSeg 需要 NVIDIA GPU。使用 DINO：
```bash
python -m scanner_tool.cli.main features datasets/xxx --type dino
```

### Q: WiFi 传输失败
1. 确保手机和电脑在同一局域网
2. 检查防火墙是否阻止了端口
3. 尝试使用手机热点

### Q: 内存不足
减少采样帧数：
```bash
python -m scanner_tool.cli.main visualize datasets/xxx --every 120
```

### Q: autolabel 不可用
autolabel 需要 NVIDIA GPU (CUDA) 才能工作。在 macOS 上，部分功能（如 3D 点云分割、NeRF 渲染）不可用。

### Q: PyQt6 报错
```bash
# 重新安装 PyQt6
conda activate scanner_tool
pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip
pip install PyQt6
```

---

## 获取帮助

```bash
# 激活环境
conda activate scanner_tool

# 查看所有命令
python -m scanner_tool.cli.main --help

# 查看具体命令帮助
python -m scanner_tool.cli.main visualize --help
python -m scanner_tool.cli.main features --help
python -m scanner_tool.cli.main train --help

# 查看平台信息和功能可用性
python -m scanner_tool.cli.main platform
```

---

## 环境信息

- **conda 环境名**: `scanner_tool`
- **Python 版本**: 3.10 (推荐)
- **依赖文件**: `requirements.txt`

**完整功能需要 NVIDIA GPU (CUDA)**，macOS 上部分高级功能不可用。
