# iPhone 3D Scanner

一个完整的 3D 扫描解决方案，包含 iOS 扫描应用和 PC 端处理工具。

> **致谢**: 本项目基于 [Stray Robots Scanner](https://github.com/StrayRobots/scanner) 及其附属软件开发，感谢原作者的开源贡献！

## 项目结构

```
├── scanner/          # iOS Scanner App - 🔧 深度定制开发
├── scanner_tool/     # PC 端处理工具 - 🆕 全新开发
├── StrayVisualizer/  # 原始可视化工具 - 📦 未修改（已被 scanner_tool 替代）
└── datasets/         # 扫描数据存放目录
```

---

## 🔧 iOS Scanner App (`scanner/`)

**状态：深度定制开发**

基于 [Stray Robots Scanner](https://github.com/StrayRobots/scanner) 进行了大量改进和功能增强。

### 我们的改进

| 功能 | 原项目 | 我们的版本 |
|------|--------|-----------|
| UI 设计 | 基础 UIKit | 现代 SwiftUI + 深色主题 |
| 数据传输 | 仅有线 | WiFi 无线传输 + 有线 |
| 批量操作 | 无 | 多选、批量删除、批量传输 |
| 数据管理 | 基础列表 | 搜索、重命名、统计信息 |
| 用户体验 | 基础 | 触觉反馈、动画、中文本地化 |

### 系统要求
- iPhone 12 Pro 或更新（需要 LiDAR）
- iOS 14.0+
- Xcode 14.0+

### 安装
```bash
cd scanner
pod install
open StrayScanner.xcworkspace
```

详细文档：[scanner/README.md](scanner/README.md)

---

## 🆕 PC 处理工具 (`scanner_tool/`)

**状态：全新开发**

我们从零开发的 Python 工具集，整合并扩展了多个开源项目的功能。

### 功能特性

| 功能 | 说明 | 来源 |
|------|------|------|
| WiFi 数据传输 | 接收 iOS App 传输的数据 | 🆕 新开发 |
| 点云可视化 | 查看扫描的 3D 点云 | 整合自 StrayVisualizer |
| NeRF 训练 | 神经辐射场 3D 重建 | 整合自 Autolabel |
| 语义分割 | 2D/3D 语义分割 | 整合自 Autolabel |
| 特征提取 | DINO/LSeg 视觉特征 | 整合自 Autolabel |
| 标注 GUI | 图形化标注工具 | 🆕 新开发 |
| 数据导入 | 支持多种数据集格式 | 🆕 新开发 |

### 安装
```bash
cd scanner_tool
conda create -n scanner_tool python=3.10
conda activate scanner_tool
pip install -r requirements.txt
```

### 快速使用
```bash
# 启动 WiFi 接收服务器
python -m scanner_tool.cli.main serve --port 8080

# 可视化点云
python -m scanner_tool.cli.main visualize /path/to/dataset

# 查看所有命令
python -m scanner_tool.cli.main --help
```

详细文档：[scanner_tool/README.md](scanner_tool/README.md)

---

## 📦 StrayVisualizer (`StrayVisualizer/`)

**状态：原始项目，未修改**

这是 [StrayVisualizer](https://github.com/kekeblom/StrayVisualizer) 的原始代码，保留作为参考。

> ⚠️ **注意**：StrayVisualizer 的所有功能已被 `scanner_tool` 完全覆盖和增强，建议使用 `scanner_tool`。

---

## 工作流程

```
┌─────────────────┐     WiFi      ┌─────────────────┐
│   iOS Scanner   │ ───────────▶  │   scanner_tool  │
│   (iPhone)      │               │   (PC)          │
└─────────────────┘               └─────────────────┘
        │                                 │
        ▼                                 ▼
   采集 RGB-D 数据                  处理、可视化、分析
   LiDAR + 相机                    点云、NeRF、语义分割
```

### 典型使用流程

1. **iOS 端采集数据**
   - 打开 Scanner App
   - 点击录制按钮采集数据
   - 录制完成后保存

2. **传输到 PC**
   - PC 端运行：`python -m scanner_tool.cli.main serve`
   - iOS 端选择数据集 → WiFi 传输
   - 输入 PC 的 IP:端口

3. **PC 端处理**
   - 可视化：`python -m scanner_tool.cli.main visualize datasets/xxx`
   - 更多处理功能见 scanner_tool 文档

---

## 致谢

本项目基于以下开源项目开发，感谢原作者的贡献：

| 项目 | 用途 | 链接 |
|------|------|------|
| Stray Robots Scanner | iOS App 基础 | [GitHub](https://github.com/StrayRobots/scanner) |
| StrayVisualizer | 可视化参考 | [GitHub](https://github.com/kekeblom/StrayVisualizer) |
| Autolabel | NeRF + 语义分割 | [GitHub](https://github.com/ethz-asl/autolabel) |

---

## 许可证

MIT License
