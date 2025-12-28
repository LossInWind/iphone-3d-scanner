# iPhone 3D Scanner

一个完整的 3D 扫描解决方案，包含 iOS 扫描应用和 PC 端处理工具。

> **致谢**: 本项目基于 [Stray Robots Scanner](https://github.com/StrayRobots/scanner) 及其附属软件开发，感谢原作者的开源贡献！

## 项目结构

```
├── scanner/          # iOS Scanner App (Swift/SwiftUI)
├── scanner_tool/     # PC 端处理工具 (Python)
├── StrayVisualizer/  # 数据可视化工具
└── datasets/         # 扫描数据存放目录（不包含在仓库中）
```

## iOS Scanner App

使用 iPhone 的 LiDAR 传感器进行 3D 扫描，支持：
- RGB + 深度数据采集
- IMU 数据记录
- 相机位姿追踪
- WiFi 数据传输到 PC
- 现代化 UI 设计

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

## PC 处理工具

Python 工具集，用于处理和分析扫描数据：
- 点云生成和可视化
- 数据格式转换
- WiFi 数据接收服务器
- 自动标注功能

### 安装
```bash
cd scanner_tool
pip install -e .
```

### 使用
```bash
# 启动 WiFi 接收服务器
scanner-tool server --port 8080

# 可视化点云
scanner-tool visualize /path/to/dataset

# 导出数据
scanner-tool export /path/to/dataset --format ply
```

## WiFi 传输

iOS App 可以通过 WiFi 直接将扫描数据传输到 PC：

1. 在 PC 上启动接收服务器：`scanner-tool server`
2. 确保手机和电脑在同一 WiFi 网络
3. 在 App 中输入服务器地址并传输

## 参考项目

本项目参考和基于以下开源项目开发：

- **[Stray Robots Scanner](https://github.com/StrayRobots/scanner)** - 原始 iOS 扫描应用
- **[StrayVisualizer](https://github.com/kekeblom/StrayVisualizer)** - 数据可视化工具
- **[Autolabel](https://github.com/ethz-asl/autolabel)** - 交互式体积标注工具

## 许可证

MIT License
