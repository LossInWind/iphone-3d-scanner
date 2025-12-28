# iOS Scanner App 环境搭建指南

## 1. 打开项目

使用 Xcode 打开 workspace 文件：
```bash
open scanner/StrayScanner.xcworkspace
```

**注意**: 必须使用 `.xcworkspace` 而不是 `.xcodeproj`

## 2. 配置开发者证书 (任务 1.2)

1. 在 Xcode 中选择项目导航器中的 `StrayScanner` 项目
2. 选择 `StrayScanner` target
3. 点击 `Signing & Capabilities` 标签
4. 在 `Team` 下拉菜单中选择你的开发团队
   - 如果没有，点击 "Add Account..." 登录你的 Apple ID
5. 修改 `Bundle Identifier` 为唯一值，例如：
   - `com.yourname.StrayScanner`
   - `com.yourcompany.scanner`

## 3. 编译运行到真机 (任务 1.3)

### 设备要求
- iPhone 12 Pro / Pro Max 或更新
- iPad Pro 2020 或更新
- 必须有 LiDAR 传感器

### 步骤
1. 用 USB 线连接 iOS 设备到 Mac
2. 在设备上信任此电脑
3. 在 Xcode 顶部选择你的设备作为运行目标
4. 点击 ▶️ 运行按钮 或按 `Cmd + R`
5. 首次运行需要在设备上信任开发者：
   - 设置 → 通用 → VPN与设备管理 → 信任开发者

### 验证功能
- [ ] App 成功启动
- [ ] 相机预览正常显示
- [ ] 可以开始/停止录制
- [ ] 录制后数据集出现在列表中

## 常见问题

### "Signing for StrayScanner requires a development team"
→ 在 Signing & Capabilities 中选择你的开发团队

### "The bundle identifier is already in use"
→ 修改 Bundle Identifier 为唯一值

### "Device is not available"
→ 确保设备已解锁并信任此电脑

### 编译错误 "No such module 'ARKit'"
→ 确保选择的是真机而不是模拟器（ARKit 不支持模拟器）

## 项目结构说明

```
scanner/StrayScanner/
├── Controllers/
│   └── RecordSessionViewController.swift  # 录制控制器
├── Helpers/
│   ├── VideoEncoder.swift                 # RGB 视频编码
│   ├── DepthEncoder.swift                 # 深度图编码
│   ├── ConfidenceEncoder.swift            # 置信度图编码
│   ├── OdometryEncoder.swift              # 位姿编码
│   └── IMUEncoder.swift                   # IMU 数据编码
├── Models/
│   └── Recording+CoreData*.swift          # 数据模型
└── Views/                                 # UI 视图
```

## 下一步

完成环境搭建后，继续执行 tasks.md 中的任务 2（基础功能验证）。
