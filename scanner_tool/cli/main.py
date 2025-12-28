#!/usr/bin/env python3
"""
Scanner Tool CLI - 统一命令行入口

用法:
    scanner-tool visualize <path> [options]
    scanner-tool convert <input> <output> [options]
    scanner-tool import <format> <input> <output> [options]
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def cmd_visualize(args):
    """可视化命令"""
    from scanner_tool.core.visualization import (
        Visualizer, VisualizationConfig, validate_scene,
        export_mesh, export_point_cloud
    )
    
    # 验证场景
    valid, error_msg = validate_scene(args.path)
    if not valid:
        print(f"Error: Invalid scene directory: {args.path}")
        print(f"  {error_msg}")
        return 1
    
    print(f"Loading scene from: {args.path}")
    
    # 创建配置
    config = VisualizationConfig(
        every=args.every,
        confidence_level=args.confidence,
        voxel_size=args.voxel_size
    )
    
    # 创建可视化器
    vis = Visualizer(args.path, config)
    
    # 默认显示所有
    if not any([args.trajectory, args.frames, args.point_clouds, args.integrate]):
        args.trajectory = True
        args.frames = True
        args.point_clouds = True
    
    if args.trajectory:
        print("Creating trajectory...")
        vis.show_trajectory()
    
    if args.frames:
        print("Creating camera frames...")
        vis.show_frames()
    
    if args.point_clouds:
        print("Creating point clouds...")
        vis.show_point_clouds()
        
        # 导出点云 (如果指定)
        if args.pointcloud_output:
            pc = vis._geometries[-1]  # 最后添加的是点云
            print(f"Saving point cloud to: {args.pointcloud_output}")
            if export_point_cloud(pc, args.pointcloud_output):
                print("  ✓ Point cloud saved successfully")
            else:
                print("  ✗ Failed to save point cloud")
    
    if args.integrate:
        print("Integrating RGB-D...")
        mesh = vis.integrate_mesh()
        
        # 导出网格 (如果指定)
        if args.mesh_output:
            print(f"Saving mesh to: {args.mesh_output}")
            if export_mesh(mesh, args.mesh_output):
                print("  ✓ Mesh saved successfully")
            else:
                print("  ✗ Failed to save mesh")
    
    print("Opening visualization window...")
    vis.visualize()
    return 0


def cmd_convert(args):
    """数据转换命令"""
    print(f"Converting {args.input} to {args.output}")
    print(f"Format: {args.format}")
    
    if args.format == 'open3d':
        # 调用 StrayVisualizer 的转换脚本
        import subprocess
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'StrayVisualizer', 'convert_to_open3d.py'
        )
        result = subprocess.run([
            sys.executable, script_path, args.input, args.output
        ])
        return result.returncode
    else:
        print(f"Unsupported format: {args.format}")
        return 1


def cmd_map(args):
    """位姿估计命令 (SfM 管线)"""
    from scanner_tool.core.pose_estimation import (
        PoseEstimator, PoseEstimationConfig,
        SceneBoundsCalculator
    )
    
    print(f"Running SfM pipeline on: {args.path}")
    
    # 创建配置
    config = PoseEstimationConfig(
        debug=args.debug,
        visualize=args.visualize
    )
    
    # 创建位姿估计器
    estimator = PoseEstimator(config)
    
    # 检查依赖
    if not estimator.is_available:
        print("Error: SfM dependencies not available.")
        print("Please install hloc and pycolmap:")
        print("  pip install hloc pycolmap")
        return 1
    
    try:
        # 运行 SfM
        print("Running Structure from Motion...")
        result = estimator.run_sfm(args.path)
        
        print(f"✓ SfM completed successfully")
        print(f"  - Estimated {len(result.poses)} camera poses")
        print(f"  - Intrinsics saved to: {args.path}/intrinsics.txt")
        print(f"  - Distortion saved to: {args.path}/distortion_parameters.txt")
        
        if result.bounds:
            print(f"  - Bounds saved to: {args.path}/bbox.txt")
            print(f"    Min: {result.bounds.min_bounds}")
            print(f"    Max: {result.bounds.max_bounds}")
        
        return 0
        
    except Exception as e:
        print(f"✗ SfM failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_bounds(args):
    """计算场景边界命令"""
    from scanner_tool.core.pose_estimation import SceneBoundsCalculator
    
    print(f"Computing scene bounds for: {args.path}")
    
    calculator = SceneBoundsCalculator(args.path)
    
    if not calculator.is_available:
        print("Error: Open3D not available.")
        print("Please install open3d: pip install open3d")
        return 1
    
    try:
        bounds = calculator.compute_bounds(
            stride=args.stride,
            nb_neighbors=args.neighbors,
            std_ratio=args.std_ratio
        )
        
        if args.output:
            calculator.save_bounds(bounds, args.output)
            print(f"✓ Bounds saved to: {args.output}")
        else:
            calculator.save_bounds(bounds)
            print(f"✓ Bounds saved to: {args.path}/bbox.txt")
        
        print(f"  Min: {bounds.min_bounds}")
        print(f"  Max: {bounds.max_bounds}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Bounds computation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_features(args):
    """特征提取命令"""
    from scanner_tool.core.features import (
        FeatureExtractor, FeatureExtractionConfig,
        check_feature_extraction_available, _get_device_string
    )
    
    print(f"Extracting {args.type} features from: {args.path}")
    
    # 检查功能可用性
    availability = check_feature_extraction_available()
    
    if not availability['torch']:
        print("Error: PyTorch not available.")
        print("Please install PyTorch: pip install torch torchvision")
        return 1
    
    # 显示设备信息
    print(f"Device: {availability['device']}")
    
    if args.type == 'dino' and not availability['dino']:
        print("Error: DINO feature extraction not available.")
        return 1
    
    if args.type == 'lseg':
        if not args.checkpoint:
            print("Error: LSeg requires --checkpoint argument")
            return 1
        if not availability['lseg']:
            print("Error: LSeg requires CUDA. Not available on this platform.")
            return 1
    
    # 创建配置
    config = FeatureExtractionConfig(
        feature_type=args.type,
        output_dim=args.dim,
        autoencode=args.autoencode,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size
    )
    
    # 创建提取器
    extractor = FeatureExtractor(config)
    
    try:
        # 提取特征
        result = extractor.extract(
            args.path,
            output_path=args.output,
            visualize=args.visualize,
            video_path=args.video
        )
        
        if result.success:
            print(f"✓ Feature extraction completed")
            print(f"  - Type: {result.feature_type}")
            print(f"  - Frames: {result.num_frames}")
            print(f"  - Shape: {result.feature_shape}")
            print(f"  - Output: {result.output_path}")
            print(f"  - PCA available: {result.pca_available}")
            return 0
        else:
            print(f"✗ Feature extraction failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def cmd_train(args):
    """NeRF 训练命令"""
    from scanner_tool.core.training import (
        train_nerf, TrainingConfig,
        check_training_available
    )
    
    print(f"Training NeRF model on: {args.path}")
    
    # 检查功能可用性
    availability = check_training_available()
    
    if not availability['cuda']:
        print("Warning: CUDA not available.")
        print("Full NeRF training with semantic features requires NVIDIA GPU.")
        if not availability['autolabel']:
            print("Error: autolabel module not available.")
            return 1
    
    if not availability['full_training']:
        print("\nNote: Running in limited mode.")
        print("For full semantic training, use NVIDIA GPU with CUDA.\n")
    
    # 显示平台信息
    print("\n--- Platform Info ---")
    print(f"CUDA available: {availability['cuda']}")
    print(f"autolabel available: {availability['autolabel']}")
    print(f"Full training: {availability['full_training']}")
    print("---------------------\n")
    
    # 创建配置
    config = TrainingConfig(
        iterations=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        rgb_weight=args.rgb_weight,
        depth_weight=args.depth_weight,
        semantic_weight=getattr(args, 'semantic_weight', 0.04),
        feature_weight=getattr(args, 'feature_weight', 0.01),
        features=getattr(args, 'features', None),
        eval_after_train=getattr(args, 'eval', False)
    )
    
    try:
        # 训练
        print(f"Starting training for {args.iters} iterations...")
        result = train_nerf(
            args.path, 
            config, 
            workspace=getattr(args, 'workspace', None)
        )
        
        if result.success:
            print(f"\n✓ Training completed")
            print(f"  - Model directory: {result.model_path}")
            print(f"  - Iterations: {result.iterations}")
            return 0
        else:
            print(f"✗ Training failed: {result.error_message}")
            return 1
            
    except Exception as e:
        print(f"✗ Training failed: {e}")
        if getattr(args, 'debug', False):
            import traceback
            traceback.print_exc()
        return 1


def cmd_import(args):
    """数据导入命令"""
    from scanner_tool.core.importers import (
        ScannerImporter, ARKitScenesImporter, 
        ScanNetImporter, ReplicaImporter
    )
    
    print(f"Importing {args.format} data from {args.input} to {args.output}")
    
    importers = {
        'scanner': ScannerImporter,
        'arkitscenes': ARKitScenesImporter,
        'scannet': ScanNetImporter,
        'replica': ReplicaImporter,
    }
    
    if args.format not in importers:
        print(f"Unsupported format: {args.format}")
        print(f"Supported formats: {', '.join(importers.keys())}")
        return 1
    
    importer = importers[args.format]()
    
    # 验证输入
    if not importer.validate_input(args.input):
        print(f"Error: Invalid {args.format} dataset: {args.input}")
        return 1
    
    # 构建参数
    kwargs = {}
    if args.format == 'scanner':
        kwargs['subsample'] = args.subsample
        kwargs['rotate'] = args.rotate
        kwargs['confidence_threshold'] = args.confidence
    elif args.format == 'arkitscenes':
        kwargs['confidence_threshold'] = args.confidence
    elif args.format == 'scannet':
        kwargs['stride'] = args.stride
        kwargs['max_frames'] = args.max_frames
    elif args.format == 'replica':
        kwargs['compute_bounds'] = not args.no_bounds
    
    # 执行导入
    result = importer.import_data(args.input, args.output, **kwargs)
    
    if result.success:
        print(f"✓ {result}")
        return 0
    else:
        print(f"✗ {result}")
        return 1


def cmd_platform(args):
    """显示平台信息命令"""
    from scanner_tool.core.training import (
        detect_platform, check_training_available, print_platform_info
    )
    
    print("=== Scanner Tool Platform Info ===\n")
    
    # 平台信息
    print_platform_info()
    
    # 功能可用性
    print("\n--- Feature Availability ---")
    availability = check_training_available()
    print(f"PyTorch: {'✓' if availability['torch'] else '✗'}")
    print(f"CUDA: {'✓' if availability['cuda'] else '✗'}")
    print(f"MPS (Apple Silicon): {'✓' if availability['mps'] else '✗'}")
    print(f"Autolabel: {'✓' if availability['autolabel'] else '✗'}")
    print(f"Training available: {'✓' if availability['available'] else '✗'}")
    
    # 推荐配置
    print("\n--- Recommended Configuration ---")
    info = detect_platform()
    if info.cuda_available:
        print("✓ NVIDIA GPU detected - optimal performance")
        if availability['autolabel']:
            print("  Using: autolabel + tiny-cuda-nn (fastest)")
        else:
            print("  Using: Pure PyTorch (install autolabel for better performance)")
    elif info.mps_available:
        print("✓ Apple Silicon detected - good performance")
        print("  Using: Pure PyTorch + MPS acceleration")
    else:
        print("⚠ CPU only - training will be slow")
        print("  Using: Pure PyTorch (CPU)")
    
    return 0


def cmd_label(args):
    """语义标注 GUI 命令"""
    from scanner_tool.gui import check_gui_available, run_labeling_gui, LabelingConfig
    
    print(f"Opening labeling GUI for: {args.path}")
    
    # 检查 GUI 可用性
    availability = check_gui_available()
    if not availability['available']:
        print("Error: GUI dependencies not available.")
        if not availability['pyqt6']:
            print("  - PyQt6 not installed: pip install PyQt6")
        if not availability['pil']:
            print("  - PIL not installed: pip install Pillow")
        return 1
    
    # 创建配置
    config = LabelingConfig(
        brush_size=args.brush_size,
        canvas_width=args.width,
        auto_save=not args.no_autosave
    )
    
    # 运行 GUI
    success = run_labeling_gui(args.path, config)
    return 0 if success else 1


def cmd_export(args):
    """导出命令"""
    from scanner_tool.core.export import (
        SemanticExporter, ExportConfig, FormatConverter,
        check_export_available
    )
    
    print(f"Exporting from: {args.path}")
    
    # 检查可用性
    availability = check_export_available()
    if not availability['basic']:
        print("Error: Export dependencies not available.")
        print("Please install: pip install opencv-python Pillow")
        return 1
    
    if args.format == 'semantic':
        # 导出语义分割图
        config = ExportConfig(
            objects_per_class=args.objects
        )
        exporter = SemanticExporter(config)
        
        if args.from_model:
            result = exporter.export_from_model(
                args.path,
                model_dir=args.model_dir,
                output_path=args.output
            )
        else:
            result = exporter.export_from_annotations(
                args.path,
                output_path=args.output
            )
        
        if result.success:
            print(f"✓ Exported {result.num_frames} frames to: {result.output_path}")
            return 0
        else:
            print(f"✗ Export failed: {result.error_message}")
            return 1
    
    elif args.format == 'instant-ngp':
        # 转换为 instant-ngp 格式
        converter = FormatConverter()
        result = converter.to_instant_ngp(args.path, args.output)
        
        if result.success:
            print(f"✓ Converted {result.num_frames} frames to: {result.output_path}")
            return 0
        else:
            print(f"✗ Conversion failed: {result.error_message}")
            return 1
    
    else:
        print(f"Unsupported format: {args.format}")
        return 1


def cmd_evaluate(args):
    """评估命令"""
    from scanner_tool.core.evaluation import (
        SemanticEvaluator, EvaluationConfig,
        check_evaluation_available
    )
    
    print(f"Evaluating: {args.pred} vs {args.gt}")
    
    # 检查可用性
    availability = check_evaluation_available()
    if not availability['available']:
        print("Error: Evaluation dependencies not available.")
        print("Please install: pip install opencv-python Pillow")
        return 1
    
    # 创建评估器
    config = EvaluationConfig(
        ignore_background=not args.include_background
    )
    evaluator = SemanticEvaluator(config)
    
    # 评估
    result = evaluator.evaluate_scene(args.pred, args.gt)
    
    if result.success:
        print(result)
        
        # 保存结果
        if args.output:
            evaluator.save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")
        
        return 0
    else:
        print(f"✗ Evaluation failed: {result.error_message}")
        return 1


def cmd_query(args):
    """语言查询命令"""
    from scanner_tool.core.language import (
        LanguageSegmenter, LanguageConfig,
        check_language_available
    )
    
    print(f"Language query on: {args.path}")
    print(f"Prompts: {args.prompts}")
    
    # 检查可用性
    availability = check_language_available()
    if not availability['available']:
        print("Error: Language query dependencies not available.")
        print("Please install: pip install torch")
        return 1
    
    if args.type == 'lseg' and not availability['lseg']:
        print("Error: LSeg requires CUDA.")
        return 1
    
    # 创建配置
    config = LanguageConfig(
        feature_type=args.type,
        checkpoint=args.checkpoint
    )
    
    # 创建分割器
    segmenter = LanguageSegmenter(config)
    
    if not segmenter.is_available:
        print(f"Error: {args.type} encoder not available.")
        if args.type == 'clip':
            print("Please install: pip install clip")
        return 1
    
    # 执行分割
    result = segmenter.segment_scene(
        args.path,
        args.prompts,
        output_path=args.output
    )
    
    if result.success:
        print(f"✓ Language segmentation completed")
        print(f"  Classes: {result.class_names}")
        if args.output:
            print(f"  Output: {args.output}")
        return 0
    else:
        print(f"✗ Query failed: {result.error_message}")
        return 1


def cmd_realtime(args):
    """实时分割服务命令"""
    from scanner_tool.core.realtime import (
        RealtimeSegmentationService, RealtimeConfig,
        check_realtime_available
    )
    
    print(f"Starting realtime segmentation service")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    
    # 检查可用性
    availability = check_realtime_available()
    if not availability['available']:
        print("Error: Realtime service dependencies not available.")
        print("Please install: pip install torch Pillow")
        return 1
    
    # 创建配置
    config = RealtimeConfig(
        input_dir=args.input,
        output_dir=args.output,
        feature_type=args.type,
        checkpoint=args.checkpoint,
        prompts=args.prompts,
        config_file=args.config,
        save_features=args.save_features
    )
    
    # 创建并运行服务
    service = RealtimeSegmentationService(config)
    
    if not service.is_available:
        print("Error: Service not available.")
        return 1
    
    try:
        service.run()
        return 0
    except KeyboardInterrupt:
        print("\nService stopped by user")
        return 0
    except Exception as e:
        print(f"✗ Service error: {e}")
        return 1


def cmd_serve(args):
    """WiFi 传输服务器命令"""
    from scanner_tool.core.transfer import (
        TransferServer, TransferConfig,
        check_transfer_available
    )
    
    # 检查可用性
    availability = check_transfer_available()
    if not availability['available']:
        print("Error: Transfer server not available.")
        return 1
    
    # 创建配置
    config = TransferConfig(
        port=args.port,
        output_dir=args.output
    )
    
    # 创建并运行服务器
    server = TransferServer(config)
    
    try:
        server.start()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"✗ Server error: {e}")
        return 1


def cmd_pointcloud(args):
    """3D 点云语义分割命令"""
    from scanner_tool.core.pointcloud import (
        PointCloudSegmenter, PointCloudConfig,
        check_pointcloud_available
    )
    
    print(f"3D point cloud segmentation on: {args.path}")
    
    # 检查可用性
    availability = check_pointcloud_available()
    if not availability['available']:
        print("Error: Point cloud segmentation requires CUDA + Open3D + autolabel")
        print(f"  CUDA: {'✓' if availability['cuda'] else '✗'}")
        print(f"  Open3D: {'✓' if availability['open3d'] else '✗'}")
        print(f"  autolabel: {'✓' if availability['autolabel'] else '✗'}")
        return 1
    
    # 创建配置
    config = PointCloudConfig(
        batch_size=args.batch_size,
        stride=args.stride,
        features=args.features
    )
    
    # 创建分割器
    segmenter = PointCloudSegmenter(config)
    
    # 确定输出路径
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.path, 'pointcloud.ply')
    
    # 执行分割
    if args.prompts:
        # 语义分割模式
        if not args.checkpoint:
            print("Error: --checkpoint is required for semantic segmentation")
            return 1
        
        print(f"Prompts: {args.prompts}")
        result = segmenter.segment_pointcloud(
            args.path,
            output_path,
            args.prompts,
            args.checkpoint,
            workspace=args.workspace,
            visualize=args.visualize
        )
    else:
        # 仅提取点云
        result = segmenter.extract_pointcloud(
            args.path,
            output_path,
            workspace=args.workspace,
            visualize=args.visualize
        )
    
    if result.success:
        print(f"✓ Point cloud saved to: {result.output_path}")
        print(f"  Points: {result.num_points}")
        return 0
    else:
        print(f"✗ Failed: {result.error_message}")
        return 1


def cmd_render(args):
    """NeRF 渲染视频命令"""
    from scanner_tool.core.render import (
        NeRFRenderer, RenderConfig,
        check_render_available
    )
    
    print(f"Rendering NeRF video from: {args.path}")
    
    # 检查可用性
    availability = check_render_available()
    if not availability['available']:
        print("Error: Rendering requires CUDA + OpenCV + autolabel")
        print(f"  CUDA: {'✓' if availability['cuda'] else '✗'}")
        print(f"  OpenCV: {'✓' if availability['cv2'] else '✗'}")
        print(f"  autolabel: {'✓' if availability['autolabel'] else '✗'}")
        return 1
    
    # 创建配置
    config = RenderConfig(
        fps=args.fps,
        stride=args.stride,
        max_depth=args.max_depth
    )
    
    # 创建渲染器
    renderer = NeRFRenderer(config)
    
    # 确定输出路径
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.path, 'render.mp4')
    
    # 渲染
    result = renderer.render_video(
        args.path,
        output_path,
        model_dir=args.model_dir,
        classes=args.classes,
        feature_checkpoint=args.checkpoint
    )
    
    if result.success:
        print(f"✓ Video saved to: {result.output_path}")
        print(f"  Frames: {result.num_frames}")
        return 0
    else:
        print(f"✗ Failed: {result.error_message}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        prog='scanner-tool',
        description='PC 端 3D 场景处理工具'
    )
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # visualize 子命令
    vis_parser = subparsers.add_parser('visualize', help='可视化数据集')
    vis_parser.add_argument('path', type=str, help='数据集路径')
    vis_parser.add_argument('--trajectory', '-t', action='store_true', 
                           help='显示相机轨迹')
    vis_parser.add_argument('--frames', '-f', action='store_true', 
                           help='显示相机坐标系')
    vis_parser.add_argument('--point-clouds', '-p', action='store_true', 
                           help='显示点云')
    vis_parser.add_argument('--integrate', '-i', action='store_true', 
                           help='RGB-D 积分重建')
    vis_parser.add_argument('--mesh-output', '-o', type=str, default=None,
                           help='网格输出文件路径 (PLY/OBJ/STL)')
    vis_parser.add_argument('--pointcloud-output', type=str, default=None,
                           help='点云输出文件路径 (PLY/PCD)')
    vis_parser.add_argument('--every', type=int, default=60, 
                           help='每隔 N 帧采样 (默认: 60)')
    vis_parser.add_argument('--voxel-size', type=float, default=0.015, 
                           help='体素大小 (米, 默认: 0.015)')
    vis_parser.add_argument('--confidence', '-c', type=int, default=1,
                           help='置信度过滤阈值 (0/1/2, 默认: 1)')
    vis_parser.set_defaults(func=cmd_visualize)
    
    # convert 子命令
    conv_parser = subparsers.add_parser('convert', help='转换数据格式')
    conv_parser.add_argument('input', type=str, help='输入路径')
    conv_parser.add_argument('output', type=str, help='输出路径')
    conv_parser.add_argument('--format', '-f', type=str, default='open3d',
                            choices=['open3d', 'rosbag'],
                            help='输出格式 (默认: open3d)')
    conv_parser.set_defaults(func=cmd_convert)
    
    # import 子命令
    import_parser = subparsers.add_parser('import', help='导入数据集')
    import_parser.add_argument('format', type=str,
                              choices=['scanner', 'arkitscenes', 'scannet', 'replica'],
                              help='数据集格式')
    import_parser.add_argument('input', type=str, help='输入数据集路径')
    import_parser.add_argument('output', type=str, help='输出目录路径')
    import_parser.add_argument('--subsample', type=int, default=1,
                              help='帧采样间隔 (scanner 格式, 默认: 1)')
    import_parser.add_argument('--rotate', action='store_true',
                              help='旋转 90 度 (scanner 格式)')
    import_parser.add_argument('--confidence', '-c', type=int, default=2,
                              help='深度置信度阈值 (scanner/arkitscenes, 默认: 2)')
    import_parser.add_argument('--stride', type=int, default=5,
                              help='帧采样步长 (scannet 格式, 默认: 5)')
    import_parser.add_argument('--max-frames', type=int, default=750,
                              help='最大帧数 (scannet 格式, 默认: 750)')
    import_parser.add_argument('--no-bounds', action='store_true',
                              help='不计算场景边界 (replica 格式)')
    import_parser.set_defaults(func=cmd_import)
    
    # map 子命令 (SfM 位姿估计)
    map_parser = subparsers.add_parser('map', help='运行 SfM 位姿估计')
    map_parser.add_argument('path', type=str, help='场景目录路径')
    map_parser.add_argument('--debug', '-d', action='store_true',
                           help='保存调试信息到 /tmp/sfm_debug')
    map_parser.add_argument('--visualize', '-v', action='store_true',
                           help='可视化 SfM 结果')
    map_parser.set_defaults(func=cmd_map)
    
    # bounds 子命令 (场景边界计算)
    bounds_parser = subparsers.add_parser('bounds', help='计算场景边界')
    bounds_parser.add_argument('path', type=str, help='场景目录路径')
    bounds_parser.add_argument('--output', '-o', type=str, default=None,
                              help='输出文件路径 (默认: <path>/bbox.txt)')
    bounds_parser.add_argument('--stride', type=int, default=1,
                              help='帧采样步长 (默认: 1)')
    bounds_parser.add_argument('--neighbors', type=int, default=20,
                              help='统计滤波邻居数 (默认: 20)')
    bounds_parser.add_argument('--std-ratio', type=float, default=2.0,
                              help='统计滤波标准差比率 (默认: 2.0)')
    bounds_parser.add_argument('--debug', '-d', action='store_true',
                              help='显示调试信息')
    bounds_parser.set_defaults(func=cmd_bounds)
    
    # features 子命令 (特征提取)
    feat_parser = subparsers.add_parser('features', help='提取视觉特征 (DINO/LSeg)')
    feat_parser.add_argument('path', type=str, help='场景目录路径')
    feat_parser.add_argument('--type', '-t', type=str, default='dino',
                            choices=['dino', 'lseg'],
                            help='特征类型 (默认: dino)')
    feat_parser.add_argument('--dim', type=int, default=64,
                            help='输出特征维度 (默认: 64)')
    feat_parser.add_argument('--autoencode', '-a', action='store_true',
                            help='使用自编码器压缩特征')
    feat_parser.add_argument('--checkpoint', '-c', type=str, default=None,
                            help='LSeg 模型检查点路径 (LSeg 必需)')
    feat_parser.add_argument('--batch-size', '-b', type=int, default=2,
                            help='批处理大小 (默认: 2)')
    feat_parser.add_argument('--output', '-o', type=str, default=None,
                            help='输出 HDF5 文件路径 (默认: <path>/features.hdf)')
    feat_parser.add_argument('--visualize', '-v', action='store_true',
                            help='可视化特征图')
    feat_parser.add_argument('--video', type=str, default=None,
                            help='特征可视化视频输出路径')
    feat_parser.add_argument('--debug', '-d', action='store_true',
                            help='显示调试信息')
    feat_parser.set_defaults(func=cmd_features)
    
    # train 子命令 (NeRF 训练)
    train_parser = subparsers.add_parser('train', help='训练 NeRF 模型')
    train_parser.add_argument('path', type=str, help='场景目录路径')
    train_parser.add_argument('--iters', '-i', type=int, default=10000,
                             help='训练迭代次数 (默认: 10000)')
    train_parser.add_argument('--batch-size', '-b', type=int, default=4096,
                             help='批处理大小 (默认: 4096)')
    train_parser.add_argument('--lr', type=float, default=5e-3,
                             help='学习率 (默认: 5e-3)')
    train_parser.add_argument('--rgb-weight', type=float, default=1.0,
                             help='RGB 损失权重 (默认: 1.0)')
    train_parser.add_argument('--depth-weight', type=float, default=0.1,
                             help='深度损失权重 (默认: 0.1)')
    train_parser.add_argument('--semantic-weight', type=float, default=1.0,
                             help='语义损失权重 (默认: 1.0)')
    train_parser.add_argument('--feature-weight', type=float, default=0.5,
                             help='特征损失权重 (默认: 0.5)')
    train_parser.add_argument('--features', '-f', type=str, default=None,
                             choices=['dino', 'lseg'],
                             help='特征监督类型')
    train_parser.add_argument('--feature-dim', type=int, default=64,
                             help='特征维度 (默认: 64)')
    train_parser.add_argument('--geometric-features', '-g', type=int, default=15,
                             help='几何特征维度 (默认: 15)')
    train_parser.add_argument('--encoding', type=str, default='hg+freq',
                             choices=['freq', 'hg', 'hg+freq'],
                             help='位置编码类型 (默认: hg+freq)')
    train_parser.add_argument('--workspace', '-w', type=str, default=None,
                             help='工作目录 (默认: <path>/nerf/)')
    train_parser.add_argument('--eval', '-e', action='store_true',
                             help='训练后评估模型')
    train_parser.add_argument('--debug', '-d', action='store_true',
                             help='显示调试信息')
    train_parser.set_defaults(func=cmd_train)
    
    # platform 子命令 (平台信息)
    platform_parser = subparsers.add_parser('platform', help='显示平台和设备信息')
    platform_parser.set_defaults(func=cmd_platform)
    
    # label 子命令 (语义标注 GUI)
    label_parser = subparsers.add_parser('label', help='打开语义标注 GUI')
    label_parser.add_argument('path', type=str, help='场景目录路径')
    label_parser.add_argument('--brush-size', '-b', type=int, default=5,
                             help='画笔大小 (默认: 5)')
    label_parser.add_argument('--width', '-w', type=int, default=720,
                             help='画布宽度 (默认: 720)')
    label_parser.add_argument('--no-autosave', action='store_true',
                             help='禁用自动保存')
    label_parser.set_defaults(func=cmd_label)
    
    # export 子命令 (导出)
    export_parser = subparsers.add_parser('export', help='导出语义分割图或转换格式')
    export_parser.add_argument('path', type=str, help='场景目录路径')
    export_parser.add_argument('--format', '-f', type=str, default='semantic',
                              choices=['semantic', 'instant-ngp'],
                              help='导出格式 (默认: semantic)')
    export_parser.add_argument('--output', '-o', type=str, default=None,
                              help='输出目录路径')
    export_parser.add_argument('--from-model', action='store_true',
                              help='从训练模型导出 (而非手动标注)')
    export_parser.add_argument('--model-dir', type=str, default=None,
                              help='模型目录路径 (--from-model 时使用)')
    export_parser.add_argument('--objects', type=int, default=None,
                              help='每类保留的最大连通域数 (后处理)')
    export_parser.set_defaults(func=cmd_export)
    
    # evaluate 子命令 (评估)
    eval_parser = subparsers.add_parser('evaluate', help='评估语义分割结果')
    eval_parser.add_argument('pred', type=str, help='预测结果目录')
    eval_parser.add_argument('gt', type=str, help='真值目录')
    eval_parser.add_argument('--output', '-o', type=str, default=None,
                            help='结果输出文件 (JSON)')
    eval_parser.add_argument('--include-background', action='store_true',
                            help='包含背景类 (默认忽略)')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # query 子命令 (语言查询)
    query_parser = subparsers.add_parser('query', help='基于文本的语义分割')
    query_parser.add_argument('path', type=str, help='场景目录路径')
    query_parser.add_argument('--prompts', '-p', type=str, nargs='+', required=True,
                             help='文本提示列表 (如: "chair" "table" "floor")')
    query_parser.add_argument('--type', '-t', type=str, default='clip',
                             choices=['clip', 'lseg'],
                             help='特征类型 (默认: clip)')
    query_parser.add_argument('--checkpoint', '-c', type=str, default=None,
                             help='LSeg 检查点路径 (LSeg 必需)')
    query_parser.add_argument('--output', '-o', type=str, default=None,
                             help='输出目录路径')
    query_parser.set_defaults(func=cmd_query)
    
    # realtime 子命令 (实时分割服务)
    realtime_parser = subparsers.add_parser('realtime', help='启动实时语义分割服务')
    realtime_parser.add_argument('--input', '-i', type=str, required=True,
                                help='输入目录路径 (监控 rgb/ 子目录)')
    realtime_parser.add_argument('--output', '-o', type=str, required=True,
                                help='输出目录路径')
    realtime_parser.add_argument('--prompts', '-p', type=str, nargs='+',
                                default=['background', 'object'],
                                help='分割类别 (默认: background object)')
    realtime_parser.add_argument('--type', '-t', type=str, default='clip',
                                choices=['clip', 'lseg'],
                                help='特征类型 (默认: clip)')
    realtime_parser.add_argument('--checkpoint', '-c', type=str, default=None,
                                help='LSeg 检查点路径 (LSeg 必需)')
    realtime_parser.add_argument('--config', type=str, default=None,
                                help='动态配置文件路径 (JSON)')
    realtime_parser.add_argument('--save-features', action='store_true',
                                help='保存特征图 (NPY 格式)')
    realtime_parser.set_defaults(func=cmd_realtime)
    
    # serve 子命令 (WiFi 传输服务器)
    serve_parser = subparsers.add_parser('serve', help='启动 WiFi 传输服务器接收 iOS 数据')
    serve_parser.add_argument('--port', '-p', type=int, default=8080,
                             help='服务器端口 (默认: 8080)')
    serve_parser.add_argument('--output', '-o', type=str, default='./datasets',
                             help='数据集保存目录 (默认: ./datasets)')
    serve_parser.set_defaults(func=cmd_serve)
    
    # pointcloud 子命令 (3D 点云语义分割)
    pc_parser = subparsers.add_parser('pointcloud', help='3D 点云语义分割 (需要 CUDA)')
    pc_parser.add_argument('path', type=str, help='场景目录路径')
    pc_parser.add_argument('--output', '-o', type=str, default=None,
                          help='输出点云文件路径 (默认: <path>/pointcloud.ply)')
    pc_parser.add_argument('--prompts', '-p', type=str, nargs='+', default=None,
                          help='语义分割文本提示 (如: "chair" "table" "floor")')
    pc_parser.add_argument('--checkpoint', '-c', type=str, default=None,
                          help='特征提取器检查点路径 (语义分割必需)')
    pc_parser.add_argument('--features', '-f', type=str, default='lseg',
                          choices=['lseg', 'dino'],
                          help='特征类型 (默认: lseg)')
    pc_parser.add_argument('--workspace', '-w', type=str, default=None,
                          help='NeRF 模型工作目录')
    pc_parser.add_argument('--batch-size', '-b', type=int, default=8192,
                          help='批处理大小 (默认: 8192)')
    pc_parser.add_argument('--stride', type=int, default=1,
                          help='帧采样步长 (默认: 1)')
    pc_parser.add_argument('--visualize', '-v', action='store_true',
                          help='可视化结果')
    pc_parser.set_defaults(func=cmd_pointcloud)
    
    # render 子命令 (NeRF 渲染视频)
    render_parser = subparsers.add_parser('render', help='从 NeRF 模型渲染视频 (需要 CUDA)')
    render_parser.add_argument('path', type=str, help='场景目录路径')
    render_parser.add_argument('--output', '-o', type=str, default=None,
                              help='输出视频文件路径 (默认: <path>/render.mp4)')
    render_parser.add_argument('--model-dir', '-m', type=str, default=None,
                              help='NeRF 模型目录路径')
    render_parser.add_argument('--classes', type=str, nargs='+', default=None,
                              help='语义类别列表 (用于开放词汇分割)')
    render_parser.add_argument('--checkpoint', '-c', type=str, default=None,
                              help='特征提取器检查点路径')
    render_parser.add_argument('--fps', type=int, default=5,
                              help='输出视频帧率 (默认: 5)')
    render_parser.add_argument('--stride', type=int, default=1,
                              help='帧采样步长 (默认: 1)')
    render_parser.add_argument('--max-depth', type=float, default=7.5,
                              help='深度可视化最大值 (默认: 7.5)')
    render_parser.set_defaults(func=cmd_render)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
