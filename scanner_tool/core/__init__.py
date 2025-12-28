"""核心处理模块"""

# 基础导入 (需要 cv2)
try:
    from .importers import (
        BaseImporter,
        ImportResult,
        ScannerImporter,
        ARKitScenesImporter,
        ScanNetImporter,
        ReplicaImporter,
    )
    _importers_available = True
except ImportError:
    _importers_available = False

# 可视化模块 (需要 open3d)
try:
    from .visualization import (
        Visualizer,
        SceneData,
        VisualizationConfig,
        load_scene_data,
        create_trajectory,
        create_camera_frames,
        create_point_cloud,
        integrate_rgbd,
        export_mesh,
        export_point_cloud,
        visualize,
        validate_scene,
    )
    _visualization_available = True
except ImportError:
    _visualization_available = False

# 位姿估计模块 (需要 hloc, pycolmap)
try:
    from .pose_estimation import (
        BoundingBox,
        PoseEstimationResult,
        PoseEstimationConfig,
        PoseEstimator,
        SceneBoundsCalculator,
        run_sfm_pipeline,
        compute_scene_bounds,
    )
    _pose_estimation_available = True
except ImportError:
    _pose_estimation_available = False

# 特征提取模块 (需要 torch, h5py)
try:
    from .features import (
        FeatureExtractionConfig,
        FeatureExtractionResult,
        BaseFeatureExtractor,
        DinoFeatureExtractor,
        LSegFeatureExtractor,
        FeatureExtractor,
        FeatureCompressor,
        get_feature_extractor,
        extract_dino_features,
        extract_lseg_features,
        load_features,
        visualize_feature_map,
        check_feature_extraction_available,
    )
    _features_available = True
except ImportError:
    _features_available = False

# 训练模块 (需要 torch)
try:
    from .training import (
        DeviceType,
        BackendType,
        PlatformInfo,
        TrainingConfig,
        TrainingResult,
        detect_platform,
        get_device_string,
        print_platform_info,
        check_training_available,
        train_nerf,
        load_trained_model,
        query_3d_semantics,
        train_simple,
    )
    _training_available = True
except ImportError:
    _training_available = False

__all__ = []

# 条件导出 - Importers
if _importers_available:
    __all__.extend([
        'BaseImporter',
        'ImportResult',
        'ScannerImporter',
        'ARKitScenesImporter',
        'ScanNetImporter',
        'ReplicaImporter',
    ])

# 条件导出
if _visualization_available:
    __all__.extend([
        'Visualizer',
        'SceneData',
        'VisualizationConfig',
        'load_scene_data',
        'create_trajectory',
        'create_camera_frames',
        'create_point_cloud',
        'integrate_rgbd',
        'export_mesh',
        'export_point_cloud',
        'visualize',
        'validate_scene',
    ])

if _pose_estimation_available:
    __all__.extend([
        'BoundingBox',
        'PoseEstimationResult',
        'PoseEstimationConfig',
        'PoseEstimator',
        'SceneBoundsCalculator',
        'run_sfm_pipeline',
        'compute_scene_bounds',
    ])

if _features_available:
    __all__.extend([
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
    ])

if _training_available:
    __all__.extend([
        'DeviceType',
        'BackendType',
        'PlatformInfo',
        'TrainingConfig',
        'TrainingResult',
        'detect_platform',
        'get_device_string',
        'print_platform_info',
        'check_training_available',
        'train_nerf',
        'load_trained_model',
        'query_3d_semantics',
        'train_simple',
    ])

# 导出模块 (需要 cv2/PIL)
try:
    from .export import (
        ExportConfig,
        ExportResult,
        SemanticExporter,
        VideoRenderer,
        FormatConverter,
        check_export_available,
    )
    _export_available = True
except ImportError:
    _export_available = False

# 评估模块 (需要 cv2/PIL)
try:
    from .evaluation import (
        EvaluationConfig,
        EvaluationResult,
        SemanticEvaluator,
        ConfusionMatrix,
        check_evaluation_available,
    )
    _evaluation_available = True
except ImportError:
    _evaluation_available = False

if _export_available:
    __all__.extend([
        'ExportConfig',
        'ExportResult',
        'SemanticExporter',
        'VideoRenderer',
        'FormatConverter',
        'check_export_available',
    ])

if _evaluation_available:
    __all__.extend([
        'EvaluationConfig',
        'EvaluationResult',
        'SemanticEvaluator',
        'ConfusionMatrix',
        'check_evaluation_available',
    ])

# 语言查询模块 (需要 torch)
try:
    from .language import (
        LanguageConfig,
        LanguageQueryResult,
        TextEncoder,
        CLIPTextEncoder,
        LSegTextEncoder,
        LanguageSegmenter,
        get_text_encoder,
        check_language_available,
    )
    _language_available = True
except ImportError:
    _language_available = False

if _language_available:
    __all__.extend([
        'LanguageConfig',
        'LanguageQueryResult',
        'TextEncoder',
        'CLIPTextEncoder',
        'LSegTextEncoder',
        'LanguageSegmenter',
        'get_text_encoder',
        'check_language_available',
    ])

# 实时服务模块 (需要 torch)
try:
    from .realtime import (
        RealtimeConfig,
        Frame,
        FrameProcessor,
        DirectoryWatcher,
        RealtimeSegmentationService,
        run_realtime_service,
        check_realtime_available,
    )
    _realtime_available = True
except ImportError:
    _realtime_available = False

if _realtime_available:
    __all__.extend([
        'RealtimeConfig',
        'Frame',
        'FrameProcessor',
        'DirectoryWatcher',
        'RealtimeSegmentationService',
        'run_realtime_service',
        'check_realtime_available',
    ])

# 3D 点云分割模块 (需要 CUDA + Open3D + autolabel)
try:
    from .pointcloud import (
        PointCloudConfig,
        PointCloudResult,
        PointCloudSegmenter,
        check_pointcloud_available,
    )
    _pointcloud_available = True
except ImportError:
    _pointcloud_available = False

if _pointcloud_available:
    __all__.extend([
        'PointCloudConfig',
        'PointCloudResult',
        'PointCloudSegmenter',
        'check_pointcloud_available',
    ])

# NeRF 渲染模块 (需要 CUDA + OpenCV + autolabel)
try:
    from .render import (
        RenderConfig,
        RenderResult,
        FeatureTransformer,
        NeRFRenderer,
        check_render_available,
    )
    _render_available = True
except ImportError:
    _render_available = False

if _render_available:
    __all__.extend([
        'RenderConfig',
        'RenderResult',
        'FeatureTransformer',
        'NeRFRenderer',
        'check_render_available',
    ])
