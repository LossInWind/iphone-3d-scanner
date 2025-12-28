//
//  SessionDetailView.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 12/30/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import SwiftUI
import AVKit
import CoreData

class SessionDetailViewModel: ObservableObject {
    private var dataContext: NSManagedObjectContext?
    @Published var fileSize: String = "计算中..."
    @Published var frameCount: String = "..."
    @Published var resolution: String = "..."
    @Published var player: AVPlayer?
    @Published var isPlayerReady: Bool = false

    init() {
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        self.dataContext = appDelegate?.persistentContainer.viewContext
    }
    
    func title(recording: Recording) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        
        if let created = recording.createdAt {
            return dateFormatter.string(from: created)
        } else {
            return recording.name ?? "Recording"
        }
    }
    
    func shortTitle(recording: Recording) -> String {
        // 优先显示自定义名称
        if let name = recording.name, !name.isEmpty, !name.hasPrefix("Recording ") {
            return name
        }
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        
        if let created = recording.createdAt {
            return dateFormatter.string(from: created)
        } else {
            return recording.name ?? "Recording"
        }
    }
    
    func loadMetadata(recording: Recording) {
        // 计算文件大小 - 后台线程
        if let dirPath = recording.directoryPath() {
            DispatchQueue.global(qos: .utility).async { [weak self] in
                var totalSize: Int64 = 0
                let fileManager = FileManager.default
                
                if let enumerator = fileManager.enumerator(at: dirPath, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
                    for case let fileURL as URL in enumerator {
                        if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                            totalSize += Int64(size)
                        }
                    }
                }
                
                let formatter = ByteCountFormatter()
                formatter.countStyle = .file
                
                DispatchQueue.main.async {
                    self?.fileSize = formatter.string(fromByteCount: totalSize)
                }
            }
        }
        
        // 异步获取视频信息
        if let videoURL = recording.absoluteRgbPath() {
            DispatchQueue.global(qos: .utility).async { [weak self] in
                let asset = AVURLAsset(url: videoURL, options: [AVURLAssetPreferPreciseDurationAndTimingKey: false])
                
                // 使用异步加载
                asset.loadValuesAsynchronously(forKeys: ["tracks", "duration"]) {
                    var error: NSError?
                    let tracksStatus = asset.statusOfValue(forKey: "tracks", error: &error)
                    let durationStatus = asset.statusOfValue(forKey: "duration", error: &error)
                    
                    guard tracksStatus == .loaded, durationStatus == .loaded else {
                        return
                    }
                    
                    if let videoTrack = asset.tracks(withMediaType: .video).first {
                        let fps = videoTrack.nominalFrameRate
                        let duration = CMTimeGetSeconds(asset.duration)
                        let estimatedFrames = Int(fps * Float(duration))
                        
                        let size = videoTrack.naturalSize
                        let transform = videoTrack.preferredTransform
                        let isPortrait = transform.a == 0 && abs(transform.b) == 1
                        
                        let width = isPortrait ? Int(size.height) : Int(size.width)
                        let height = isPortrait ? Int(size.width) : Int(size.height)
                        
                        DispatchQueue.main.async {
                            self?.frameCount = "\(estimatedFrames)"
                            self?.resolution = "\(width)×\(height)"
                        }
                    }
                }
            }
        }
    }
    
    func setupPlayer(recording: Recording) {
        // 异步创建播放器
        guard let url = recording.absoluteRgbPath() else { return }
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let asset = AVURLAsset(url: url, options: [
                AVURLAssetPreferPreciseDurationAndTimingKey: false
            ])
            
            let playerItem = AVPlayerItem(asset: asset)
            let newPlayer = AVPlayer(playerItem: playerItem)
            
            DispatchQueue.main.async {
                self?.player = newPlayer
                self?.isPlayerReady = true
            }
        }
    }
    
    func cleanup() {
        player?.pause()
        player = nil
        isPlayerReady = false
    }

    func delete(recording: Recording) {
        recording.deleteFiles()
        self.dataContext?.delete(recording)
        do {
            try self.dataContext?.save()
        } catch let error as NSError {
            print("Could not save recording. \(error), \(error.userInfo)")
        }
    }
}

struct SessionDetailView: View {
    @ObservedObject var viewModel = SessionDetailViewModel()
    var recording: Recording
    @Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    
    @State private var showTransferView = false
    @State private var showDeleteAlert = false
    @State private var showDeleteConfirmation = false

    let defaultUrl = URL(fileURLWithPath: "")

    var body: some View {
        ZStack {
            AppColors.background
                .ignoresSafeArea()
            
            ScrollView {
                VStack(spacing: AppSpacing.md) {
                    // 视频预览
                    videoPreviewSection
                    
                    // 数据统计
                    statsSection
                    
                    // 操作按钮
                    actionsSection
                    
                    Spacer(minLength: AppSpacing.xl)
                }
                .padding(.horizontal, AppSpacing.md)
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text(viewModel.shortTitle(recording: recording))
                    .font(AppFonts.headline)
                    .foregroundColor(AppColors.primary)
            }
        }
        .onAppear {
            // 延迟加载，让界面先显示
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                viewModel.loadMetadata(recording: recording)
                viewModel.setupPlayer(recording: recording)
            }
        }
        .onDisappear {
            viewModel.cleanup()
        }
        .sheet(isPresented: $showTransferView) {
            transferSheet
        }
        .alert(isPresented: $showDeleteAlert) {
            Alert(
                title: Text("确认删除"),
                message: Text("删除后无法恢复，确定要删除这个录制吗？"),
                primaryButton: .destructive(Text("删除")) {
                    deleteItem()
                },
                secondaryButton: .cancel(Text("取消"))
            )
        }
    }
    
    // MARK: - 视频预览
    
    private var videoPreviewSection: some View {
        CardView(padding: 0) {
            VStack(spacing: 0) {
                // 视频播放器
                if viewModel.isPlayerReady, let player = viewModel.player {
                    VideoPlayer(player: player)
                        .aspectRatio(4/3, contentMode: .fit)
                        .cornerRadius(AppCorners.large)
                } else {
                    Rectangle()
                        .fill(AppColors.cardBackgroundDark)
                        .aspectRatio(4/3, contentMode: .fit)
                        .overlay(
                            VStack(spacing: AppSpacing.sm) {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: AppColors.primary))
                                Text("加载视频...")
                                    .font(AppFonts.caption)
                                    .foregroundColor(AppColors.secondary)
                            }
                        )
                        .cornerRadius(AppCorners.large)
                }
            }
        }
    }
    
    // MARK: - 数据统计
    
    private var statsSection: some View {
        CardView {
            VStack(spacing: AppSpacing.md) {
                HStack {
                    Text("数据集信息")
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                    Spacer()
                }
                
                AppDivider()
                
                // 统计网格
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: AppSpacing.md) {
                    statRow(icon: "clock.fill", title: "时长", value: formattedDuration)
                    statRow(icon: "film.fill", title: "帧数", value: viewModel.frameCount)
                    statRow(icon: "aspectratio.fill", title: "分辨率", value: viewModel.resolution)
                    statRow(icon: "internaldrive.fill", title: "文件大小", value: viewModel.fileSize)
                }
            }
        }
    }
    
    private func statRow(icon: String, title: String, value: String) -> some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: icon)
                .font(.system(size: 16))
                .foregroundColor(AppColors.accent)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(AppFonts.caption)
                    .foregroundColor(AppColors.secondary)
                Text(value)
                    .font(AppFonts.headline)
                    .foregroundColor(AppColors.primary)
            }
            
            Spacer()
        }
        .padding(AppSpacing.sm)
        .background(
            RoundedRectangle(cornerRadius: AppCorners.small)
                .fill(AppColors.cardBackgroundDark)
        )
    }
    
    private var formattedDuration: String {
        let duration = Int(round(recording.duration))
        let minutes = duration / 60
        let seconds = duration % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
    
    // MARK: - 操作按钮
    
    private var actionsSection: some View {
        VStack(spacing: AppSpacing.sm) {
            // WiFi 传输按钮
            Button(action: {
                showTransferView = true
            }) {
                HStack {
                    Image(systemName: "wifi")
                    Text("WiFi 传输到电脑")
                }
            }
            .buttonStyle(PrimaryButtonStyle())
            
            // 分享按钮
            Button(action: shareDataset) {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("分享数据集")
                }
            }
            .buttonStyle(SecondaryButtonStyle())
            
            // 删除按钮
            Button(action: {
                showDeleteAlert = true
            }) {
                HStack {
                    Image(systemName: "trash")
                    Text("删除录制")
                }
            }
            .buttonStyle(DangerButtonStyle())
        }
    }
    
    // MARK: - 传输 Sheet
    
    private var transferSheet: some View {
        Group {
            if let datasetURL = recording.directoryPath() {
                TransferView(
                    datasetURL: datasetURL,
                    datasetName: viewModel.title(recording: recording),
                    onDismiss: {
                        showTransferView = false
                    }
                )
            } else {
                VStack(spacing: AppSpacing.md) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 40))
                        .foregroundColor(AppColors.warning)
                    
                    Text("无法找到数据集目录")
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                    
                    Button("关闭") {
                        showTransferView = false
                    }
                    .buttonStyle(SecondaryButtonStyle())
                    .padding(.horizontal, AppSpacing.xl)
                }
                .padding()
            }
        }
    }
    
    // MARK: - Actions
    
    private func shareDataset() {
        guard let dirPath = recording.directoryPath() else { return }
        
        // 确保目录存在
        guard FileManager.default.fileExists(atPath: dirPath.path) else {
            print("Directory does not exist: \(dirPath.path)")
            return
        }
        
        // 创建一个临时 zip 文件用于分享
        let zipFileName = "\(recording.name ?? "dataset").zip"
        let tempDir = FileManager.default.temporaryDirectory
        let zipURL = tempDir.appendingPathComponent(zipFileName)
        
        // 删除旧的 zip 文件（如果存在）
        try? FileManager.default.removeItem(at: zipURL)
        
        // 使用 FileManager 压缩目录
        do {
            // 创建 zip 归档
            let coordinator = NSFileCoordinator()
            var error: NSError?
            
            coordinator.coordinate(readingItemAt: dirPath, options: .forUploading, error: &error) { zipTempURL in
                do {
                    try FileManager.default.copyItem(at: zipTempURL, to: zipURL)
                } catch {
                    print("Failed to copy zip: \(error)")
                }
            }
            
            if let error = error {
                print("Coordination error: \(error)")
                // 如果压缩失败，直接分享目录
                presentShareSheet(items: [dirPath])
                return
            }
            
            // 分享 zip 文件
            if FileManager.default.fileExists(atPath: zipURL.path) {
                presentShareSheet(items: [zipURL])
            } else {
                // 回退到分享目录
                presentShareSheet(items: [dirPath])
            }
        }
    }
    
    private func presentShareSheet(items: [Any]) {
        let activityVC = UIActivityViewController(
            activityItems: items,
            applicationActivities: nil
        )
        
        // 设置 iPad 的 popover 位置
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = windowScene.windows.first?.rootViewController {
            
            // 对于 iPad，需要设置 popover 的源视图
            if let popover = activityVC.popoverPresentationController {
                popover.sourceView = rootVC.view
                popover.sourceRect = CGRect(x: rootVC.view.bounds.midX, y: rootVC.view.bounds.midY, width: 0, height: 0)
                popover.permittedArrowDirections = []
            }
            
            rootVC.present(activityVC, animated: true)
        }
    }

    private func deleteItem() {
        viewModel.delete(recording: recording)
        self.presentationMode.wrappedValue.dismiss()
    }
}

struct SessionDetailView_Previews: PreviewProvider {
    static var recording: Recording = { () -> Recording in
        let rec = Recording()
        rec.id = UUID()
        rec.name = "Placeholder name"
        rec.createdAt = Date()
        rec.duration = 30.0
        return rec
    }()

    static var previews: some View {
        NavigationView {
            SessionDetailView(recording: recording)
        }
    }
}
