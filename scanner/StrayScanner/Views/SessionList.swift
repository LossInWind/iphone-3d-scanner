//
//  SessionList.swift
//  Stray Scanner
//
//  Created by Kenneth Blomqvist on 11/15/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import SwiftUI
import CoreData

class SessionListViewModel: ObservableObject {
    private var dataContext: NSManagedObjectContext?
    @Published var sessions: [Recording] = []
    @Published var totalStorageUsed: String = "-"
    @Published var totalRecordings: Int = 0
    @Published var isLoading: Bool = true

    init() {
        // 延迟获取 context，避免阻塞初始化
        DispatchQueue.main.async { [weak self] in
            guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return }
            self?.dataContext = appDelegate.persistentContainer.viewContext
            NotificationCenter.default.addObserver(self as Any, selector: #selector(self?.sessionsChanged), name: NSNotification.Name("sessionsChanged"), object: nil)
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    func fetchSessions() {
        guard let dataContext = dataContext else {
            // 如果 context 还没准备好，延迟重试
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) { [weak self] in
                self?.fetchSessions()
            }
            return
        }
        
        // 在后台线程执行 Core Data 查询
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let request = NSFetchRequest<NSManagedObject>(entityName: "Recording")
            request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
            
            do {
                let fetched: [NSManagedObject] = try dataContext.fetch(request)
                let sessions = fetched.compactMap { $0 as? Recording }
                
                DispatchQueue.main.async {
                    self?.sessions = sessions
                    self?.totalRecordings = sessions.count
                    self?.isLoading = false
                    self?.calculateTotalStorage()
                }
            } catch let error as NSError {
                print("Something went wrong. Error: \(error), \(error.userInfo)")
                DispatchQueue.main.async {
                    self?.isLoading = false
                }
            }
        }
    }
    
    func deleteRecordings(_ recordings: [Recording]) {
        for recording in recordings {
            recording.deleteFiles()
            dataContext?.delete(recording)
        }
        do {
            try dataContext?.save()
            fetchSessions()
            NotificationCenter.default.post(name: NSNotification.Name("sessionsChanged"), object: nil)
        } catch let error as NSError {
            print("Could not delete recordings. \(error), \(error.userInfo)")
        }
    }
    
    func renameRecording(_ recording: Recording, newName: String) {
        recording.name = newName
        do {
            try dataContext?.save()
            fetchSessions()
            NotificationCenter.default.post(name: NSNotification.Name("sessionsChanged"), object: nil)
        } catch let error as NSError {
            print("Could not rename recording. \(error), \(error.userInfo)")
        }
    }
    
    private func calculateTotalStorage() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            var totalSize: Int64 = 0
            
            for recording in self?.sessions ?? [] {
                if let dirPath = recording.directoryPath() {
                    totalSize += self?.directorySize(at: dirPath) ?? 0
                }
            }
            
            DispatchQueue.main.async {
                self?.totalStorageUsed = self?.formatBytes(totalSize) ?? "未知"
            }
        }
    }
    
    private func directorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
            for case let fileURL as URL in enumerator {
                if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(fileSize)
                }
            }
        }
        
        return totalSize
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    @objc func sessionsChanged() {
        fetchSessions()
    }
}

struct SessionList: View {
    @ObservedObject var viewModel = SessionListViewModel()
    @State private var showingInfo = false
    @State private var searchText = ""
    
    // 选择模式相关状态
    @State private var isSelectionMode = false
    @State private var selectedRecordings: Set<UUID> = []
    @State private var showDeleteAlert = false
    @State private var showShareSheet = false
    @State private var showShareOptions = false  // 分享选项菜单
    @State private var showWifiTransfer = false  // WiFi 传输视图
    
    // 重命名相关状态
    @State private var showRenameAlert = false
    @State private var renameText = ""
    @State private var recordingToRename: Recording?
    
    // 导航状态（用于解决 NavigationLink 与长按手势冲突）
    @State private var selectedNavigationId: UUID? = nil
    
    // 是否已完成初始化
    @State private var isInitialized = false
    
    // 静态变量确保 appearance 只设置一次
    private static var appearanceConfigured = false

    init() {
        // 只在第一次时配置 appearance
        if !SessionList.appearanceConfigured {
            SessionList.configureAppearance()
            SessionList.appearanceConfigured = true
        }
    }
    
    private static func configureAppearance() {
        let appearance = UINavigationBarAppearance()
        appearance.configureWithOpaqueBackground()
        appearance.backgroundColor = UIColor(named: "BackgroundColor")
        appearance.titleTextAttributes = [.foregroundColor: UIColor(named: "TextColor") ?? .white]
        appearance.largeTitleTextAttributes = [.foregroundColor: UIColor(named: "TextColor") ?? .white]
        
        UINavigationBar.appearance().standardAppearance = appearance
        UINavigationBar.appearance().scrollEdgeAppearance = appearance
        UITableView.appearance().backgroundColor = UIColor(named: "BackgroundColor")
    }
    
    var filteredSessions: [Recording] {
        if searchText.isEmpty {
            return viewModel.sessions
        }
        return viewModel.sessions.filter { recording in
            let title = sessionTitle(for: recording)
            return title.localizedCaseInsensitiveContains(searchText)
        }
    }
    
    private func sessionTitle(for recording: Recording) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        
        if let created = recording.createdAt {
            return dateFormatter.string(from: created)
        }
        return recording.name ?? "Recording"
    }
    
    // 获取选中的录制
    private var selectedRecordingsList: [Recording] {
        viewModel.sessions.filter { selectedRecordings.contains($0.id ?? UUID()) }
    }

    var body: some View {
        NavigationView {
            ZStack {
                AppColors.background
                    .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // 顶部标题栏
                    headerView
                    
                    // 顶部统计卡片（非选择模式时显示）
                    if !isSelectionMode {
                        statsHeader
                    } else {
                        selectionHeader
                    }
                    
                    // 录制列表
                    if !viewModel.sessions.isEmpty {
                        sessionListContent
                    } else {
                        emptyStateView
                    }
                    
                    // 底部按钮
                    if isSelectionMode {
                        selectionActionBar
                    } else {
                        bottomRecordButton
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .navigationBarHidden(true)
            .sheet(isPresented: $showingInfo) {
                InformationView()
            }
            .alert(isPresented: $showDeleteAlert) {
                Alert(
                    title: Text("确认删除"),
                    message: Text("确定要删除选中的 \(selectedRecordings.count) 个录制吗？此操作无法撤销。"),
                    primaryButton: .destructive(Text("删除")) {
                        deleteSelectedRecordings()
                    },
                    secondaryButton: .cancel(Text("取消"))
                )
            }
            .sheet(isPresented: $showShareSheet) {
                ShareSheet(items: getShareItems())
            }
            .sheet(isPresented: $showWifiTransfer) {
                BatchWifiTransferView(recordings: selectedRecordingsList, onDismiss: {
                    showWifiTransfer = false
                    exitSelectionMode()
                })
            }
            .actionSheet(isPresented: $showShareOptions) {
                ActionSheet(
                    title: Text("选择分享方式"),
                    message: Text("选择如何分享选中的 \(selectedRecordings.count) 个录制"),
                    buttons: [
                        .default(Text("📡 WiFi 传输到电脑")) {
                            showWifiTransfer = true
                        },
                        .default(Text("📤 系统分享")) {
                            showShareSheet = true
                        },
                        .cancel(Text("取消"))
                    ]
                )
            }
            .sheet(isPresented: $showRenameAlert) {
                RenameSheet(
                    currentName: renameText,
                    onRename: { newName in
                        if let recording = recordingToRename {
                            viewModel.renameRecording(recording, newName: newName)
                        }
                        showRenameAlert = false
                        recordingToRename = nil
                        // 如果只选了一个，退出选择模式
                        if selectedRecordings.count == 1 {
                            exitSelectionMode()
                        }
                    },
                    onCancel: {
                        showRenameAlert = false
                        recordingToRename = nil
                    }
                )
            }
            .onAppear {
                // 延迟加载数据，让界面先显示
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                    viewModel.fetchSessions()
                }
                
                // 清理已删除的条目（先在主线程获取 delegate，再在后台执行）
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    let delegate = UIApplication.shared.delegate as? AppDelegate
                    DispatchQueue.global(qos: .utility).async {
                        delegate?.appDaemon?.removeDeletedEntries()
                    }
                }
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }
    
    // MARK: - 顶部标题栏
    
    private var headerView: some View {
        HStack {
            if isSelectionMode {
                Button(action: {
                    exitSelectionMode()
                }) {
                    Text("取消")
                        .font(AppFonts.body)
                        .foregroundColor(AppColors.accent)
                }
            } else {
                Text("扫描数据")
                    .font(AppFonts.title)
                    .foregroundColor(AppColors.primary)
            }
            
            Spacer()
            
            if isSelectionMode {
                Button(action: {
                    selectAll()
                }) {
                    Text(selectedRecordings.count == viewModel.sessions.count ? "取消全选" : "全选")
                        .font(AppFonts.body)
                        .foregroundColor(AppColors.accent)
                }
            } else {
                IconButton(icon: "info.circle", action: {
                    showingInfo.toggle()
                }, size: 36, iconSize: 18)
            }
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, AppSpacing.sm)
        .padding(.bottom, AppSpacing.xs)
    }
    
    // MARK: - 选择模式头部
    
    private var selectionHeader: some View {
        CardView(padding: AppSpacing.md) {
            HStack {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 24))
                    .foregroundColor(AppColors.accent)
                
                Text("已选择 \(selectedRecordings.count) 项")
                    .font(AppFonts.headline)
                    .foregroundColor(AppColors.primary)
                
                Spacer()
                
                if selectedRecordings.count > 0 {
                    Text(selectedStorageSize)
                        .font(AppFonts.caption)
                        .foregroundColor(AppColors.secondary)
                }
            }
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, AppSpacing.sm)
    }
    
    private var selectedStorageSize: String {
        var totalSize: Int64 = 0
        for recording in selectedRecordingsList {
            if let dirPath = recording.directoryPath() {
                totalSize += directorySize(at: dirPath)
            }
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: totalSize)
    }
    
    private func directorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
            for case let fileURL as URL in enumerator {
                if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(fileSize)
                }
            }
        }
        return totalSize
    }
    
    // MARK: - 统计头部
    
    private var statsHeader: some View {
        CardView(padding: AppSpacing.md) {
            HStack(spacing: AppSpacing.md) {
                StatItem(
                    icon: "folder.fill",
                    title: "数据集",
                    value: "\(viewModel.totalRecordings)"
                )
                
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 1, height: 50)
                
                StatItem(
                    icon: "internaldrive.fill",
                    title: "存储占用",
                    value: viewModel.totalStorageUsed
                )
                
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(width: 1, height: 50)
                
                StatItem(
                    icon: "clock.fill",
                    title: "总时长",
                    value: totalDuration
                )
            }
        }
        .padding(.horizontal, AppSpacing.md)
        .padding(.top, AppSpacing.sm)
    }
    
    private var totalDuration: String {
        let total = viewModel.sessions.reduce(0) { $0 + $1.duration }
        let minutes = Int(total) / 60
        let seconds = Int(total) % 60
        if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        }
        return "\(seconds)s"
    }
    
    // MARK: - 列表内容
    
    private var sessionListContent: some View {
        ScrollView {
            LazyVStack(spacing: AppSpacing.sm) {
                // 搜索栏
                searchBar
                    .padding(.horizontal, AppSpacing.md)
                    .padding(.top, AppSpacing.sm)
                
                // 提示文字
                if !isSelectionMode {
                    Text("长按可进入选择模式")
                        .font(AppFonts.caption)
                        .foregroundColor(AppColors.secondary.opacity(0.6))
                        .padding(.top, AppSpacing.xs)
                }
                
                // 列表项
                ForEach(Array(filteredSessions.enumerated()), id: \.element) { index, recording in
                    sessionRowItem(recording: recording)
                        .padding(.horizontal, AppSpacing.md)
                }
            }
            .padding(.bottom, 100)
        }
    }
    
    private func sessionRowItem(recording: Recording) -> some View {
        let isSelected = selectedRecordings.contains(recording.id ?? UUID())
        
        return Group {
            if isSelectionMode {
                // 选择模式：点击选择/取消选择
                Button(action: {
                    toggleSelection(recording)
                }) {
                    SessionRowCard(
                        session: recording,
                        isSelectionMode: true,
                        isSelected: isSelected
                    )
                }
                .buttonStyle(PlainButtonStyle())
            } else {
                // 正常模式：使用 ZStack + NavigationLink 解决长按手势冲突
                ZStack {
                    // 隐藏的 NavigationLink
                    NavigationLink(
                        destination: SessionDetailView(recording: recording),
                        tag: recording.id ?? UUID(),
                        selection: $selectedNavigationId
                    ) {
                        EmptyView()
                    }
                    .opacity(0)
                    
                    // 可见的卡片，支持点击和长按
                    SessionRowCard(
                        session: recording,
                        isSelectionMode: false,
                        isSelected: false
                    )
                    .contentShape(Rectangle())
                    .onTapGesture {
                        // 点击导航到详情
                        selectedNavigationId = recording.id
                    }
                    .onLongPressGesture(minimumDuration: 0.5) {
                        // 长按进入选择模式
                        enterSelectionMode(with: recording)
                    }
                }
            }
        }
    }
    
    private var searchBar: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(AppColors.secondary)
            
            TextField("搜索录制...", text: $searchText)
                .foregroundColor(AppColors.primary)
        }
        .padding(AppSpacing.sm)
        .background(
            RoundedRectangle(cornerRadius: AppCorners.medium)
                .fill(AppColors.cardBackground)
        )
    }
    
    // MARK: - 空状态
    
    private var emptyStateView: some View {
        Spacer()
            .frame(maxHeight: .infinity)
            .overlay(
                EmptyStateView(
                    icon: "video.badge.plus",
                    title: "暂无录制",
                    message: "点击下方按钮开始录制您的第一个 3D 扫描数据集"
                )
            )
    }
    
    // MARK: - 选择模式操作栏
    
    private var selectionActionBar: some View {
        VStack {
            Spacer()
            
            HStack(spacing: AppSpacing.sm) {
                // 重命名按钮（只有选中一个时可用）
                Button(action: {
                    startRename()
                }) {
                    VStack(spacing: AppSpacing.xs) {
                        Image(systemName: "pencil")
                            .font(.system(size: 20))
                        Text("重命名")
                            .font(AppFonts.caption)
                    }
                    .foregroundColor(selectedRecordings.count == 1 ? AppColors.accent : AppColors.secondary.opacity(0.5))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, AppSpacing.sm)
                }
                .disabled(selectedRecordings.count != 1)
                
                // 分享按钮
                Button(action: {
                    showShareOptions = true
                }) {
                    VStack(spacing: AppSpacing.xs) {
                        Image(systemName: "square.and.arrow.up")
                            .font(.system(size: 20))
                        Text("分享")
                            .font(AppFonts.caption)
                    }
                    .foregroundColor(selectedRecordings.isEmpty ? AppColors.secondary.opacity(0.5) : AppColors.accent)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, AppSpacing.sm)
                }
                .disabled(selectedRecordings.isEmpty)
                
                // 删除按钮
                Button(action: {
                    showDeleteAlert = true
                }) {
                    VStack(spacing: AppSpacing.xs) {
                        Image(systemName: "trash")
                            .font(.system(size: 20))
                        Text("删除")
                            .font(AppFonts.caption)
                    }
                    .foregroundColor(selectedRecordings.isEmpty ? AppColors.secondary.opacity(0.5) : AppColors.danger)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, AppSpacing.sm)
                }
                .disabled(selectedRecordings.isEmpty)
            }
            .padding(.horizontal, AppSpacing.lg)
            .padding(.vertical, AppSpacing.sm)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.large)
                    .fill(AppColors.cardBackground)
                    .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: -5)
            )
            .padding(.horizontal, AppSpacing.md)
            .padding(.bottom, AppSpacing.lg)
        }
    }
    
    // MARK: - 底部录制按钮
    
    private var bottomRecordButton: some View {
        VStack {
            Spacer()
            
            NavigationLink(destination: NewSessionView()) {
                HStack(spacing: AppSpacing.sm) {
                    Image(systemName: "record.circle")
                        .font(.system(size: 22, weight: .semibold))
                    Text("开始新录制")
                        .font(AppFonts.headline)
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, AppSpacing.md)
                .background(
                    RoundedRectangle(cornerRadius: AppCorners.extraLarge)
                        .fill(AppColors.primaryGradient)
                        .shadow(color: Color.blue.opacity(0.3), radius: 10, x: 0, y: 5)
                )
            }
            .padding(.horizontal, AppSpacing.lg)
            .padding(.bottom, AppSpacing.lg)
        }
        .background(
            LinearGradient(
                colors: [AppColors.background.opacity(0), AppColors.background],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 120)
            .allowsHitTesting(false)
        )
    }
    
    // MARK: - 选择模式操作
    
    private func enterSelectionMode(with recording: Recording) {
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        
        isSelectionMode = true
        if let id = recording.id {
            selectedRecordings.insert(id)
        }
    }
    
    private func exitSelectionMode() {
        isSelectionMode = false
        selectedRecordings.removeAll()
    }
    
    private func toggleSelection(_ recording: Recording) {
        guard let id = recording.id else { return }
        
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
        
        if selectedRecordings.contains(id) {
            selectedRecordings.remove(id)
        } else {
            selectedRecordings.insert(id)
        }
    }
    
    private func selectAll() {
        if selectedRecordings.count == viewModel.sessions.count {
            selectedRecordings.removeAll()
        } else {
            selectedRecordings = Set(viewModel.sessions.compactMap { $0.id })
        }
    }
    
    private func deleteSelectedRecordings() {
        let recordingsToDelete = selectedRecordingsList
        viewModel.deleteRecordings(recordingsToDelete)
        exitSelectionMode()
    }
    
    private func getShareItems() -> [URL] {
        return selectedRecordingsList.compactMap { $0.directoryPath() }
    }
    
    private func startRename() {
        guard selectedRecordings.count == 1,
              let recording = selectedRecordingsList.first else { return }
        
        recordingToRename = recording
        renameText = recording.name ?? sessionTitle(for: recording)
        showRenameAlert = true
    }
}

// MARK: - 分享 Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - 重命名 Sheet

struct RenameSheet: View {
    @State var currentName: String
    let onRename: (String) -> Void
    let onCancel: () -> Void
    
    var body: some View {
        NavigationView {
            ZStack {
                AppColors.background
                    .ignoresSafeArea()
                
                VStack(spacing: AppSpacing.lg) {
                    // 图标
                    ZStack {
                        Circle()
                            .fill(AppColors.accent.opacity(0.2))
                            .frame(width: 80, height: 80)
                        
                        Image(systemName: "pencil.circle.fill")
                            .font(.system(size: 40))
                            .foregroundColor(AppColors.accent)
                    }
                    .padding(.top, AppSpacing.xl)
                    
                    Text("重命名录制")
                        .font(AppFonts.title2)
                        .foregroundColor(AppColors.primary)
                    
                    // 输入框
                    VStack(alignment: .leading, spacing: AppSpacing.xs) {
                        Text("名称")
                            .font(AppFonts.caption)
                            .foregroundColor(AppColors.secondary)
                        
                        TextField("输入新名称", text: $currentName)
                            .font(AppFonts.body)
                            .foregroundColor(AppColors.primary)
                            .padding()
                            .background(
                                RoundedRectangle(cornerRadius: AppCorners.medium)
                                    .fill(AppColors.cardBackground)
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: AppCorners.medium)
                                    .stroke(AppColors.accent.opacity(0.5), lineWidth: 1)
                            )
                    }
                    .padding(.horizontal, AppSpacing.lg)
                    
                    Spacer()
                    
                    // 按钮
                    VStack(spacing: AppSpacing.sm) {
                        Button(action: {
                            if !currentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                onRename(currentName.trimmingCharacters(in: .whitespacesAndNewlines))
                            }
                        }) {
                            Text("确认")
                                .font(AppFonts.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: AppCorners.medium)
                                        .fill(currentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray : AppColors.accent)
                                )
                        }
                        .disabled(currentName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        
                        Button(action: onCancel) {
                            Text("取消")
                                .font(AppFonts.headline)
                                .foregroundColor(AppColors.primary)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(
                                    RoundedRectangle(cornerRadius: AppCorners.medium)
                                        .stroke(AppColors.primary.opacity(0.3), lineWidth: 1)
                                )
                        }
                    }
                    .padding(.horizontal, AppSpacing.lg)
                    .padding(.bottom, AppSpacing.lg)
                }
            }
            .navigationBarHidden(true)
        }
    }
}

// MARK: - 新的列表行卡片样式

struct SessionRowCard: View {
    var session: Recording
    var isSelectionMode: Bool = false
    var isSelected: Bool = false
    
    @State private var thumbnailImage: UIImage?
    @State private var fileSize: String = "..."
    
    var body: some View {
        CardView(padding: AppSpacing.sm) {
            HStack(spacing: AppSpacing.md) {
                // 选择指示器
                if isSelectionMode {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 24))
                        .foregroundColor(isSelected ? AppColors.accent : AppColors.secondary.opacity(0.5))
                }
                
                // 缩略图
                thumbnailView
                
                // 信息
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text(sessionTitle())
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                        .lineLimit(1)
                    
                    HStack(spacing: AppSpacing.md) {
                        Label(formattedDuration, systemImage: "clock")
                            .font(AppFonts.caption)
                            .foregroundColor(AppColors.secondary)
                        
                        Label(fileSize, systemImage: "doc")
                            .font(AppFonts.caption)
                            .foregroundColor(AppColors.secondary)
                    }
                }
                
                Spacer()
                
                // 箭头（非选择模式时显示）
                if !isSelectionMode {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(AppColors.secondary.opacity(0.5))
                }
            }
        }
        .overlay(
            RoundedRectangle(cornerRadius: AppCorners.large)
                .stroke(isSelected ? AppColors.accent : Color.clear, lineWidth: 2)
        )
        .onAppear {
            loadThumbnail()
            calculateFileSize()
        }
    }
    
    private var thumbnailView: some View {
        Group {
            if let image = thumbnailImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle()
                    .fill(AppColors.cardBackgroundDark)
                    .overlay(
                        Image(systemName: "video.fill")
                            .foregroundColor(AppColors.secondary.opacity(0.5))
                    )
            }
        }
        .frame(width: 80, height: 60)
        .cornerRadius(AppCorners.small)
        .clipped()
    }
    
    private var formattedDuration: String {
        let duration = Int(round(session.duration))
        if duration >= 60 {
            let minutes = duration / 60
            let seconds = duration % 60
            return "\(minutes):\(String(format: "%02d", seconds))"
        }
        return "\(duration)s"
    }
    
    private func sessionTitle() -> String {
        // 优先显示自定义名称
        if let name = session.name, !name.isEmpty, !name.hasPrefix("Recording ") {
            return name
        }
        
        // 否则显示日期时间
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        
        if let created = session.createdAt {
            return dateFormatter.string(from: created)
        }
        return session.name ?? "录制"
    }
    
    private func loadThumbnail() {
        guard let videoURL = session.absoluteRgbPath() else { return }
        
        // 使用更低优先级的队列，避免阻塞
        DispatchQueue.global(qos: .utility).async {
            let asset = AVURLAsset(url: videoURL, options: [
                AVURLAssetPreferPreciseDurationAndTimingKey: false
            ])
            
            let imageGenerator = AVAssetImageGenerator(asset: asset)
            imageGenerator.appliesPreferredTrackTransform = true
            imageGenerator.maximumSize = CGSize(width: 160, height: 120) // 限制缩略图大小
            imageGenerator.requestedTimeToleranceBefore = .zero
            imageGenerator.requestedTimeToleranceAfter = CMTime(seconds: 2, preferredTimescale: 600) // 允许更大的时间容差
            
            let time = CMTime(seconds: 0.1, preferredTimescale: 600)
            
            do {
                let cgImage = try imageGenerator.copyCGImage(at: time, actualTime: nil)
                let uiImage = UIImage(cgImage: cgImage)
                
                DispatchQueue.main.async {
                    self.thumbnailImage = uiImage
                }
            } catch {
                // 静默失败，显示默认图标
                print("Error generating thumbnail: \(error)")
            }
        }
    }
    
    private func calculateFileSize() {
        guard let dirPath = session.directoryPath() else {
            fileSize = "未知"
            return
        }
        
        DispatchQueue.global(qos: .background).async {
            var totalSize: Int64 = 0
            let fileManager = FileManager.default
            
            if let enumerator = fileManager.enumerator(at: dirPath, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
                for case let fileURL as URL in enumerator {
                    if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                        totalSize += Int64(fileSize)
                    }
                }
            }
            
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file
            
            DispatchQueue.main.async {
                self.fileSize = formatter.string(fromByteCount: totalSize)
            }
        }
    }
}

import AVFoundation

struct SessionList_Previews: PreviewProvider {
    static var previews: some View {
        SessionList()
    }
}

// MARK: - 批量 WiFi 传输视图

struct BatchWifiTransferView: View {
    let recordings: [Recording]
    var onDismiss: (() -> Void)?
    
    @StateObject private var viewModel = BatchTransferViewModel()
    
    var body: some View {
        NavigationView {
            ZStack {
                AppColors.background
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: AppSpacing.md) {
                        // 数据集列表卡片
                        datasetsInfoCard
                        
                        // 服务器配置卡片
                        serverConfigCard
                        
                        // 传输状态卡片
                        if viewModel.isTransferring || viewModel.resultMessage != nil {
                            transferStatusCard
                        }
                        
                        Spacer(minLength: AppSpacing.xl)
                        
                        // 操作按钮
                        actionButtons
                    }
                    .padding(AppSpacing.md)
                }
            }
            .navigationTitle("WiFi 批量传输")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: {
                        onDismiss?()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(AppColors.secondary)
                    }
                    .disabled(viewModel.isTransferring)
                }
            }
        }
    }
    
    // MARK: - 数据集列表卡片
    
    private var datasetsInfoCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                // 标题
                HStack {
                    ZStack {
                        Circle()
                            .fill(AppColors.accent.opacity(0.2))
                            .frame(width: 40, height: 40)
                        
                        Image(systemName: "folder.fill.badge.plus")
                            .font(.system(size: 18))
                            .foregroundColor(AppColors.accent)
                    }
                    
                    VStack(alignment: .leading, spacing: 2) {
                        Text("待传输数据集")
                            .font(AppFonts.headline)
                            .foregroundColor(AppColors.primary)
                        
                        Text("\(recordings.count) 个数据集")
                            .font(AppFonts.caption)
                            .foregroundColor(AppColors.secondary)
                    }
                    
                    Spacer()
                    
                    Text(totalSize)
                        .font(AppFonts.mono)
                        .foregroundColor(AppColors.accent)
                }
                
                // 数据集列表
                VStack(spacing: AppSpacing.xs) {
                    ForEach(recordings, id: \.id) { recording in
                        HStack {
                            Image(systemName: viewModel.completedRecordings.contains(recording.id ?? UUID()) ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(viewModel.completedRecordings.contains(recording.id ?? UUID()) ? AppColors.success : AppColors.secondary.opacity(0.5))
                            
                            Text(recordingName(recording))
                                .font(AppFonts.body)
                                .foregroundColor(AppColors.primary)
                                .lineLimit(1)
                            
                            Spacer()
                            
                            if viewModel.currentRecordingId == recording.id {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: AppColors.accent))
                                    .scaleEffect(0.8)
                            }
                        }
                        .padding(.vertical, AppSpacing.xs)
                    }
                }
            }
        }
    }
    
    private var totalSize: String {
        var total: Int64 = 0
        for recording in recordings {
            if let dirPath = recording.directoryPath() {
                total += directorySize(at: dirPath)
            }
        }
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: total)
    }
    
    private func directorySize(at url: URL) -> Int64 {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        
        if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) {
            for case let fileURL as URL in enumerator {
                if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                    totalSize += Int64(fileSize)
                }
            }
        }
        return totalSize
    }
    
    private func recordingName(_ recording: Recording) -> String {
        if let name = recording.name, !name.isEmpty, !name.hasPrefix("Recording ") {
            return name
        }
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .medium
        dateFormatter.timeStyle = .short
        
        if let created = recording.createdAt {
            return dateFormatter.string(from: created)
        }
        return recording.name ?? "录制"
    }
    
    // MARK: - 服务器配置卡片
    
    private var serverConfigCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                // 标题
                HStack {
                    Image(systemName: "network")
                        .foregroundColor(AppColors.accent)
                    Text("服务器地址")
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                }
                
                // 输入框
                HStack(spacing: AppSpacing.sm) {
                    TextField("192.168.1.100:8080", text: $viewModel.serverAddress)
                        .font(AppFonts.mono)
                        .foregroundColor(AppColors.primary)
                        .padding(AppSpacing.sm)
                        .background(
                            RoundedRectangle(cornerRadius: AppCorners.small)
                                .fill(AppColors.cardBackgroundDark)
                        )
                        .keyboardType(.numbersAndPunctuation)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                    
                    // 测试连接按钮
                    Button(action: {
                        viewModel.testConnection()
                    }) {
                        Image(systemName: "antenna.radiowaves.left.and.right")
                            .font(.system(size: 18))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(
                                RoundedRectangle(cornerRadius: AppCorners.small)
                                    .fill(viewModel.serverAddress.isEmpty ? Color.gray : AppColors.accent)
                            )
                    }
                    .disabled(viewModel.isTransferring || viewModel.serverAddress.isEmpty)
                }
                
                // 连接状态
                if let status = viewModel.connectionStatus {
                    HStack(spacing: AppSpacing.sm) {
                        Image(systemName: status.isConnected ? "checkmark.circle.fill" : "xmark.circle.fill")
                            .foregroundColor(status.isConnected ? AppColors.success : AppColors.danger)
                        
                        Text(status.message)
                            .font(AppFonts.caption)
                            .foregroundColor(status.isConnected ? AppColors.success : AppColors.danger)
                    }
                    .padding(.top, AppSpacing.xs)
                }
                
                // 提示
                Text("请确保手机和电脑在同一 WiFi 网络下")
                    .font(AppFonts.caption)
                    .foregroundColor(AppColors.secondary)
            }
        }
    }
    
    // MARK: - 传输状态卡片
    
    private var transferStatusCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                // 标题
                HStack {
                    Image(systemName: viewModel.isTransferring ? "arrow.up.circle" : (viewModel.isSuccess ? "checkmark.circle" : "exclamationmark.triangle"))
                        .foregroundColor(viewModel.isTransferring ? AppColors.accent : (viewModel.isSuccess ? AppColors.success : AppColors.warning))
                    
                    Text(viewModel.isTransferring ? "传输中" : (viewModel.isSuccess ? "传输完成" : "传输结果"))
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                    
                    Spacer()
                    
                    if viewModel.isTransferring {
                        Text("\(viewModel.completedRecordings.count)/\(recordings.count)")
                            .font(AppFonts.mono)
                            .foregroundColor(AppColors.accent)
                    } else if viewModel.isSuccess {
                        StatusBadge(text: "成功", color: AppColors.success)
                    }
                }
                
                if viewModel.isTransferring {
                    // 进度显示
                    if let progress = viewModel.progress {
                        VStack(alignment: .leading, spacing: AppSpacing.sm) {
                            Text(progress.description)
                                .font(AppFonts.caption)
                                .foregroundColor(AppColors.secondary)
                            
                            // 进度条
                            GeometryReader { geometry in
                                ZStack(alignment: .leading) {
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(AppColors.cardBackgroundDark)
                                        .frame(height: 8)
                                    
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(AppColors.primaryGradient)
                                        .frame(width: geometry.size.width * CGFloat(progress.percentage / 100), height: 8)
                                }
                            }
                            .frame(height: 8)
                            
                            HStack {
                                Text("\(Int(progress.percentage))%")
                                    .font(AppFonts.mono)
                                    .foregroundColor(AppColors.accent)
                                
                                Spacer()
                            }
                        }
                    } else {
                        HStack {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: AppColors.accent))
                            Text("准备中...")
                                .font(AppFonts.caption)
                                .foregroundColor(AppColors.secondary)
                        }
                    }
                } else if let message = viewModel.resultMessage {
                    // 结果消息
                    Text(message)
                        .font(AppFonts.body)
                        .foregroundColor(viewModel.isSuccess ? AppColors.success : AppColors.warning)
                }
            }
        }
    }
    
    // MARK: - 操作按钮
    
    private var actionButtons: some View {
        VStack(spacing: AppSpacing.sm) {
            if viewModel.isTransferring {
                Button(action: {
                    viewModel.cancelTransfer()
                }) {
                    HStack {
                        Image(systemName: "xmark.circle")
                        Text("取消传输")
                    }
                }
                .buttonStyle(DangerButtonStyle())
            } else {
                Button(action: {
                    viewModel.startBatchTransfer(recordings: recordings)
                }) {
                    HStack {
                        Image(systemName: "arrow.up.circle.fill")
                        Text("发送 \(recordings.count) 个数据集到电脑")
                    }
                }
                .buttonStyle(PrimaryButtonStyle(isEnabled: !viewModel.serverAddress.isEmpty))
                .disabled(viewModel.serverAddress.isEmpty)
            }
        }
    }
}

// MARK: - 批量传输 ViewModel

class BatchTransferViewModel: ObservableObject {
    /// 服务器地址
    @Published var serverAddress: String {
        didSet {
            transferService.serverAddress = serverAddress
        }
    }
    
    /// 是否正在传输
    @Published var isTransferring: Bool = false
    
    /// 传输进度
    @Published var progress: TransferProgress?
    
    /// 连接状态
    @Published var connectionStatus: ConnectionStatus?
    
    /// 结果消息
    @Published var resultMessage: String?
    
    /// 是否成功
    @Published var isSuccess: Bool = false
    
    /// 已完成的录制 ID
    @Published var completedRecordings: Set<UUID> = []
    
    /// 当前正在传输的录制 ID
    @Published var currentRecordingId: UUID?
    
    /// 传输服务
    private let transferService = TransferService()
    
    /// 是否已取消
    private var isCancelled: Bool = false
    
    struct ConnectionStatus {
        let isConnected: Bool
        let message: String
    }
    
    init() {
        self.serverAddress = transferService.serverAddress
    }
    
    /// 测试连接
    func testConnection() {
        connectionStatus = nil
        
        transferService.testConnection { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success:
                    self?.connectionStatus = ConnectionStatus(
                        isConnected: true,
                        message: "连接成功"
                    )
                case .failure(let error):
                    self?.connectionStatus = ConnectionStatus(
                        isConnected: false,
                        message: error.localizedDescription
                    )
                }
            }
        }
    }
    
    /// 开始批量传输
    func startBatchTransfer(recordings: [Recording]) {
        isTransferring = true
        progress = nil
        resultMessage = nil
        isSuccess = false
        completedRecordings.removeAll()
        currentRecordingId = nil
        isCancelled = false
        
        // 递归传输每个录制
        transferNext(recordings: recordings, index: 0)
    }
    
    private func transferNext(recordings: [Recording], index: Int) {
        guard !isCancelled else {
            DispatchQueue.main.async {
                self.isTransferring = false
                self.currentRecordingId = nil
                self.resultMessage = "传输已取消，已完成 \(self.completedRecordings.count)/\(recordings.count) 个"
                self.isSuccess = false
            }
            return
        }
        
        guard index < recordings.count else {
            // 全部完成
            DispatchQueue.main.async {
                self.isTransferring = false
                self.currentRecordingId = nil
                self.isSuccess = true
                self.resultMessage = "全部传输完成！\(recordings.count) 个数据集已成功发送到电脑。"
            }
            return
        }
        
        let recording = recordings[index]
        guard let datasetURL = recording.directoryPath() else {
            // 跳过无效的录制
            transferNext(recordings: recordings, index: index + 1)
            return
        }
        
        DispatchQueue.main.async {
            self.currentRecordingId = recording.id
        }
        
        transferService.uploadDataset(
            datasetURL: datasetURL,
            progress: { [weak self] progress in
                DispatchQueue.main.async {
                    self?.progress = progress
                }
            },
            completion: { [weak self] result in
                guard let self = self else { return }
                
                DispatchQueue.main.async {
                    switch result {
                    case .success:
                        if let id = recording.id {
                            self.completedRecordings.insert(id)
                        }
                        // 继续下一个
                        self.transferNext(recordings: recordings, index: index + 1)
                        
                    case .failure(let error):
                        self.isTransferring = false
                        self.currentRecordingId = nil
                        self.isSuccess = false
                        self.resultMessage = "传输失败：\(error.localizedDescription)\n已完成 \(self.completedRecordings.count)/\(recordings.count) 个"
                    }
                }
            }
        )
    }
    
    /// 取消传输
    func cancelTransfer() {
        isCancelled = true
        transferService.cancelTransfer()
    }
}
