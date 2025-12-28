//
//  TransferView.swift
//  StrayScanner
//
//  WiFi 数据传输界面
//  UI Redesigned - Modern + Professional Style
//

import SwiftUI

struct TransferView: View {
    /// 数据集 URL
    let datasetURL: URL
    
    /// 数据集名称
    let datasetName: String
    
    /// 传输服务
    @StateObject private var viewModel = TransferViewModel()
    
    /// 关闭回调
    var onDismiss: (() -> Void)?
    
    var body: some View {
        NavigationView {
            ZStack {
                AppColors.background
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: AppSpacing.md) {
                        // 数据集信息卡片
                        datasetInfoCard
                        
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
            .navigationTitle("WiFi 传输")
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
    
    // MARK: - 数据集信息卡片
    
    private var datasetInfoCard: some View {
        CardView {
            HStack(spacing: AppSpacing.md) {
                // 图标
                ZStack {
                    Circle()
                        .fill(AppColors.accent.opacity(0.2))
                        .frame(width: 50, height: 50)
                    
                    Image(systemName: "folder.fill")
                        .font(.system(size: 22))
                        .foregroundColor(AppColors.accent)
                }
                
                // 信息
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text("数据集")
                        .font(AppFonts.caption)
                        .foregroundColor(AppColors.secondary)
                    
                    Text(datasetName)
                        .font(AppFonts.headline)
                        .foregroundColor(AppColors.primary)
                        .lineLimit(2)
                }
                
                Spacer()
            }
        }
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
                        StatusBadge(text: "进行中", color: AppColors.accent)
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
                    viewModel.startTransfer(datasetURL: datasetURL)
                }) {
                    HStack {
                        Image(systemName: "arrow.up.circle.fill")
                        Text("发送到电脑")
                    }
                }
                .buttonStyle(PrimaryButtonStyle(isEnabled: !viewModel.serverAddress.isEmpty))
                .disabled(viewModel.serverAddress.isEmpty)
            }
        }
    }
}

// MARK: - ViewModel

class TransferViewModel: ObservableObject {
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
    
    /// 传输服务
    private let transferService = TransferService()
    
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
    
    /// 开始传输
    func startTransfer(datasetURL: URL) {
        isTransferring = true
        progress = nil
        resultMessage = nil
        isSuccess = false
        
        transferService.uploadDataset(
            datasetURL: datasetURL,
            progress: { [weak self] progress in
                DispatchQueue.main.async {
                    self?.progress = progress
                }
            },
            completion: { [weak self] result in
                DispatchQueue.main.async {
                    self?.isTransferring = false
                    
                    switch result {
                    case .success:
                        self?.isSuccess = true
                        self?.resultMessage = "传输完成！数据集已成功发送到电脑。"
                    case .failure(let error):
                        self?.isSuccess = false
                        self?.resultMessage = error.localizedDescription
                    }
                }
            }
        )
    }
    
    /// 取消传输
    func cancelTransfer() {
        transferService.cancelTransfer()
        isTransferring = false
        resultMessage = "传输已取消"
        isSuccess = false
    }
}

// MARK: - Preview

struct TransferView_Previews: PreviewProvider {
    static var previews: some View {
        TransferView(
            datasetURL: URL(fileURLWithPath: "/tmp/test"),
            datasetName: "2024年1月15日 14:30"
        )
    }
}
