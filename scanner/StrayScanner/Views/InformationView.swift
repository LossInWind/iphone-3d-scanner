//
//  InformationView.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 2/28/21.
//  Copyright © 2021 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import SwiftUI

struct InformationView: View {
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            ZStack {
                AppColors.background
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: AppSpacing.md) {
                        // 应用信息卡片
                        appInfoCard
                        
                        // 功能介绍卡片
                        featuresCard
                        
                        // 数据传输说明卡片
                        transferGuideCard
                        
                        // 链接卡片
                        linksCard
                        
                        // 免责声明
                        disclaimerCard
                        
                        // 版本信息
                        versionInfo
                    }
                    .padding(AppSpacing.md)
                }
            }
            .navigationTitle("关于")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        presentationMode.wrappedValue.dismiss()
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 24))
                            .foregroundColor(AppColors.secondary)
                    }
                }
            }
        }
    }
    
    // MARK: - 应用信息卡片
    
    private var appInfoCard: some View {
        CardView {
            VStack(spacing: AppSpacing.md) {
                // 应用图标
                ZStack {
                    RoundedRectangle(cornerRadius: AppCorners.large)
                        .fill(AppColors.primaryGradient)
                        .frame(width: 80, height: 80)
                    
                    Image(systemName: "camera.viewfinder")
                        .font(.system(size: 36))
                        .foregroundColor(.white)
                }
                
                VStack(spacing: AppSpacing.xs) {
                    Text("Stray Scanner")
                        .font(AppFonts.title2)
                        .foregroundColor(AppColors.primary)
                    
                    Text("3D 扫描数据采集工具")
                        .font(AppFonts.body)
                        .foregroundColor(AppColors.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, AppSpacing.md)
        }
    }
    
    // MARK: - 功能介绍卡片
    
    private var featuresCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                sectionHeader(icon: "sparkles", title: "功能特点")
                
                VStack(alignment: .leading, spacing: AppSpacing.sm) {
                    featureRow(icon: "video.fill", text: "录制 RGB 视频和深度数据")
                    featureRow(icon: "cube.fill", text: "利用 LiDAR 获取精确深度信息")
                    featureRow(icon: "location.fill", text: "记录相机位姿和 IMU 数据")
                    featureRow(icon: "wifi", text: "WiFi 无线传输到电脑")
                    featureRow(icon: "slider.horizontal.3", text: "可调节帧率设置")
                }
            }
        }
    }
    
    private func featureRow(icon: String, text: String) -> some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(AppColors.accent)
                .frame(width: 24)
            
            Text(text)
                .font(AppFonts.body)
                .foregroundColor(AppColors.primary)
            
            Spacer()
        }
    }
    
    // MARK: - 数据传输说明卡片
    
    private var transferGuideCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                sectionHeader(icon: "arrow.up.arrow.down", title: "数据传输方式")
                
                VStack(alignment: .leading, spacing: AppSpacing.md) {
                    // WiFi 传输
                    transferMethodRow(
                        icon: "wifi",
                        title: "WiFi 传输（推荐）",
                        description: "在电脑上运行接收服务，通过 WiFi 直接传输数据集。"
                    )
                    
                    AppDivider()
                    
                    // 有线传输
                    transferMethodRow(
                        icon: "cable.connector",
                        title: "有线传输",
                        description: "Mac: 通过 Finder 访问设备文件\nWindows: 通过 iTunes 访问"
                    )
                    
                    AppDivider()
                    
                    // 文件 App
                    transferMethodRow(
                        icon: "folder",
                        title: "文件 App",
                        description: "在「文件」App 中访问：\n浏览 > 我的 iPhone > Stray Scanner"
                    )
                }
            }
        }
    }
    
    private func transferMethodRow(icon: String, title: String, description: String) -> some View {
        HStack(alignment: .top, spacing: AppSpacing.md) {
            ZStack {
                Circle()
                    .fill(AppColors.accent.opacity(0.2))
                    .frame(width: 36, height: 36)
                
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundColor(AppColors.accent)
            }
            
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text(title)
                    .font(AppFonts.headline)
                    .foregroundColor(AppColors.primary)
                
                Text(description)
                    .font(AppFonts.caption)
                    .foregroundColor(AppColors.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            
            Spacer()
        }
    }
    
    // MARK: - 链接卡片
    
    private var linksCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                sectionHeader(icon: "link", title: "相关链接")
                
                VStack(spacing: AppSpacing.sm) {
                    linkButton(
                        icon: "chevron.left.forwardslash.chevron.right",
                        title: "GitHub 源代码",
                        url: "https://github.com/LossInWind/iphone-3d-scanner"
                    )
                    
                    linkButton(
                        icon: "book",
                        title: "使用文档",
                        url: "https://github.com/LossInWind/iphone-3d-scanner#readme"
                    )
                    
                    linkButton(
                        icon: "exclamationmark.bubble",
                        title: "报告问题",
                        url: "https://github.com/LossInWind/iphone-3d-scanner/issues"
                    )
                    
                    linkButton(
                        icon: "heart.fill",
                        title: "原始项目 (Stray Robots)",
                        url: "https://github.com/StrayRobots/scanner"
                    )
                }
            }
        }
    }
    
    private func linkButton(icon: String, title: String, url: String) -> some View {
        Button(action: {
            if let url = URL(string: url) {
                UIApplication.shared.open(url)
            }
        }) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 16))
                    .foregroundColor(AppColors.accent)
                    .frame(width: 24)
                
                Text(title)
                    .font(AppFonts.body)
                    .foregroundColor(AppColors.primary)
                
                Spacer()
                
                Image(systemName: "arrow.up.right")
                    .font(.system(size: 12))
                    .foregroundColor(AppColors.secondary)
            }
            .padding(AppSpacing.sm)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.small)
                    .fill(AppColors.cardBackgroundDark)
            )
        }
    }
    
    // MARK: - 免责声明
    
    private var disclaimerCard: some View {
        CardView {
            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                sectionHeader(icon: "exclamationmark.shield", title: "免责声明")
                
                Text("本应用按「原样」提供。作者或版权持有人不对因使用本软件而产生的任何索赔、损害或其他责任承担责任。")
                    .font(AppFonts.caption)
                    .foregroundColor(AppColors.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
    
    // MARK: - 版本信息
    
    private var versionInfo: some View {
        VStack(spacing: AppSpacing.xs) {
            Text("版本 \(appVersion)")
                .font(AppFonts.caption)
                .foregroundColor(AppColors.secondary)
            
            Text("基于 Stray Robots Scanner 开发")
                .font(AppFonts.caption2)
                .foregroundColor(AppColors.secondary.opacity(0.7))
        }
        .padding(.vertical, AppSpacing.md)
    }
    
    private var appVersion: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
    }
    
    // MARK: - Helper Views
    
    private func sectionHeader(icon: String, title: String) -> some View {
        HStack(spacing: AppSpacing.sm) {
            Image(systemName: icon)
                .foregroundColor(AppColors.accent)
            
            Text(title)
                .font(AppFonts.headline)
                .foregroundColor(AppColors.primary)
        }
    }
}

struct InformationView_Previews: PreviewProvider {
    static var previews: some View {
        InformationView()
    }
}
