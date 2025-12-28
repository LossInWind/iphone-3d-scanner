//
//  Theme.swift
//  StrayScanner
//
//  统一的设计主题和样式定义
//

import SwiftUI

// MARK: - 颜色主题

struct AppColors {
    // 主色调
    static let primary = Color("TextColor")
    static let secondary = Color.gray
    static let accent = Color.blue
    
    // 背景色
    static let background = Color("BackgroundColor")
    static let cardBackground = Color(UIColor.systemGray6).opacity(0.15)
    static let cardBackgroundDark = Color(UIColor.systemGray5).opacity(0.2)
    
    // 状态色
    static let success = Color.green
    static let warning = Color.orange
    static let danger = Color("DangerColor")
    static let recording = Color.red
    
    // 渐变色
    static let primaryGradient = LinearGradient(
        colors: [Color.blue, Color.blue.opacity(0.8)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    
    static let recordingGradient = LinearGradient(
        colors: [Color.red, Color.red.opacity(0.8)],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
}

// MARK: - 字体样式

struct AppFonts {
    static let largeTitle = Font.system(size: 34, weight: .bold, design: .rounded)
    static let title = Font.system(size: 28, weight: .bold, design: .rounded)
    static let title2 = Font.system(size: 22, weight: .semibold, design: .rounded)
    static let title3 = Font.system(size: 20, weight: .semibold, design: .rounded)
    static let headline = Font.system(size: 17, weight: .semibold, design: .rounded)
    static let body = Font.system(size: 17, weight: .regular, design: .default)
    static let callout = Font.system(size: 16, weight: .regular, design: .default)
    static let caption = Font.system(size: 12, weight: .regular, design: .default)
    static let caption2 = Font.system(size: 11, weight: .regular, design: .default)
    
    // 等宽字体（用于数字显示）
    static let mono = Font.system(size: 17, weight: .medium, design: .monospaced)
    static let monoLarge = Font.system(size: 24, weight: .bold, design: .monospaced)
}

// MARK: - 间距

struct AppSpacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 16
    static let lg: CGFloat = 24
    static let xl: CGFloat = 32
}

// MARK: - 圆角

struct AppCorners {
    static let small: CGFloat = 8
    static let medium: CGFloat = 12
    static let large: CGFloat = 16
    static let extraLarge: CGFloat = 24
}

// MARK: - 自定义按钮样式

struct PrimaryButtonStyle: ButtonStyle {
    var isEnabled: Bool = true
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(AppFonts.headline)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, AppSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.medium)
                    .fill(isEnabled ? AppColors.primaryGradient : LinearGradient(colors: [Color.gray], startPoint: .leading, endPoint: .trailing))
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct SecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(AppFonts.headline)
            .foregroundColor(AppColors.primary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, AppSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.medium)
                    .stroke(AppColors.primary, lineWidth: 1.5)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

struct DangerButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(AppFonts.headline)
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, AppSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.medium)
                    .fill(AppColors.danger)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}

// MARK: - 卡片视图

struct CardView<Content: View>: View {
    let content: Content
    var padding: CGFloat = AppSpacing.md
    
    init(padding: CGFloat = AppSpacing.md, @ViewBuilder content: () -> Content) {
        self.padding = padding
        self.content = content()
    }
    
    var body: some View {
        content
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: AppCorners.large)
                    .fill(AppColors.cardBackground)
            )
    }
}

// MARK: - 统计信息项

struct StatItem: View {
    let icon: String
    let title: String
    let value: String
    var valueColor: Color = AppColors.primary
    
    var body: some View {
        VStack(spacing: AppSpacing.xs) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundColor(AppColors.accent)
            
            Text(value)
                .font(AppFonts.headline)
                .foregroundColor(valueColor)
            
            Text(title)
                .font(AppFonts.caption)
                .foregroundColor(AppColors.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - 紧凑型统计信息项

struct MiniStatItem: View {
    let icon: String
    let value: String
    let label: String
    
    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundColor(AppColors.accent)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundColor(AppColors.primary)
                
                Text(label)
                    .font(.system(size: 11))
                    .foregroundColor(AppColors.secondary)
            }
        }
        .padding(.horizontal, AppSpacing.sm)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(AppColors.cardBackground)
        )
    }
}

// MARK: - 全宽统计信息项（自适应宽度）

struct FullWidthStatItem: View {
    let icon: String
    let value: String
    let label: String
    
    var body: some View {
        VStack(spacing: 6) {
            // 图标
            Image(systemName: icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundColor(AppColors.accent)
            
            // 数值
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundColor(AppColors.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            
            // 标签
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(AppColors.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(AppColors.cardBackground)
        )
    }
}

// MARK: - 状态标签

struct StatusBadge: View {
    let text: String
    let color: Color
    
    var body: some View {
        Text(text)
            .font(AppFonts.caption2)
            .fontWeight(.semibold)
            .foregroundColor(.white)
            .padding(.horizontal, AppSpacing.sm)
            .padding(.vertical, AppSpacing.xs)
            .background(
                Capsule()
                    .fill(color)
            )
    }
}

// MARK: - 图标按钮

struct IconButton: View {
    let icon: String
    let action: () -> Void
    var size: CGFloat = 44
    var iconSize: CGFloat = 20
    
    var body: some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: iconSize, weight: .medium))
                .foregroundColor(AppColors.primary)
                .frame(width: size, height: size)
                .background(
                    Circle()
                        .fill(AppColors.cardBackground)
                )
        }
    }
}

// MARK: - 分隔线

struct AppDivider: View {
    var body: some View {
        Rectangle()
            .fill(Color.gray.opacity(0.2))
            .frame(height: 1)
    }
}

// MARK: - 空状态视图

struct EmptyStateView: View {
    let icon: String
    let title: String
    let message: String
    var action: (() -> Void)? = nil
    var actionTitle: String? = nil
    
    var body: some View {
        VStack(spacing: AppSpacing.lg) {
            Image(systemName: icon)
                .font(.system(size: 60))
                .foregroundColor(AppColors.secondary.opacity(0.5))
            
            VStack(spacing: AppSpacing.sm) {
                Text(title)
                    .font(AppFonts.title3)
                    .foregroundColor(AppColors.primary)
                
                Text(message)
                    .font(AppFonts.body)
                    .foregroundColor(AppColors.secondary)
                    .multilineTextAlignment(.center)
            }
            
            if let action = action, let actionTitle = actionTitle {
                Button(action: action) {
                    Text(actionTitle)
                }
                .buttonStyle(PrimaryButtonStyle())
                .padding(.horizontal, AppSpacing.xl)
            }
        }
        .padding(AppSpacing.xl)
    }
}

// MARK: - 预览

struct Theme_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            CardView {
                VStack(alignment: .leading, spacing: 12) {
                    Text("卡片标题")
                        .font(AppFonts.headline)
                    Text("这是卡片内容")
                        .font(AppFonts.body)
                        .foregroundColor(AppColors.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            
            HStack {
                StatItem(icon: "clock", title: "时长", value: "2:30")
                StatItem(icon: "film", title: "帧数", value: "450")
                StatItem(icon: "internaldrive", title: "大小", value: "128MB")
            }
            
            Button("主要按钮") {}
                .buttonStyle(PrimaryButtonStyle())
            
            Button("次要按钮") {}
                .buttonStyle(SecondaryButtonStyle())
            
            StatusBadge(text: "录制中", color: .red)
        }
        .padding()
        .background(Color.black)
    }
}
