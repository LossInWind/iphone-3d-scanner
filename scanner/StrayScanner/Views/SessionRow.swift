//
//  SessionRow.swift
//  Stray Scanner
//
//  Created by Kenneth Blomqvist on 11/15/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//  注意：主要的卡片样式已移至 SessionList.swift 中的 SessionRowCard
//  此文件保留用于兼容性
//

import SwiftUI

struct SessionRow: View {
    var session: Recording
    
    var body: some View {
        let duration = String(format: "%ds", Int(round(session.duration)))
        HStack {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text(sessionTitle())
                    .font(AppFonts.headline)
                    .foregroundColor(AppColors.primary)
                    .multilineTextAlignment(.leading)
                
                Text(duration)
                    .font(AppFonts.caption)
                    .foregroundColor(AppColors.secondary)
                    .multilineTextAlignment(.leading)
            }
            Spacer()
        }
        .padding(.vertical, AppSpacing.xs)
    }
    
    private func sessionTitle() -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        
        if let created = session.createdAt {
            return dateFormatter.string(from: created)
        } else {
            return "Session"
        }
    }
}
