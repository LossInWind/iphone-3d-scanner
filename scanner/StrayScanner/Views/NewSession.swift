//
//  NewSession.swift
//  Stray Scanner
//
//  Created by Kenneth Blomqvist on 11/28/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import SwiftUI

struct NavigationConfigurator: UIViewControllerRepresentable {
    var configure: (UINavigationController) -> Void = { _ in }

    func makeUIViewController(context: UIViewControllerRepresentableContext<NavigationConfigurator>) -> UIViewController {
        UIViewController()
    }
    func updateUIViewController(_ uiViewController: UIViewController, context: UIViewControllerRepresentableContext<NavigationConfigurator>) {
        if let nc = uiViewController.navigationController {
            self.configure(nc)
        }
    }
}

struct RecordSessionManager: UIViewControllerRepresentable {
    @Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    
    func makeUIViewController(context: Context) -> some UIViewController {
        let viewController = RecordSessionViewController(nibName: "RecordSessionView", bundle: nil)
        viewController.setDismissFunction {
            presentationMode.wrappedValue.dismiss()
            viewController.setDismissFunction(Optional.none)
        }
        return viewController
    }

    func updateUIViewController(_ uiViewController: UIViewControllerType, context: Context) {}
}

struct NewSessionView : View {
    
    var body: some View {
        RecordSessionManager()
            .padding(.vertical, 0.0)
            .navigationBarTitle("录制")
            .navigationBarTitleDisplayMode(.inline)
            .edgesIgnoringSafeArea(.all)
            .background(NavigationConfigurator { nc in
                // 设置导航栏样式
                let appearance = UINavigationBarAppearance()
                appearance.configureWithOpaqueBackground()
                appearance.backgroundColor = UIColor(named: "BackgroundColor")?.withAlphaComponent(0.9)
                appearance.titleTextAttributes = [
                    .foregroundColor: UIColor(named: "TextColor") ?? .white,
                    .font: UIFont.systemFont(ofSize: 17, weight: .semibold)
                ]
                
                nc.navigationBar.standardAppearance = appearance
                nc.navigationBar.scrollEdgeAppearance = appearance
                nc.navigationBar.compactAppearance = appearance
                nc.navigationBar.tintColor = UIColor(named: "TextColor")
            })
    }
}

struct NewSessionView_Previews: PreviewProvider {
    static var previews: some View {
        NewSessionView()
    }
}
