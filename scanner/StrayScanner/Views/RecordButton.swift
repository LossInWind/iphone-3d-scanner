//
//  RecordButton.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 12/27/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import Foundation
import UIKit

@IBDesignable
class RecordButton : UIView {
    private let circleStroke: CGFloat = 4.0
    private let animationDuration = 0.15
    private var recording: Bool = false
    private var disk: CALayer!
    private var pulseLayer: CAShapeLayer?
    private var callback: Optional<(Bool) -> Void> = Optional.none

    override public init(frame: CGRect) {
        super.init(frame: frame)
        self.backgroundColor = UIColor.clear
    }

    override func layoutSubviews() {
        setup()
    }

    required public init?(coder aCoder: NSCoder) {
        super.init(coder: aCoder)
        self.backgroundColor = UIColor.clear
    }

    func setCallback(callback: @escaping (Bool) -> Void) {
        self.callback = Optional.some(callback)
    }

    @objc func buttonPressed() {
        // 触觉反馈
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
        
        self.animateButton()
        self.recording = !self.recording
        self.callback?(self.recording)
        
        // 录制状态的脉冲动画
        if self.recording {
            startPulseAnimation()
        } else {
            stopPulseAnimation()
        }
    }

    private func setup() {
        // 清除旧的图层
        layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        drawShadow()
        drawEdge()
        drawInner()
        self.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(buttonPressed)))
    }
    
    private func drawShadow() {
        // 添加阴影效果
        self.layer.shadowColor = UIColor.red.cgColor
        self.layer.shadowOffset = CGSize(width: 0, height: 4)
        self.layer.shadowRadius = 8
        self.layer.shadowOpacity = 0.3
    }

    private func drawEdge() {
        let circleLayer = CAShapeLayer()
        let circleRadius: CGFloat = self.bounds.height * 0.5 - circleStroke
        let x: CGFloat = self.bounds.size.width / 2
        let y: CGFloat = self.bounds.size.height / 2
        let path = CGMutablePath()
        path.addArc(center: CGPoint(x: x, y: y), radius: circleRadius, startAngle: CGFloat(0), endAngle: CGFloat(Float.pi * 2.0), clockwise: false)
        circleLayer.path = path
        circleLayer.fillColor = UIColor.clear.cgColor
        circleLayer.strokeColor = UIColor.white.withAlphaComponent(0.9).cgColor
        circleLayer.lineWidth = circleStroke
        circleLayer.opacity = 1.0
        self.layer.addSublayer(circleLayer)
    }

    private func drawInner() {
        disk = CALayer()
        let diameter = self.bounds.height - circleStroke * 5.0
        let radius = diameter * 0.5
        let x: CGFloat = self.bounds.width * 0.5
        let y: CGFloat = self.bounds.height * 0.5
        let rect = CGRect(x: 0, y: 0, width: diameter, height: diameter)
        
        // 渐变效果
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [
            UIColor(red: 1.0, green: 0.3, blue: 0.3, alpha: 1.0).cgColor,
            UIColor.red.cgColor
        ]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.frame = rect
        gradientLayer.cornerRadius = radius
        
        disk.addSublayer(gradientLayer)
        disk.position.x = x
        disk.position.y = y
        disk.bounds = rect
        disk.cornerRadius = radius
        disk.masksToBounds = true
        disk.opacity = 1.0
        disk.transform = CATransform3DIdentity
        self.layer.addSublayer(disk)
    }
    
    private func startPulseAnimation() {
        // 创建脉冲图层
        let pulseLayer = CAShapeLayer()
        let circleRadius: CGFloat = self.bounds.height * 0.5
        let x: CGFloat = self.bounds.size.width / 2
        let y: CGFloat = self.bounds.size.height / 2
        let path = CGMutablePath()
        path.addArc(center: CGPoint(x: x, y: y), radius: circleRadius, startAngle: 0, endAngle: CGFloat.pi * 2, clockwise: false)
        pulseLayer.path = path
        pulseLayer.fillColor = UIColor.clear.cgColor
        pulseLayer.strokeColor = UIColor.red.withAlphaComponent(0.5).cgColor
        pulseLayer.lineWidth = 2
        pulseLayer.opacity = 0
        
        self.layer.insertSublayer(pulseLayer, at: 0)
        self.pulseLayer = pulseLayer
        
        // 脉冲动画
        let scaleAnimation = CABasicAnimation(keyPath: "transform.scale")
        scaleAnimation.fromValue = 1.0
        scaleAnimation.toValue = 1.3
        
        let opacityAnimation = CABasicAnimation(keyPath: "opacity")
        opacityAnimation.fromValue = 0.8
        opacityAnimation.toValue = 0.0
        
        let animationGroup = CAAnimationGroup()
        animationGroup.animations = [scaleAnimation, opacityAnimation]
        animationGroup.duration = 1.0
        animationGroup.repeatCount = .infinity
        animationGroup.timingFunction = CAMediaTimingFunction(name: .easeOut)
        
        pulseLayer.add(animationGroup, forKey: "pulse")
    }
    
    private func stopPulseAnimation() {
        pulseLayer?.removeAllAnimations()
        pulseLayer?.removeFromSuperlayer()
        pulseLayer = nil
    }

    private func animateButton() {
        // Called before the flag is flipped.
        if self.recording {
            // Finished recording.
            self.animateToIdle()
        } else {
            self.animateToRecording()
        }
    }

    private func animateToRecording() {
        // 圆角动画
        let squaringAnimation = CABasicAnimation(keyPath: "cornerRadius")
        squaringAnimation.fromValue = cornerRadius(recording: false)
        squaringAnimation.toValue = cornerRadius(recording: true)
        squaringAnimation.isRemovedOnCompletion = false
        squaringAnimation.fillMode = CAMediaTimingFillMode.forwards

        // 缩放动画
        let scaleAnimation = CABasicAnimation(keyPath: "transform.scale")
        scaleAnimation.fromValue = 1.0
        scaleAnimation.toValue = 0.45
        scaleAnimation.isRemovedOnCompletion = false
        scaleAnimation.fillMode = CAMediaTimingFillMode.forwards

        let animationGroup = CAAnimationGroup()
        animationGroup.animations = [squaringAnimation, scaleAnimation]
        animationGroup.isRemovedOnCompletion = false
        animationGroup.fillMode = CAMediaTimingFillMode.forwards
        animationGroup.duration = animationDuration
        animationGroup.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        disk.add(animationGroup, forKey: "toRecording")
        
        // 更新阴影
        UIView.animate(withDuration: animationDuration) {
            self.layer.shadowOpacity = 0.5
            self.layer.shadowRadius = 12
        }
    }

    private func animateToIdle() {
        // 圆角动画
        let roundingAnimation = CABasicAnimation(keyPath: "cornerRadius")
        roundingAnimation.fromValue = cornerRadius(recording: true)
        roundingAnimation.toValue = cornerRadius(recording: false)
        roundingAnimation.fillMode = CAMediaTimingFillMode.forwards
        roundingAnimation.isRemovedOnCompletion = false

        // 缩放动画
        let scaleAnimation = CABasicAnimation(keyPath: "transform.scale")
        scaleAnimation.fromValue = 0.45
        scaleAnimation.toValue = 1.0
        scaleAnimation.isRemovedOnCompletion = false
        scaleAnimation.fillMode = CAMediaTimingFillMode.forwards

        let animationGroup = CAAnimationGroup()
        animationGroup.animations = [roundingAnimation, scaleAnimation]
        animationGroup.isRemovedOnCompletion = false
        animationGroup.fillMode = CAMediaTimingFillMode.forwards
        animationGroup.duration = animationDuration
        animationGroup.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        disk.add(animationGroup, forKey: "toIdle")
        
        // 更新阴影
        UIView.animate(withDuration: animationDuration) {
            self.layer.shadowOpacity = 0.3
            self.layer.shadowRadius = 8
        }
    }

    @objc override func prepareForInterfaceBuilder() {
        super.prepareForInterfaceBuilder()
        drawEdge()
        drawInner()
        self.backgroundColor = UIColor.clear
    }

    private func cornerRadius(recording: Bool) -> CGFloat {
        if recording {
            return 8.0
        } else {
            return (self.bounds.height - circleStroke * 5.0) * 0.5
        }
    }
}
