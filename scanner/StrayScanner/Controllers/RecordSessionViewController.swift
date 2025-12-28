//
//  RecordSessionViewController.swift
//  Stray Scanner
//
//  Created by Kenneth Blomqvist on 11/28/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//
//  UI Redesigned - Modern + Professional Style
//

import Foundation
import UIKit
import Metal
import ARKit
import CoreData
import CoreMotion

let FpsDividers: [Int] = [1, 2, 4, 12, 60]
let AvailableFpsSettings: [Int] = FpsDividers.map { Int(60 / $0) }
let FpsUserDefaultsKey: String = "FPS"

class MetalView : UIView {
    override class var layerClass: AnyClass {
        get {
            return CAMetalLayer.self
        }
    }
    override var layer: CAMetalLayer {
        return super.layer as! CAMetalLayer
    }
}

class RecordSessionViewController : UIViewController, ARSessionDelegate {
    private var unsupported: Bool = false
    private var arConfiguration: ARWorldTrackingConfiguration?
    private let session = ARSession()
    private let motionManager = CMMotionManager()
    private var renderer: CameraRenderer?
    private var updateLabelTimer: Timer?
    private var startedRecording: Date?
    private var dataContext: NSManagedObjectContext!
    private var datasetEncoder: DatasetEncoder?
    private let imuOperationQueue = OperationQueue()
    private var chosenFpsSetting: Int = 0
    private var frameCount: Int = 0
    
    // 帧处理队列 - 用于异步处理 ARFrame 数据
    private let frameProcessingQueue = DispatchQueue(label: "frameProcessingQueue", qos: .userInitiated)
    
    // 状态指示器
    private var statusIndicator: UIView?
    private var viewModeLabel: UILabel?
    
    @IBOutlet private var rgbView: MetalView!
    @IBOutlet private var depthView: MetalView!
    @IBOutlet private var recordButton: RecordButton!
    @IBOutlet private var timeLabel: UILabel!
    @IBOutlet weak var fpsButton: UIButton!
    var dismissFunction: Optional<() -> Void> = Optional.none
    
    func setDismissFunction(_ fn: Optional<() -> Void>) {
        self.dismissFunction = fn
    }
    override func viewWillAppear(_ animated: Bool) {
        self.chosenFpsSetting = UserDefaults.standard.integer(forKey: FpsUserDefaultsKey)
        updateFpsSetting()
    }

    override func viewDidLoad() {
        guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return }
        self.dataContext = appDelegate.persistentContainer.newBackgroundContext()
        self.renderer = CameraRenderer(rgbLayer: rgbView.layer, depthLayer: depthView.layer)

        depthView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(viewTapped)))
        rgbView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(viewTapped)))
        
        setViewProperties()
        setupStatusIndicator()
        setupViewModeLabel()
        session.delegate = self

        recordButton.setCallback { (recording: Bool) in
            self.toggleRecording(recording)
        }
        
        // 美化 FPS 按钮
        fpsButton.layer.masksToBounds = true
        fpsButton.layer.cornerRadius = 15.0
        fpsButton.backgroundColor = UIColor(named: "DarkColor")?.withAlphaComponent(0.8)
        
        // 美化时间标签
        timeLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 24, weight: .semibold)
        
        imuOperationQueue.qualityOfService = .userInitiated
    }
    
    private func setupStatusIndicator() {
        // 录制状态指示器（红点）
        let indicator = UIView()
        indicator.backgroundColor = .red
        indicator.layer.cornerRadius = 6
        indicator.translatesAutoresizingMaskIntoConstraints = false
        indicator.isHidden = true
        indicator.alpha = 0
        view.addSubview(indicator)
        
        NSLayoutConstraint.activate([
            indicator.widthAnchor.constraint(equalToConstant: 12),
            indicator.heightAnchor.constraint(equalToConstant: 12),
            indicator.trailingAnchor.constraint(equalTo: timeLabel.leadingAnchor, constant: -8),
            indicator.centerYAnchor.constraint(equalTo: timeLabel.centerYAnchor)
        ])
        
        statusIndicator = indicator
    }
    
    private func setupViewModeLabel() {
        // 视图模式标签
        let label = UILabel()
        label.text = "深度"
        label.font = UIFont.systemFont(ofSize: 14, weight: .medium)
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        label.textAlignment = .center
        label.layer.cornerRadius = 12
        label.layer.masksToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(label)
        
        NSLayoutConstraint.activate([
            label.widthAnchor.constraint(equalToConstant: 60),
            label.heightAnchor.constraint(equalToConstant: 24),
            label.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            label.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16)
        ])
        
        viewModeLabel = label
    }

    override func viewDidDisappear(_ animated: Bool) {
        session.pause();
    }

    override func viewWillDisappear(_ animated: Bool) {
        updateLabelTimer?.invalidate()
        datasetEncoder = nil
    }

    override func viewDidAppear(_ animated: Bool) {
        startSession()
    }

    private func startSession() {
        let config = ARWorldTrackingConfiguration()
        arConfiguration = config
        if !ARWorldTrackingConfiguration.isSupported || !ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            print("AR is not supported.")
            unsupported = true
        } else {
            config.frameSemantics.insert(.sceneDepth)
            session.run(config)
        }
    }
    
    private func startRawIMU() {
        if self.motionManager.isAccelerometerAvailable {
            self.motionManager.accelerometerUpdateInterval = 1.0 / 1200.0 // Set update rate
            self.motionManager.startAccelerometerUpdates(to: imuOperationQueue) { (data, error) in
                guard let data = data else {
                    if let error = error {
                        print("Error retrieving accelerometer data: \(error.localizedDescription)")
                    }
                    return
                }
                self.datasetEncoder?.addRawAccelerometer(data: data)
            }
        } else {
            print("Accelerometer not available on this device.")
        }

        if self.motionManager.isGyroAvailable {
            self.motionManager.gyroUpdateInterval = 1.0 / 1200.0 // Set update rate
            self.motionManager.startGyroUpdates(to: imuOperationQueue) { (data, error) in
                guard let data = data else {
                    if let error = error {
                        print("Error retrieving gyroscope data: \(error.localizedDescription)")
                    }
                    return
                }
                self.datasetEncoder?.addRawGyroscope(data: data)
            }
        } else {
            print("Gyroscope not available on this device.")
        }
    }

    private func stopRawIMU() {
        if self.motionManager.isAccelerometerActive {
            self.motionManager.stopAccelerometerUpdates()
            print("Stopped accelerometer updates.")
        }
        if self.motionManager.isGyroActive {
            self.motionManager.stopGyroUpdates()
            print("Stopped gyroscope updates.")
        }
    }
    
    private func toggleRecording(_ recording: Bool) {
        if unsupported {
            showUnsupportedAlert()
            return
        }
        if recording && self.startedRecording == nil {
            startRecording()
        } else if self.startedRecording != nil && !recording {
            stopRecording()
        } else {
            print("This should not happen. We are either not recording and want to stop, or we are recording and want to start.")
        }
    }

    private func startRecording() {
        self.startedRecording = Date()
        self.frameCount = 0
        updateTime()
        updateLabelTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            self.updateTime()
        }
        
        // 显示录制指示器
        showRecordingIndicator()
        
        startRawIMU()
        datasetEncoder = DatasetEncoder(arConfiguration: arConfiguration!, fpsDivider: FpsDividers[chosenFpsSetting])
        startRawIMU()
    }
    
    private func showRecordingIndicator() {
        statusIndicator?.isHidden = false
        UIView.animate(withDuration: 0.3) {
            self.statusIndicator?.alpha = 1
        }
        
        // 闪烁动画
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = 1.0
        animation.toValue = 0.3
        animation.duration = 0.8
        animation.autoreverses = true
        animation.repeatCount = .infinity
        statusIndicator?.layer.add(animation, forKey: "blink")
    }
    
    private func hideRecordingIndicator() {
        statusIndicator?.layer.removeAllAnimations()
        UIView.animate(withDuration: 0.3) {
            self.statusIndicator?.alpha = 0
        } completion: { _ in
            self.statusIndicator?.isHidden = true
        }
    }

    private func stopRecording() {
        guard let started = self.startedRecording else {
            print("Hasn't started recording. Something is wrong.")
            return
        }
        startedRecording = nil
        updateLabelTimer?.invalidate()
        updateLabelTimer = nil
        
        // 隐藏录制指示器
        hideRecordingIndicator()
        
        // Stop IMU updates
        stopRawIMU()
        datasetEncoder?.wrapUp()
        if let encoder = datasetEncoder {
            switch encoder.status {
                case .allGood:
                    saveRecording(started, encoder)
                    showSuccessHaptic()
                case .videoEncodingError:
                    showError()
                case .directoryCreationError:
                    showError()
            }
        } else {
            print("No dataset encoder. Something is wrong.")
        }
        self.dismissFunction?()
    }
    
    private func showSuccessHaptic() {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }

    private func saveRecording(_ started: Date, _ encoder: DatasetEncoder) {
        let sessionCount = countSessions()
        
        let duration = Date().timeIntervalSince(started)
        let entity = NSEntityDescription.entity(forEntityName: "Recording", in: self.dataContext)!
        let recording: Recording = Recording(entity: entity, insertInto: self.dataContext)
        recording.setValue(datasetEncoder!.id, forKey: "id")
        recording.setValue(duration, forKey: "duration")
        recording.setValue(started, forKey: "createdAt")
        recording.setValue("Recording \(sessionCount)", forKey: "name")
        recording.setValue(datasetEncoder!.rgbFilePath.relativeString, forKey: "rgbFilePath")
        recording.setValue(datasetEncoder!.depthFilePath.relativeString, forKey: "depthFilePath")
        do {
            try self.dataContext.save()
        } catch let error as NSError {
            print("Could not save recording. \(error), \(error.userInfo)")
        }
    }

    private func showError() {
        let controller = UIAlertController(title: "Error",
            message: "Something went wrong when encoding video. This should not have happened. You might want to file a bug report.",
            preferredStyle: .alert)
        controller.addAction(UIAlertAction(title: NSLocalizedString("OK", comment: "Default Action"), style: .default, handler: { _ in
            self.dismiss(animated: true, completion: nil)
        }))
        self.present(controller, animated: true, completion: nil)
    }

    private func updateTime() {
        guard let started = self.startedRecording else { return }
        let seconds = Date().timeIntervalSince(started)
        let minutes: Int = Int(floor(seconds / 60).truncatingRemainder(dividingBy: 60))
        let hours: Int = Int(floor(seconds / 3600))
        let roundSeconds: Int = Int(floor(seconds.truncatingRemainder(dividingBy: 60)))
        self.timeLabel.text = String(format: "%02d:%02d:%02d", hours, minutes, roundSeconds)
    }

    @objc func viewTapped() {
        // 触觉反馈
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
        
        switch renderer!.renderMode {
            case .depth:
                renderer!.renderMode = RenderMode.rgb
                rgbView.isHidden = false
                depthView.isHidden = true
                viewModeLabel?.text = "RGB"
            case .rgb:
                renderer!.renderMode = RenderMode.depth
                depthView.isHidden = false
                rgbView.isHidden = true
                viewModeLabel?.text = "深度"
        }
        
        // 标签动画
        UIView.animate(withDuration: 0.15, animations: {
            self.viewModeLabel?.transform = CGAffineTransform(scaleX: 1.1, y: 1.1)
        }) { _ in
            UIView.animate(withDuration: 0.15) {
                self.viewModeLabel?.transform = .identity
            }
        }
    }
    
    @IBAction func fpsButtonTapped() {
        // 触觉反馈
        let generator = UIImpactFeedbackGenerator(style: .light)
        generator.impactOccurred()
        
        chosenFpsSetting = (chosenFpsSetting + 1) % AvailableFpsSettings.count
        updateFpsSetting()
        UserDefaults.standard.set(chosenFpsSetting, forKey: FpsUserDefaultsKey)
        
        // 按钮动画
        UIView.animate(withDuration: 0.1, animations: {
            self.fpsButton.transform = CGAffineTransform(scaleX: 0.95, y: 0.95)
        }) { _ in
            UIView.animate(withDuration: 0.1) {
                self.fpsButton.transform = .identity
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // 渲染当前帧（这是同步的，但很快）
        self.renderer!.render(frame: frame)
        
        // 如果正在录制，立即提取数据并释放 ARFrame
        if startedRecording != nil {
            if let encoder = datasetEncoder {
                // 在主线程上快速提取必要的数据
                // 这样 ARFrame 可以在 delegate 方法返回后立即被释放
                encoder.add(frame: frame)
                frameCount += 1
            } else {
                print("There is no video encoder. That can't be good.")
            }
        }
        // ARFrame 在这里离开作用域，应该被立即释放
    }

    private func setViewProperties() {
        self.view.backgroundColor = UIColor(named: "BackgroundColor")
        
        // 添加圆角到视图
        rgbView.layer.cornerRadius = 12
        rgbView.layer.masksToBounds = true
        depthView.layer.cornerRadius = 12
        depthView.layer.masksToBounds = true
    }
    
    private func updateFpsSetting() {
        let fps = AvailableFpsSettings[chosenFpsSetting]
        let buttonLabel: String = "\(fps) fps"
        fpsButton.setTitle(buttonLabel, for: UIControl.State.normal)
    }
    
    private func showUnsupportedAlert() {
        let alert = UIAlertController(title: "Unsupported device", message: "This device doesn't seem to have the required level of ARKit support.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: { _ in
            self.dismissFunction?()
        }))
        self.present(alert, animated: true)
    }
    
    private func countSessions() -> Int {
        guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return 0 }
        let request = NSFetchRequest<NSManagedObject>(entityName: "Recording")
        do {
            let fetched: [NSManagedObject] = try appDelegate.persistentContainer.viewContext.fetch(request)
            return fetched.count
        } catch let error {
            print("Could not fetch sessions for counting. \(error.localizedDescription)")
        }
        return 0
    }
}
