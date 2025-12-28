//
//  DatasetEncoder.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 1/2/21.
//  Copyright © 2021 Stray Robots. All rights reserved.
//

import Foundation
import ARKit
import CryptoKit
import CoreMotion

class DatasetEncoder {
    enum Status {
        case allGood
        case videoEncodingError
        case directoryCreationError
    }
    private let rgbEncoder: VideoEncoder
    private let depthEncoder: DepthEncoder
    private let confidenceEncoder: ConfidenceEncoder
    private let datasetDirectory: URL
    private let odometryEncoder: OdometryEncoder
    private let imuEncoder: IMUEncoder
    private var lastCameraIntrinsics: simd_float3x3?
    private var currentFrame: Int = -1
    private var savedFrames: Int = 0
    private let frameInterval: Int // Only save every frameInterval-th frame.
    public let id: UUID
    public let rgbFilePath: URL // Relative to app document directory.
    public let depthFilePath: URL // Relative to app document directory.
    public let cameraMatrixPath: URL
    public let odometryPath: URL
    public let imuPath: URL
    public var status = Status.allGood
    private let queue: DispatchQueue
    
    // 队列深度控制 - 防止 ARFrame 积压
    private var pendingFrames: Int = 0
    private let maxPendingFrames: Int = 2  // 降低到 2 帧，更激进地控制积压
    private let pendingLock = NSLock()
    
    // 帧跳过计数器 - 用于监控
    private var skippedFrames: Int = 0
    
    private var latestAccelerometerData: (timestamp: Double, data: simd_double3)?
    private var latestGyroscopeData: (timestamp: Double, data: simd_double3)?


    init(arConfiguration: ARWorldTrackingConfiguration, fpsDivider: Int = 1) {
        self.frameInterval = fpsDivider
        // 使用高优先级队列加速编码
        self.queue = DispatchQueue(label: "encoderQueue", qos: .userInitiated)
        
        let width = arConfiguration.videoFormat.imageResolution.width
        let height = arConfiguration.videoFormat.imageResolution.height
        var theId: UUID = UUID()
        datasetDirectory = DatasetEncoder.createDirectory(id: &theId)
        self.id = theId
        self.rgbFilePath = datasetDirectory.appendingPathComponent("rgb.mp4")
        self.rgbEncoder = VideoEncoder(file: self.rgbFilePath, width: width, height: height)
        self.depthFilePath = datasetDirectory.appendingPathComponent("depth", isDirectory: true)
        self.depthEncoder = DepthEncoder(outDirectory: self.depthFilePath)
        let confidenceFilePath = datasetDirectory.appendingPathComponent("confidence", isDirectory: true)
        self.confidenceEncoder = ConfidenceEncoder(outDirectory: confidenceFilePath)
        self.cameraMatrixPath = datasetDirectory.appendingPathComponent("camera_matrix.csv", isDirectory: false)
        self.odometryPath = datasetDirectory.appendingPathComponent("odometry.csv", isDirectory: false)
        self.odometryEncoder = OdometryEncoder(url: self.odometryPath)
        self.imuPath = datasetDirectory.appendingPathComponent("imu.csv", isDirectory: false)
        self.imuEncoder = IMUEncoder(url: self.imuPath)
    }

    func add(frame: ARFrame) {
        let totalFrames: Int = currentFrame
        currentFrame = currentFrame + 1
        if (currentFrame % frameInterval != 0) {
            return
        }
        
        // 检查队列深度，如果积压过多则丢弃帧
        pendingLock.lock()
        let currentPending = pendingFrames
        if currentPending >= maxPendingFrames {
            pendingLock.unlock()
            skippedFrames += 1
            // 队列已满，丢弃此帧以防止 ARFrame 积压
            return
        }
        pendingFrames += 1
        pendingLock.unlock()
        
        // 只有当深度数据存在时才保存帧，保证帧号连续
        guard let sceneDepth = frame.sceneDepth else {
            pendingLock.lock()
            pendingFrames -= 1
            pendingLock.unlock()
            return
        }
        
        let frameNumber: Int = savedFrames
        savedFrames = savedFrames + 1
        
        // 提取基本数据（这些是值类型，不会持有 ARFrame）
        let timestamp = frame.timestamp
        let cameraTransform = frame.camera.transform
        let cameraIntrinsics = frame.camera.intrinsics
        
        // 保存最后一帧的相机内参
        self.lastCameraIntrinsics = cameraIntrinsics
        
        // 在主线程上快速复制 CVPixelBuffer 数据
        // 使用 autoreleasepool 确保临时对象被及时释放
        var depthMapCopy: CVPixelBuffer?
        var capturedImageCopy: CVPixelBuffer?
        var confidenceMapCopy: CVPixelBuffer?
        
        // 使用 autoreleasepool 包裹复制操作
        autoreleasepool {
            // 优先复制较小的深度图
            depthMapCopy = copyPixelBufferFast(sceneDepth.depthMap)
            // 然后复制 RGB 图像
            capturedImageCopy = copyPixelBufferFast(frame.capturedImage)
            // 置信度图是可选的
            confidenceMapCopy = sceneDepth.confidenceMap.flatMap { copyPixelBufferFast($0) }
        }
        
        guard let depthCopy = depthMapCopy, let imageCopy = capturedImageCopy else {
            savedFrames -= 1
            pendingLock.lock()
            pendingFrames -= 1
            pendingLock.unlock()
            return
        }
        
        // 现在 ARFrame 可以被释放了，因为我们已经复制了所有需要的数据
        // 使用异步队列处理编码（耗时操作）
        queue.async { [weak self] in
            guard let self = self else { return }
            
            autoreleasepool {
                // 执行编码（使用复制后的数据）
                self.depthEncoder.encodeFrame(frame: depthCopy, frameNumber: frameNumber)
                if let confidence = confidenceMapCopy {
                    self.confidenceEncoder.encodeFrame(frame: confidence, frameNumber: frameNumber)
                }
                self.rgbEncoder.add(frame: VideoEncoderInput(buffer: imageCopy, time: timestamp), currentFrame: totalFrames)
                self.odometryEncoder.addPose(transform: cameraTransform, timestamp: timestamp, frameNumber: frameNumber)
            }
            
            // 编码完成，减少计数
            self.pendingLock.lock()
            self.pendingFrames -= 1
            self.pendingLock.unlock()
        }
    }
    
    /// 快速复制 CVPixelBuffer - 优化版本
    private func copyPixelBufferFast(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        var newPixelBuffer: CVPixelBuffer?
        
        // 创建带有 IOSurface 属性的 buffer，可能更快
        let options: [String: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            options as CFDictionary,
            &newPixelBuffer
        )
        
        guard status == kCVReturnSuccess, let destBuffer = newPixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(destBuffer, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            CVPixelBufferUnlockBaseAddress(destBuffer, [])
        }
        
        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
        
        if planeCount > 0 {
            // 多平面格式 (YUV)
            for plane in 0..<planeCount {
                guard let srcAddr = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane),
                      let destAddr = CVPixelBufferGetBaseAddressOfPlane(destBuffer, plane) else {
                    continue
                }
                let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane)
                let planeHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
                memcpy(destAddr, srcAddr, bytesPerRow * planeHeight)
            }
        } else {
            // 单平面格式
            guard let srcAddr = CVPixelBufferGetBaseAddress(pixelBuffer),
                  let destAddr = CVPixelBufferGetBaseAddress(destBuffer) else {
                return nil
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            memcpy(destAddr, srcAddr, bytesPerRow * height)
        }
        
        return destBuffer
    }
    
   func addRawAccelerometer(data: CMAccelerometerData) {
        let acceleration = simd_double3(data.acceleration.x, data.acceleration.y, data.acceleration.z)
        latestAccelerometerData = (timestamp: data.timestamp, data: acceleration)
        tryWritingIMUData()
    }

    func addRawGyroscope(data: CMGyroData) {
        let rotationRate = simd_double3(data.rotationRate.x, data.rotationRate.y, data.rotationRate.z)
        latestGyroscopeData = (timestamp: data.timestamp, data: rotationRate)
        tryWritingIMUData()
    }

    private func tryWritingIMUData() {
        guard
            let accelerometer = latestAccelerometerData,
            let gyroscope = latestGyroscopeData
        else {
            return
        }

        // Write the row to the CSV with the most recent timestamp
        let timestamp = max(accelerometer.timestamp, gyroscope.timestamp)
        imuEncoder.add(
            timestamp: timestamp,
            linear: accelerometer.data,
            angular: gyroscope.data
        )

        // Clear the buffers after writing
        latestAccelerometerData = nil
        latestGyroscopeData = nil
    }

    func wrapUp() {
        // 等待所有异步任务完成
        queue.sync {}
        
        // 确保所有帧都处理完毕
        while true {
            pendingLock.lock()
            let remaining = pendingFrames
            pendingLock.unlock()
            if remaining == 0 { break }
            Thread.sleep(forTimeInterval: 0.01)
        }
        
        self.rgbEncoder.finishEncoding()
        self.imuEncoder.done()
        self.odometryEncoder.done()
        writeIntrinsics()
        switch self.rgbEncoder.status {
            case .allGood:
                status = .allGood
            case .error:
                status = .videoEncodingError
        }
        switch self.depthEncoder.status {
            case .allGood:
                status = .allGood
            case .frameEncodingError:
                status = .videoEncodingError
                print("Something went wrong encoding depth.")
        }
        switch self.confidenceEncoder.status {
            case .allGood:
                status = .allGood
            case .encodingError:
                status = .videoEncodingError
                print("Something went wrong encoding confidence values.")
        }
    }

    private func writeIntrinsics() {
        if let cameraMatrix = lastCameraIntrinsics {
            let rows = cameraMatrix.transpose.columns
            var csv: [String] = []
            for row in [rows.0, rows.1, rows.2] {
                let csvLine = "\(row.x), \(row.y), \(row.z)"
                csv.append(csvLine)
            }
            let contents = csv.joined(separator: "\n")
            do {
                try contents.write(to: self.cameraMatrixPath, atomically: true, encoding: String.Encoding.utf8)
            } catch let error {
                print("Could not write camera matrix. \(error.localizedDescription)")
            }
        }
    }

    static private func createDirectory(id: inout UUID) -> URL {
        let directoryId = hashUUID(id: id)
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        var directory = URL(fileURLWithPath: directoryId, relativeTo: url)
        if FileManager.default.fileExists(atPath: directory.absoluteString) {
            // Just in case the first 5 characters clash, try again.
            id = UUID()
            directory = DatasetEncoder.createDirectory(id: &id)
        }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        } catch let error as NSError {
            print("Error creating directory. \(error), \(error.userInfo)")
        }
        return directory
    }

    static private func hashUUID(id: UUID) -> String {
        var hasher: SHA256 = SHA256()
        hasher.update(data: id.uuidString.data(using: .ascii)!)
        let digest = hasher.finalize()
        var string = ""
        digest.makeIterator().prefix(5).forEach { (byte: UInt8) in
            string += String(format: "%02x", byte)
        }
        print("Hash: \(string)")
        return string
    }
}
