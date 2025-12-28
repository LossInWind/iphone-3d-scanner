//
//  TransferService.swift
//  StrayScanner
//
//  WiFi 数据传输服务
//

import Foundation

/// 传输错误类型
enum TransferError: Error, LocalizedError {
    case invalidURL
    case serverUnreachable
    case uploadFailed(String)
    case noFilesToUpload
    case cancelled
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "无效的服务器地址"
        case .serverUnreachable:
            return "无法连接到服务器"
        case .uploadFailed(let message):
            return "上传失败: \(message)"
        case .noFilesToUpload:
            return "没有可上传的文件"
        case .cancelled:
            return "传输已取消"
        }
    }
}

/// 传输进度
struct TransferProgress {
    let currentFile: Int
    let totalFiles: Int
    let currentFileName: String
    let bytesUploaded: Int64
    let totalBytes: Int64
    
    var percentage: Double {
        guard totalFiles > 0 else { return 0 }
        return Double(currentFile) / Double(totalFiles) * 100
    }
    
    var description: String {
        return "\(currentFile)/\(totalFiles) - \(currentFileName)"
    }
}

/// WiFi 传输服务
class TransferService {
    
    /// 服务器地址 (格式: "192.168.1.100:8080")
    var serverAddress: String {
        didSet {
            // 保存到 UserDefaults
            UserDefaults.standard.set(serverAddress, forKey: "TransferServerAddress")
        }
    }
    
    /// 是否正在传输
    private(set) var isTransferring: Bool = false
    
    /// 取消标志
    private var isCancelled: Bool = false
    
    /// URL Session
    private let session: URLSession
    
    /// 初始化
    init() {
        // 从 UserDefaults 读取上次使用的地址
        self.serverAddress = UserDefaults.standard.string(forKey: "TransferServerAddress") ?? ""
        
        // 配置 URLSession
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        self.session = URLSession(configuration: config)
    }
    
    /// 测试服务器连接
    func testConnection(completion: @escaping (Result<Void, Error>) -> Void) {
        guard let url = buildURL(path: "/status") else {
            completion(.failure(TransferError.invalidURL))
            return
        }
        
        let task = session.dataTask(with: url) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    completion(.failure(TransferError.serverUnreachable))
                    return
                }
                
                completion(.success(()))
            }
        }
        task.resume()
    }
    
    /// 上传整个数据集
    func uploadDataset(
        datasetURL: URL,
        progress: @escaping (TransferProgress) -> Void,
        completion: @escaping (Result<Void, Error>) -> Void
    ) {
        guard !isTransferring else {
            completion(.failure(TransferError.uploadFailed("已有传输正在进行")))
            return
        }
        
        isTransferring = true
        isCancelled = false
        
        // 在后台线程执行
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                // 收集所有需要上传的文件
                let files = try self.collectFiles(in: datasetURL)
                
                if files.isEmpty {
                    throw TransferError.noFilesToUpload
                }
                
                // 生成数据集 ID
                let datasetId = UUID().uuidString
                
                // 计算总大小
                var totalBytes: Int64 = 0
                for file in files {
                    let attrs = try FileManager.default.attributesOfItem(atPath: file.path)
                    totalBytes += (attrs[.size] as? Int64) ?? 0
                }
                
                var uploadedBytes: Int64 = 0
                
                // 标准化数据集路径（移除 /private 前缀）
                let normalizedDatasetPath = datasetURL.standardizedFileURL.path
                
                // 逐个上传文件
                for (index, fileURL) in files.enumerated() {
                    if self.isCancelled {
                        throw TransferError.cancelled
                    }
                    
                    // 计算相对路径（标准化后再计算）
                    let normalizedFilePath = fileURL.standardizedFileURL.path
                    var relativePath = normalizedFilePath.replacingOccurrences(
                        of: normalizedDatasetPath + "/",
                        with: ""
                    )
                    
                    // 如果还是绝对路径，尝试只取文件名部分
                    if relativePath.hasPrefix("/") {
                        // 从路径中提取 depth/xxx.png 或 confidence/xxx.png 等
                        let components = normalizedFilePath.components(separatedBy: "/")
                        if components.count >= 2 {
                            let lastTwo = components.suffix(2)
                            relativePath = lastTwo.joined(separator: "/")
                        } else {
                            relativePath = fileURL.lastPathComponent
                        }
                    }
                    
                    // 获取文件大小
                    let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                    let fileSize = (attrs[.size] as? Int64) ?? 0
                    
                    // 更新进度
                    let currentProgress = TransferProgress(
                        currentFile: index + 1,
                        totalFiles: files.count,
                        currentFileName: relativePath,
                        bytesUploaded: uploadedBytes,
                        totalBytes: totalBytes
                    )
                    DispatchQueue.main.async {
                        progress(currentProgress)
                    }
                    
                    // 上传文件
                    try self.uploadFileSync(
                        localURL: fileURL,
                        remotePath: relativePath,
                        datasetId: datasetId
                    )
                    
                    uploadedBytes += fileSize
                }
                
                // 发送完成信号
                try self.sendCompleteSignalSync(datasetId: datasetId)
                
                self.isTransferring = false
                DispatchQueue.main.async {
                    completion(.success(()))
                }
                
            } catch {
                self.isTransferring = false
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    /// 取消传输
    func cancelTransfer() {
        isCancelled = true
    }
    
    // MARK: - Private Methods
    
    /// 构建 URL
    private func buildURL(path: String, queryItems: [URLQueryItem]? = nil) -> URL? {
        guard !serverAddress.isEmpty else { return nil }
        
        var components = URLComponents()
        components.scheme = "http"
        
        // 解析地址和端口
        let parts = serverAddress.split(separator: ":")
        if parts.count == 2 {
            components.host = String(parts[0])
            components.port = Int(parts[1])
        } else {
            components.host = serverAddress
            components.port = 8080
        }
        
        components.path = path
        components.queryItems = queryItems
        
        return components.url
    }
    
    /// 收集数据集中的所有文件
    private func collectFiles(in directory: URL) throws -> [URL] {
        var files: [URL] = []
        let fileManager = FileManager.default
        
        // 需要上传的文件和目录
        let itemsToUpload = [
            "rgb.mp4",
            "depth",
            "confidence",
            "odometry.csv",
            "imu.csv",
            "camera_matrix.csv"
        ]
        
        for item in itemsToUpload {
            let itemURL = directory.appendingPathComponent(item)
            
            if !fileManager.fileExists(atPath: itemURL.path) {
                continue
            }
            
            var isDirectory: ObjCBool = false
            fileManager.fileExists(atPath: itemURL.path, isDirectory: &isDirectory)
            
            if isDirectory.boolValue {
                // 递归收集目录中的文件
                let contents = try fileManager.contentsOfDirectory(
                    at: itemURL,
                    includingPropertiesForKeys: nil
                )
                files.append(contentsOf: contents)
            } else {
                files.append(itemURL)
            }
        }
        
        return files
    }
    
    /// 同步上传单个文件
    private func uploadFileSync(localURL: URL, remotePath: String, datasetId: String) throws {
        guard let url = buildURL(
            path: "/upload",
            queryItems: [
                URLQueryItem(name: "path", value: remotePath),
                URLQueryItem(name: "dataset", value: datasetId)
            ]
        ) else {
            throw TransferError.invalidURL
        }
        
        // 读取文件数据
        let fileData = try Data(contentsOf: localURL)
        
        // 创建请求
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.setValue("\(fileData.count)", forHTTPHeaderField: "Content-Length")
        
        // 使用信号量实现同步
        let semaphore = DispatchSemaphore(value: 0)
        var uploadError: Error?
        
        let task = session.uploadTask(with: request, from: fileData) { data, response, error in
            defer { semaphore.signal() }
            
            if let error = error {
                uploadError = error
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse else {
                uploadError = TransferError.uploadFailed("无效的响应")
                return
            }
            
            if httpResponse.statusCode != 200 {
                // 尝试解析错误信息
                var errorMessage = "HTTP \(httpResponse.statusCode)"
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let message = json["error"] as? String {
                    errorMessage = message
                }
                uploadError = TransferError.uploadFailed(errorMessage)
            }
        }
        task.resume()
        
        // 等待完成
        semaphore.wait()
        
        if let error = uploadError {
            throw error
        }
    }
    
    /// 发送完成信号
    private func sendCompleteSignalSync(datasetId: String) throws {
        guard let url = buildURL(
            path: "/complete",
            queryItems: [
                URLQueryItem(name: "dataset", value: datasetId)
            ]
        ) else {
            throw TransferError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let semaphore = DispatchSemaphore(value: 0)
        var uploadError: Error?
        
        let task = session.dataTask(with: request) { data, response, error in
            defer { semaphore.signal() }
            
            if let error = error {
                uploadError = error
                return
            }
            
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                uploadError = TransferError.uploadFailed("完成信号发送失败")
                return
            }
        }
        task.resume()
        
        semaphore.wait()
        
        if let error = uploadError {
            throw error
        }
    }
}
