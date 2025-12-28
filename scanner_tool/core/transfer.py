#!/usr/bin/env python3
"""
WiFi Transfer Server - 局域网数据传输服务器

用于接收 iOS Scanner App 上传的数据集。
"""

import os
import json
import socket
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any


def get_local_ip() -> str:
    """获取本机局域网 IP 地址"""
    try:
        # 创建一个 UDP socket 连接到外部地址来获取本机 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # 备用方法：获取主机名对应的 IP
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            if ip.startswith("127."):
                # 如果是 localhost，尝试其他方法
                return "0.0.0.0"
            return ip
        except Exception:
            return "0.0.0.0"


class TransferConfig:
    """传输服务器配置"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        output_dir: str = "./datasets"
    ):
        self.host = host
        self.port = port
        self.output_dir = os.path.abspath(output_dir)


class FileUploadHandler(BaseHTTPRequestHandler):
    """处理文件上传请求的 HTTP Handler"""
    
    # 类变量，由 TransferServer 设置
    output_dir: str = "./datasets"
    active_datasets: Dict[str, str] = {}  # dataset_id -> directory path
    
    def log_message(self, format: str, *args) -> None:
        """自定义日志格式"""
        print(f"[{self.log_date_time_string()}] {args[0]}")
    
    def send_json_response(self, status: int, data: Dict[str, Any]) -> None:
        """发送 JSON 响应"""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def do_OPTIONS(self) -> None:
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self) -> None:
        """处理 GET 请求"""
        parsed = urlparse(self.path)
        
        if parsed.path == "/status":
            self.handle_status()
        else:
            self.send_json_response(404, {"error": "Not found"})
    
    def do_POST(self) -> None:
        """处理 POST 请求"""
        parsed = urlparse(self.path)
        
        if parsed.path == "/upload":
            self.handle_upload(parsed)
        elif parsed.path == "/complete":
            self.handle_complete(parsed)
        else:
            self.send_json_response(404, {"error": "Not found"})
    
    def handle_status(self) -> None:
        """处理状态检查请求"""
        self.send_json_response(200, {
            "status": "ready",
            "version": "1.0",
            "output_dir": self.output_dir
        })
    
    def handle_upload(self, parsed) -> None:
        """处理文件上传请求"""
        try:
            # 解析查询参数
            params = parse_qs(parsed.query)
            
            if "path" not in params:
                self.send_json_response(400, {"error": "Missing 'path' parameter"})
                return
            
            if "dataset" not in params:
                self.send_json_response(400, {"error": "Missing 'dataset' parameter"})
                return
            
            relative_path = params["path"][0]
            dataset_id = params["dataset"][0]
            
            # 修复 iOS 发送的绝对路径问题
            # 例如: /privatedepth/000387.png -> depth/000387.png
            if relative_path.startswith("/"):
                # 移除开头的斜杠和 "private" 前缀
                relative_path = relative_path.lstrip("/")
                if relative_path.startswith("private"):
                    relative_path = relative_path[7:]  # 移除 "private"
                # 确保路径格式正确 (depth/xxx.png)
                if not "/" in relative_path and relative_path.endswith(".png"):
                    # 尝试从路径推断目录
                    pass
            
            # 获取或创建数据集目录
            if dataset_id not in self.active_datasets:
                # 使用时间戳创建新目录
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dataset_dir = os.path.join(self.output_dir, f"{timestamp}_{dataset_id[:8]}")
                os.makedirs(dataset_dir, exist_ok=True)
                self.active_datasets[dataset_id] = dataset_dir
                print(f"  Created dataset directory: {dataset_dir}")
            
            dataset_dir = self.active_datasets[dataset_id]
            
            # 构建完整文件路径
            file_path = os.path.join(dataset_dir, relative_path)
            
            # 创建必要的子目录
            file_dir = os.path.dirname(file_path)
            if file_dir and not os.path.exists(file_dir):
                os.makedirs(file_dir, exist_ok=True)
            
            # 读取请求体并保存文件
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self.send_json_response(400, {"error": "Empty file"})
                return
            
            # 分块读取大文件
            with open(file_path, "wb") as f:
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(65536, remaining)  # 64KB chunks
                    chunk = self.rfile.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            
            print(f"  Saved: {relative_path} ({content_length} bytes)")
            self.send_json_response(200, {"status": "ok", "path": relative_path})
            
        except Exception as e:
            print(f"  Error: {e}")
            self.send_json_response(500, {"error": str(e)})
    
    def handle_complete(self, parsed) -> None:
        """处理传输完成请求"""
        try:
            params = parse_qs(parsed.query)
            
            if "dataset" not in params:
                self.send_json_response(400, {"error": "Missing 'dataset' parameter"})
                return
            
            dataset_id = params["dataset"][0]
            
            if dataset_id in self.active_datasets:
                dataset_dir = self.active_datasets[dataset_id]
                del self.active_datasets[dataset_id]
                print(f"  Dataset complete: {dataset_dir}")
                self.send_json_response(200, {
                    "status": "ok",
                    "path": dataset_dir
                })
            else:
                self.send_json_response(400, {"error": "Unknown dataset"})
                
        except Exception as e:
            print(f"  Error: {e}")
            self.send_json_response(500, {"error": str(e)})


class TransferServer:
    """WiFi 传输服务器"""
    
    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
        self.server: Optional[HTTPServer] = None
        self._running = False
    
    def start(self) -> None:
        """启动服务器（阻塞）"""
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 设置 Handler 的类变量
        FileUploadHandler.output_dir = self.config.output_dir
        FileUploadHandler.active_datasets = {}
        
        # 创建服务器
        self.server = HTTPServer(
            (self.config.host, self.config.port),
            FileUploadHandler
        )
        
        # 获取本机 IP
        local_ip = get_local_ip()
        
        print("=" * 50)
        print("Scanner Tool Transfer Server")
        print("=" * 50)
        print(f"Server running at: http://{local_ip}:{self.config.port}")
        print(f"Output directory: {self.config.output_dir}")
        print("")
        print("在 iOS App 中输入以下地址:")
        print(f"  {local_ip}:{self.config.port}")
        print("")
        print("按 Ctrl+C 停止服务器")
        print("=" * 50)
        
        self._running = True
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\n正在停止服务器...")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """停止服务器"""
        if self.server and self._running:
            self.server.shutdown()
            self._running = False
            print("服务器已停止")


def check_transfer_available() -> Dict[str, bool]:
    """检查传输功能可用性"""
    return {
        "available": True,
        "http_server": True
    }


# 便捷函数
def run_server(
    port: int = 8080,
    output_dir: str = "./datasets"
) -> None:
    """运行传输服务器"""
    config = TransferConfig(port=port, output_dir=output_dir)
    server = TransferServer(config)
    server.start()
