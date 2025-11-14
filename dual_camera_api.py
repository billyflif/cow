#!/usr/bin/env python3
"""
Orbbec 双相机同步采集系统 - 支持侧面和顶部相机

主要特性:
- 支持两个相机（侧面和顶部）同步采集
- 相机间时间同步机制
- 分相机存储数据
- RFID触发双相机同步采集
- 独立和联合采集模式

修改说明:
- 修复了设备列表访问的API问题
- 使用正确的pyorbbecsdk API方法
- 优化RFID处理逻辑：只要有新RFID就触发捕获
- 改进实时帧捕获机制
- 增强错误处理和日志记录
- 异步处理RFID触发的捕获任务

作者: Assistant
日期: 2024
"""

import os
import time
import threading
import queue
import json
from datetime import datetime
from enum import Enum
import requests
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFrameType, OBFormat, Context

# 尝试导入OBAlignMode
try:
    from pyorbbecsdk import OBAlignMode
    HAS_ALIGN_MODE = True
except ImportError:
    HAS_ALIGN_MODE = False
    print("OBAlignMode not available in this SDK version")

app = Flask(__name__)
CORS(app)  # 启用CORS


# 相机位置枚举
class CameraPosition(Enum):
    SIDE = "SIDE"
    TOP = "TOP"


# 配置
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 20000  # 20000mm
SAVE_PATH = "./captured_images"  # 基础保存目录
MAX_QUEUE_SIZE = 10  # 减小队列大小以保证实时性

# RFID配置
RFID_URL = "http://cowbodysize.muguanjia.net/rfid/read"
RFID_POLL_INTERVAL = 0.5  # 减小轮询间隔以提高响应速度
RFID_CAPTURE_CONFIG = {
    "num_frames": 50,  # RFID检测时捕获的帧数
    "capture_type": "both"  # 捕获深度和RGB
}

# 确保基础保存目录存在
os.makedirs(SAVE_PATH, exist_ok=True)


class TemporalFilter:
    """深度数据时间滤波器"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class SingleCameraManager:
    """管理单个Orbbec相机的类"""

    def __init__(self, position: CameraPosition, device_index=0):
        self.position = position
        self.device_index = device_index
        self.pipeline = None
        self.config = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.capture_thread = None
        self.temporal_filter = TemporalFilter(alpha=0.5)
        self.depth_scale = 1.0
        self.has_depth = False
        self.has_rgb = False
        self.depth_profile = None
        self.color_profile = None
        self.device = None
        self.frame_counter = 0  # 添加帧计数器

    def initialize(self, context, device_list):
        """初始化相机管道"""
        try:
            # 获取设备数量 - 使用正确的API方法
            device_count = device_list.get_count()
            if self.device_index >= device_count:
                print(f"Device index {self.device_index} out of range. Total devices: {device_count}")
                return False

            # 通过索引获取设备 - 使用正确的API方法
            self.device = device_list.get_device_by_index(self.device_index)
            device_info = self.device.get_device_info()
            print(f"Initializing {self.position.value} camera:")
            print(f"  Name: {device_info.get_name()}")
            print(f"  Serial: {device_info.get_serial_number()}")
            print(f"  Device index: {self.device_index}")

            self.config = Config()
            self.pipeline = Pipeline(self.device)

            # 尝试启用深度流
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                if profile_list is not None:
                    # 尝试不同的方法获取默认profile
                    try:
                        self.depth_profile = profile_list.get_default_video_stream_profile()
                    except AttributeError:
                        # 如果没有get_default_video_stream_profile方法，尝试其他方法
                        profile_count = profile_list.get_count()
                        if profile_count > 0:
                            self.depth_profile = profile_list.get_video_stream_profile(0)
                        else:
                            self.depth_profile = None

                    if self.depth_profile is not None:
                        print(f"  Depth profile: {self.depth_profile}")
                        self.config.enable_stream(self.depth_profile)
                        self.has_depth = True
            except Exception as e:
                print(f"  No depth sensor available: {e}")

            # 尝试启用RGB流
            try:
                profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                if profile_list is not None:
                    # 尝试不同的方法获取默认profile
                    try:
                        self.color_profile = profile_list.get_default_video_stream_profile()
                    except AttributeError:
                        # 如果没有get_default_video_stream_profile方法，尝试其他方法
                        profile_count = profile_list.get_count()
                        if profile_count > 0:
                            self.color_profile = profile_list.get_video_stream_profile(0)
                        else:
                            self.color_profile = None

                    if self.color_profile is not None:
                        print(f"  Color profile: {self.color_profile}")
                        self.config.enable_stream(self.color_profile)
                        self.has_rgb = True
            except Exception as e:
                print(f"  No RGB sensor available: {e}")

            if not self.has_depth and not self.has_rgb:
                raise Exception("No sensors available")

            # 如果深度和RGB都可用，启用帧同步
            if self.has_depth and self.has_rgb:
                try:
                    if HAS_ALIGN_MODE and hasattr(self.config, 'set_align_mode'):
                        self.config.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)
                        print("  Hardware alignment enabled")
                except Exception as e:
                    print(f"  Hardware alignment not available: {e}")

            # 启动管道
            self.pipeline.start(self.config)
            self.is_running = True

            # 启动捕获线程
            self.capture_thread = threading.Thread(target=self._capture_loop, name=f"capture_{self.position.value}")
            self.capture_thread.daemon = True
            self.capture_thread.start()

            print(f"{self.position.value} camera initialized - Depth: {self.has_depth}, RGB: {self.has_rgb}")
            return True

        except Exception as e:
            print(f"Failed to initialize {self.position.value} camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _capture_loop(self):
        """持续从相机捕获帧"""
        while self.is_running:
            try:
                # 等待帧集
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                # 获取系统时间戳
                system_timestamp = time.time()
                self.frame_counter += 1

                # 创建帧数据结构
                frame_data = {
                    'system_timestamp': system_timestamp,
                    'position': self.position.value,
                    'depth_frame': None,
                    'rgb_frame': None,
                    'frame_index': self.frame_counter,
                    'captured_at': datetime.now().isoformat()
                }

                # 处理深度帧
                if self.has_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is not None:
                        width = depth_frame.get_width()
                        height = depth_frame.get_height()
                        self.depth_scale = depth_frame.get_depth_scale()

                        # 获取硬件时间戳
                        hw_timestamp = None
                        if hasattr(depth_frame, 'get_timestamp'):
                            hw_timestamp = depth_frame.get_timestamp()

                        # 处理深度数据
                        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        depth_data = depth_data.reshape((height, width))

                        # 应用缩放和滤波
                        depth_data = depth_data.astype(np.float32) * self.depth_scale
                        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
                        depth_data = depth_data.astype(np.uint16)

                        # 应用时间滤波
                        depth_data = self.temporal_filter.process(depth_data)

                        frame_data['depth_frame'] = {
                            'data': depth_data.copy(),
                            'width': width,
                            'height': height,
                            'scale': self.depth_scale,
                            'hw_timestamp': hw_timestamp
                        }

                # 处理RGB帧
                if self.has_rgb:
                    color_frame = frames.get_color_frame()
                    if color_frame is not None:
                        width = color_frame.get_width()
                        height = color_frame.get_height()

                        # 获取硬件时间戳
                        hw_timestamp = None
                        if hasattr(color_frame, 'get_timestamp'):
                            hw_timestamp = color_frame.get_timestamp()

                        # 获取帧格式
                        format = color_frame.get_format()

                        # 获取原始数据
                        raw_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

                        # 处理不同格式
                        color_data = None
                        if format == OBFormat.MJPG:
                            decoded = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                            if decoded is not None:
                                color_data = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        elif hasattr(OBFormat, 'RGB888') and format == OBFormat.RGB888:
                            try:
                                color_data = raw_data.reshape((height, width, 3))
                            except ValueError:
                                continue
                        elif hasattr(OBFormat, 'RGB') and format == OBFormat.RGB:
                            try:
                                color_data = raw_data.reshape((height, width, 3))
                            except ValueError:
                                continue
                        elif format == OBFormat.BGR:
                            try:
                                color_data = raw_data.reshape((height, width, 3))
                                color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                            except ValueError:
                                continue
                        elif format == OBFormat.YUYV:
                            try:
                                color_data = raw_data.reshape((height, width, 2))
                                color_data = cv2.cvtColor(color_data, cv2.COLOR_YUV2RGB_YUYV)
                            except ValueError:
                                continue
                        else:
                            # 尝试作为JPEG解码
                            try:
                                decoded = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                                if decoded is not None:
                                    color_data = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                            except:
                                continue

                        if color_data is not None:
                            frame_data['rgb_frame'] = {
                                'data': color_data.copy(),
                                'width': width,
                                'height': height,
                                'hw_timestamp': hw_timestamp,
                                'format': format
                            }

                # 只有当至少有一个有效帧时才加入队列
                if frame_data['depth_frame'] is not None or frame_data['rgb_frame'] is not None:
                    # 保持队列较小以确保实时性
                    if self.frame_queue.full():
                        try:
                            # 丢弃旧帧
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame_data)

            except Exception as e:
                print(f"Error in {self.position.value} camera capture loop: {e}")

    def get_latest_frame(self, timeout=1.0):
        """获取最新的帧"""
        try:
            # 清空队列以获取最新帧
            latest_frame = None
            dropped_frames = 0
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                    dropped_frames += 1
                except queue.Empty:
                    break

            if dropped_frames > 1:
                print(f"{self.position.value} camera: Dropped {dropped_frames - 1} old frames to get latest")

            # 如果没有获取到帧，等待新帧
            if latest_frame is None:
                latest_frame = self.frame_queue.get(timeout=timeout)
                print(f"{self.position.value} camera: Waited for new frame")

            return latest_frame
        except queue.Empty:
            print(f"{self.position.value} camera: No frame available within timeout")
            return None

    def flush_queue(self):
        """清空帧队列，确保下次获取的是最新帧"""
        flushed = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        if flushed > 0:
            print(f"{self.position.value} camera: Flushed {flushed} frames from queue")

    def stop(self):
        """停止相机管道"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.pipeline:
            self.pipeline.stop()
        print(f"{self.position.value} camera stopped")


class DualCameraManager:
    """管理双相机系统的类"""

    def __init__(self):
        self.side_camera = None
        self.top_camera = None
        self.context = None
        self.device_list = None
        self.sync_lock = threading.Lock()
        self.is_initialized = False

    def initialize(self):
        """初始化双相机系统"""
        try:
            # 创建上下文
            self.context = Context()
            # 使用query_devices获取设备列表
            self.device_list = self.context.query_devices()

            # 使用正确的API获取设备数量
            device_count = self.device_list.get_count()
            print(f"Found {device_count} devices")

            if device_count < 2:
                print(f"Warning: Only {device_count} camera(s) found. System requires 2 cameras.")
                if device_count == 0:
                    return False

            # 初始化侧面相机（设备0）
            self.side_camera = SingleCameraManager(CameraPosition.SIDE, device_index=0)
            if not self.side_camera.initialize(self.context, self.device_list):
                print("Failed to initialize SIDE camera")
                return False

            # 如果有第二个相机，初始化顶部相机
            if device_count >= 2:
                self.top_camera = SingleCameraManager(CameraPosition.TOP, device_index=1)
                if not self.top_camera.initialize(self.context, self.device_list):
                    print("Failed to initialize TOP camera")
                    # 继续运行，只使用侧面相机
                    self.top_camera = None
            else:
                print("Only one camera available, TOP camera will not be initialized")

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize dual camera system: {e}")
            import traceback
            traceback.print_exc()
            return False

    def capture_frames_sync(self, num_frames, capture_type='both', cameras=['both'], flush_before_capture=True):
        """
        同步捕获多个相机的帧

        Args:
            num_frames: 要捕获的帧数
            capture_type: 'depth', 'rgb', 或 'both'
            cameras: ['side'], ['top'], 或 ['both']
            flush_before_capture: 是否在捕获前清空队列以确保实时性
        """
        if not self.is_initialized:
            return None

        # 确定要使用的相机
        use_side = 'both' in cameras or 'side' in cameras
        use_top = 'both' in cameras or 'top' in cameras

        # 如果需要，先清空队列
        if flush_before_capture:
            print("Flushing camera queues for real-time capture...")
            if use_side and self.side_camera:
                self.side_camera.flush_queue()
            if use_top and self.top_camera:
                self.top_camera.flush_queue()
            # 等待一小段时间让新帧进入
            time.sleep(0.1)

        captured_frames = []
        capture_start_time = time.time()

        for i in range(num_frames):
            frame_start_time = time.time()
            frame_set = {
                'timestamp': frame_start_time,
                'frame_index': i
            }

            # 同步捕获两个相机的帧
            with self.sync_lock:
                # 捕获侧面相机
                if use_side and self.side_camera:
                    side_frame = self.side_camera.get_latest_frame(timeout=2.0)
                    if side_frame:
                        frame_set['side'] = self._process_frame(side_frame, capture_type)
                    else:
                        print(f"Warning: No frame from SIDE camera for frame {i}")

                # 捕获顶部相机
                if use_top and self.top_camera:
                    top_frame = self.top_camera.get_latest_frame(timeout=2.0)
                    if top_frame:
                        frame_set['top'] = self._process_frame(top_frame, capture_type)
                    else:
                        print(f"Warning: No frame from TOP camera for frame {i}")

            captured_frames.append(frame_set)

            # 计算并调整帧间延迟以保持稳定的捕获率
            frame_duration = time.time() - frame_start_time
            target_frame_interval = 0.033  # ~30fps
            if i < num_frames - 1:
                sleep_time = max(0, target_frame_interval - frame_duration)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        total_capture_time = time.time() - capture_start_time
        fps = num_frames / total_capture_time
        print(f"Captured {num_frames} frames in {total_capture_time:.2f}s ({fps:.2f} fps)")

        return captured_frames

    def _process_frame(self, frame_data, capture_type):
        """处理单个相机的帧数据"""
        result = {
            'timestamp': frame_data['system_timestamp'],
            'position': frame_data['position'],
            'captured_at': frame_data.get('captured_at')
        }

        if capture_type in ['depth', 'both'] and frame_data['depth_frame']:
            result['depth'] = frame_data['depth_frame']

        if capture_type in ['rgb', 'both'] and frame_data['rgb_frame']:
            result['rgb'] = frame_data['rgb_frame']

        return result

    def save_frames(self, frames, individual_id=None):
        """保存捕获的帧"""
        saved_paths = []
        save_start_time = time.time()

        for frame_set in frames:
            frame_result = {
                'timestamp': frame_set['timestamp'],
                'frame_index': frame_set['frame_index']
            }

            # 保存侧面相机数据
            if 'side' in frame_set:
                side_paths = self._save_camera_frame(
                    frame_set['side'],
                    CameraPosition.SIDE,
                    individual_id,
                    frame_set['frame_index']
                )
                frame_result['side'] = side_paths

            # 保存顶部相机数据
            if 'top' in frame_set:
                top_paths = self._save_camera_frame(
                    frame_set['top'],
                    CameraPosition.TOP,
                    individual_id,
                    frame_set['frame_index']
                )
                frame_result['top'] = top_paths

            saved_paths.append(frame_result)

        save_duration = time.time() - save_start_time
        print(f"Saved {len(frames)} frame sets in {save_duration:.2f}s")

        return saved_paths

    def _save_camera_frame(self, camera_data, position, individual_id, frame_index):
        """保存单个相机的帧数据"""
        result = {}
        timestamp_str = datetime.fromtimestamp(camera_data['timestamp']).strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # 创建目录结构
        if individual_id:
            base_dir = os.path.join(SAVE_PATH, individual_id, position.value)
        else:
            base_dir = os.path.join(SAVE_PATH, position.value)

        # 保存深度数据
        if 'depth' in camera_data:
            depth_dir = os.path.join(base_dir, "DEPTH")
            os.makedirs(depth_dir, exist_ok=True)

            depth_data = camera_data['depth']['data']

            # 创建可视化
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            # 生成文件名
            if individual_id:
                depth_filename = f"{individual_id}_{position.value}_depth_{timestamp_str}_{frame_index}.png"
                raw_filename = f"{individual_id}_{position.value}_depth_raw_{timestamp_str}_{frame_index}.npy"
            else:
                depth_filename = f"{position.value}_depth_{timestamp_str}_{frame_index}.png"
                raw_filename = f"{position.value}_depth_raw_{timestamp_str}_{frame_index}.npy"

            depth_filepath = os.path.join(depth_dir, depth_filename)
            raw_filepath = os.path.join(depth_dir, raw_filename)

            cv2.imwrite(depth_filepath, depth_colored)
            np.save(raw_filepath, depth_data)

            # 计算统计信息
            valid_depths = depth_data[depth_data > 0]
            center_y = depth_data.shape[0] // 2
            center_x = depth_data.shape[1] // 2

            result['depth'] = {
                'image_path': depth_filepath,
                'raw_data_path': raw_filepath,
                'width': camera_data['depth']['width'],
                'height': camera_data['depth']['height'],
                'statistics': {
                    'min_depth': int(np.min(valid_depths)) if valid_depths.size > 0 else 0,
                    'max_depth': int(np.max(valid_depths)) if valid_depths.size > 0 else 0,
                    'mean_depth': float(np.mean(valid_depths)) if valid_depths.size > 0 else 0,
                    'center_distance': int(depth_data[center_y, center_x]),
                    'valid_pixel_count': int(valid_depths.size),
                    'valid_pixel_ratio': float(valid_depths.size / (depth_data.shape[0] * depth_data.shape[1]))
                }
            }

        # 保存RGB数据
        if 'rgb' in camera_data:
            rgb_dir = os.path.join(base_dir, "RGB")
            os.makedirs(rgb_dir, exist_ok=True)

            rgb_data = camera_data['rgb']['data']

            # 生成文件名
            if individual_id:
                rgb_filename = f"{individual_id}_{position.value}_rgb_{timestamp_str}_{frame_index}.png"
            else:
                rgb_filename = f"{position.value}_rgb_{timestamp_str}_{frame_index}.png"

            rgb_filepath = os.path.join(rgb_dir, rgb_filename)
            cv2.imwrite(rgb_filepath, cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR))

            result['rgb'] = {
                'image_path': rgb_filepath,
                'width': camera_data['rgb']['width'],
                'height': camera_data['rgb']['height']
            }

        return result

    def get_status(self):
        """获取双相机系统状态"""
        status = {
            'initialized': self.is_initialized,
            'side_camera': None,
            'top_camera': None
        }

        if self.side_camera:
            status['side_camera'] = {
                'running': self.side_camera.is_running,
                'has_depth': self.side_camera.has_depth,
                'has_rgb': self.side_camera.has_rgb,
                'frame_counter': self.side_camera.frame_counter,
                'queue_size': self.side_camera.frame_queue.qsize()
            }

        if self.top_camera:
            status['top_camera'] = {
                'running': self.top_camera.is_running,
                'has_depth': self.top_camera.has_depth,
                'has_rgb': self.top_camera.has_rgb,
                'frame_counter': self.top_camera.frame_counter,
                'queue_size': self.top_camera.frame_queue.qsize()
            }

        return status

    def stop(self):
        """停止双相机系统"""
        if self.side_camera:
            self.side_camera.stop()
        if self.top_camera:
            self.top_camera.stop()
        self.is_initialized = False
        print("Dual camera system stopped")


class RFIDPoller:
    """管理RFID轮询和自动捕获触发"""

    def __init__(self, dual_camera_manager):
        self.camera_manager = dual_camera_manager
        self.is_running = False
        self.poll_thread = None
        self.current_rfid = None  # 当前检测到的RFID
        self.capture_executor = ThreadPoolExecutor(max_workers=2)  # 异步执行捕获任务
        self.capture_in_progress = False  # 标记是否正在捕获
        self.total_captures = 0  # 统计总捕获次数

    def start(self):
        """启动RFID轮询"""
        self.is_running = True
        self.poll_thread = threading.Thread(target=self._poll_loop)
        self.poll_thread.daemon = True
        self.poll_thread.start()
        print(f"RFID polling started - URL: {RFID_URL}, Interval: {RFID_POLL_INTERVAL}s")

    def stop(self):
        """停止RFID轮询"""
        self.is_running = False
        if self.poll_thread:
            self.poll_thread.join()
        self.capture_executor.shutdown(wait=True)
        print(f"RFID polling stopped. Total captures: {self.total_captures}")

    def _poll_loop(self):
        """主轮询循环"""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while self.is_running:
            try:
                # 轮询RFID端点
                response = requests.get(RFID_URL, timeout=2.0)

                if response.status_code == 200:
                    consecutive_errors = 0  # 重置错误计数
                    data = response.json()
                    status = data.get('status')
                    rfid = data.get('rfid')

                    # 检查是否有有效的RFID
                    if status == 200 and rfid is not None:
                        # 检查是否是新的RFID
                        if rfid != self.current_rfid:
                            print(f"\n{'=' * 50}")
                            print(f"New RFID detected: {rfid}")
                            print(f"Previous RFID: {self.current_rfid}")
                            print(f"{'=' * 50}\n")

                            # 更新当前RFID
                            self.current_rfid = rfid

                            # 异步触发捕获，避免阻塞轮询
                            if not self.capture_in_progress:
                                self.capture_executor.submit(self._trigger_capture, rfid)
                            else:
                                print(f"Capture already in progress, skipping RFID: {rfid}")

                    elif status == 100:
                        # 无RFID检测到
                        if self.current_rfid is not None:
                            print(f"\nRFID removed: {self.current_rfid}")
                            self.current_rfid = None

                else:
                    print(f"RFID endpoint returned status code: {response.status_code}")
                    consecutive_errors += 1

            except requests.exceptions.RequestException as e:
                print(f"RFID polling error: {e}")
                consecutive_errors += 1
            except Exception as e:
                print(f"Unexpected error in RFID polling: {e}")
                consecutive_errors += 1

            # 如果连续错误过多，增加等待时间
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}), waiting longer...")
                time.sleep(5.0)
                consecutive_errors = 0
            else:
                # 正常轮询间隔
                time.sleep(RFID_POLL_INTERVAL)

    def _trigger_capture(self, rfid):
        """触发双相机捕获"""
        try:
            self.capture_in_progress = True
            capture_start_time = time.time()

            print(f"\nStarting capture for RFID: {rfid}")
            print(f"Capture config: {RFID_CAPTURE_CONFIG}")

            # 使用同步捕获方法，捕获两个相机，并确保获取实时帧
            frames = self.camera_manager.capture_frames_sync(
                RFID_CAPTURE_CONFIG["num_frames"],
                RFID_CAPTURE_CONFIG["capture_type"],
                cameras=['both'],
                flush_before_capture=True  # 确保捕获实时帧
            )

            if frames:
                # 保存捕获的帧
                save_start_time = time.time()
                saved_paths = self.camera_manager.save_frames(frames, rfid)
                save_duration = time.time() - save_start_time

                self.total_captures += 1
                total_duration = time.time() - capture_start_time

                print(f"\n{'=' * 50}")
                print(f"Capture completed for RFID: {rfid}")
                print(f"Frames captured: {len(frames)}")
                print(f"Capture duration: {total_duration:.2f}s")
                print(f"Save duration: {save_duration:.2f}s")
                print(f"Total captures so far: {self.total_captures}")
                print(f"{'=' * 50}\n")
            else:
                print(f"Failed to capture frames for RFID: {rfid}")

        except Exception as e:
            print(f"Error during RFID-triggered capture: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.capture_in_progress = False

    def get_status(self):
        """获取RFID轮询器状态"""
        return {
            'polling_enabled': self.is_running,
            'current_rfid': self.current_rfid,
            'capture_in_progress': self.capture_in_progress,
            'total_captures': self.total_captures,
            'poll_interval': RFID_POLL_INTERVAL,
            'capture_config': RFID_CAPTURE_CONFIG
        }


# 全局实例
dual_camera_manager = DualCameraManager()
rfid_poller = None  # 将在相机初始化后创建


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    status = dual_camera_manager.get_status()
    if rfid_poller:
        status['rfid_polling'] = rfid_poller.get_status()
    else:
        status['rfid_polling'] = {'enabled': False}
    return jsonify(status)


@app.route('/rfid_status', methods=['GET'])
def rfid_status():
    """获取RFID轮询状态"""
    if rfid_poller:
        return jsonify(rfid_poller.get_status())
    else:
        return jsonify({
            'polling_enabled': False,
            'error': 'RFID poller not initialized'
        })


@app.route('/capture_dual', methods=['POST'])
def capture_dual():
    """
    捕获双相机数据

    请求体:
    {
        "num_frames": 5,
        "capture_type": "both",  // "depth", "rgb", 或 "both"
        "cameras": ["both"],     // ["side"], ["top"], 或 ["both"]
        "id": "person_001",
        "flush_before_capture": true  // 是否在捕获前清空队列以确保实时性
    }
    """
    try:
        data = request.get_json()
        num_frames = data.get('num_frames', 1)
        capture_type = data.get('capture_type', 'both')
        cameras = data.get('cameras', ['both'])
        individual_id = data.get('id')
        flush_before_capture = data.get('flush_before_capture', True)

        # 验证参数
        if not isinstance(num_frames, int) or num_frames < 1:
            return jsonify({
                'success': False,
                'error': 'num_frames must be a positive integer'
            }), 400

        if num_frames > 100:
            return jsonify({
                'success': False,
                'error': 'num_frames cannot exceed 100'
            }), 400

        if capture_type not in ['depth', 'rgb', 'both']:
            return jsonify({
                'success': False,
                'error': 'capture_type must be "depth", "rgb", or "both"'
            }), 400

        # 检查相机状态
        if not dual_camera_manager.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Cameras are not initialized'
            }), 500

        # 捕获帧
        capture_start_time = time.time()
        frames = dual_camera_manager.capture_frames_sync(
            num_frames,
            capture_type,
            cameras,
            flush_before_capture
        )

        if not frames:
            return jsonify({
                'success': False,
                'error': 'Failed to capture frames'
            }), 500

        # 保存帧
        saved_paths = dual_camera_manager.save_frames(frames, individual_id)
        total_duration = time.time() - capture_start_time

        return jsonify({
            'success': True,
            'frames': saved_paths,
            'capture_type': capture_type,
            'cameras': cameras,
            'individual_id': individual_id,
            'total_duration': round(total_duration, 2),
            'fps': round(num_frames / total_duration, 2)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/capture_side', methods=['POST'])
def capture_side():
    """仅捕获侧面相机数据"""
    try:
        data = request.get_json() or {}
        data['cameras'] = ['side']
        request._cached_json = data
        return capture_dual()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/capture_top', methods=['POST'])
def capture_top():
    """仅捕获顶部相机数据"""
    try:
        data = request.get_json() or {}
        data['cameras'] = ['top']
        request._cached_json = data
        return capture_dual()
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/capture_sync', methods=['POST'])
def capture_sync():
    """同步捕获双相机数据（向后兼容）"""
    return capture_dual()


if __name__ == '__main__':
    # 初始化双相机系统
    print("=" * 60)
    print("Orbbec Dual Camera System with RFID Trigger")
    print("=" * 60)
    print("Initializing dual camera system...")

    if dual_camera_manager.initialize():
        print("\nDual camera system initialized successfully!")
        print("\nStarting RFID polling...")

        # 初始化并启动RFID轮询
        rfid_poller = RFIDPoller(dual_camera_manager)
        rfid_poller.start()

        print("\nStarting Flask server...")
        print("=" * 60)
        print("System ready! Waiting for RFID tags...")
        print("=" * 60)

        try:
            app.run(host='0.0.0.0', port=5000, debug=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if rfid_poller:
                rfid_poller.stop()
            dual_camera_manager.stop()
    else:
        print("Failed to initialize dual camera system. Exiting.")