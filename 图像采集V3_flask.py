import os
import time
import cv2
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from flask import Flask, request, jsonify
from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat, Context

# 尝试导入OBAlignMode
try:
    from pyorbbecsdk import OBAlignMode
    HAS_ALIGN_MODE = True
except ImportError:
    HAS_ALIGN_MODE = False
    print("警告: 当前SDK版本不支持OBAlignMode")

# ==================== 配置区域 ====================

VIDEO_CONFIG = {
    # 要使用的相机配置(可以配置多个相机)
    "cameras": [
        {
            "name": "相机1",  # 相机名称(用于文件命名)
            "device_index": 0,  # 设备索引(0表示第一个相机)
            "enabled": True,  # 是否启用此相机

            # 采集模式配置
            "capture_mode": {
                "depth": True,  # 是否采集深度数据
                "rgb": True,  # 是否采集RGB数据
            },

            # 深度流配置
            "depth": {
                "width": 640,
                "height": 480,
                "fps": 15,
                "format": "Y16"
            },

            # RGB流配置
            "rgb": {
                "width": 1280,
                "height": 720,
                "fps": 15,
                "format": "MJPG"
            },

            # 深度范围配置(单位:毫米)
            "depth_range": {
                "min": 20,
                "max": 60000
            }
        },
        {
            "name": "相机2",
            "device_index": 1,
            "enabled": True,
            "capture_mode": {
                "depth": True,
                "rgb": True,
            },
            "depth": {
                "width": 640,
                "height": 480,
                "fps": 15,
                "format": "Y16"
            },
            "rgb": {
                "width": 1280,
                "height": 720,
                "fps": 15,
                "format": "MJPG"
            },
            "depth_range": {
                "min": 20,
                "max": 60000
            }
        }
    ],

    # 视频分段保存配置
    "segment": {
        "duration": 60,  # 每段视频的时长(秒)
        "save_path": "./videos",  # 视频保存路径
    },

    # 视频编码配置
    "video_codec": {
        "rgb": "mp4v",
        "depth": "mp4v"
    },

    # 视频文件格式配置
    "video_format": "mp4",

    # 原始深度数据保存配置
    "raw_depth": {
        "enabled": True,
        "format": "npy",
        "save_interval": 1,
    }
}

# Flask服务配置
FLASK_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False
}

# ==================== 工具类 ====================

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

    def reset(self):
        """重置滤波器状态"""
        self.previous_frame = None


class RawDepthWriter:
    """原始深度数据写入器"""

    def __init__(self, camera_name, save_path, format_type='npy'):
        self.camera_name = camera_name
        self.save_path = save_path
        self.format_type = format_type
        self.frame_counter = 0

        os.makedirs(save_path, exist_ok=True)
        print(f"  ✓ 原始深度数据保存路径: {save_path}")
        print(f"  ✓ 保存格式: {format_type}")

    def write_depth_frame(self, depth_data_uint16):
        """保存原始深度数据(uint16格式)"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.camera_name}_depth_raw_{timestamp_str}_frame{self.frame_counter:06d}"

        if self.format_type == 'npy':
            filepath = os.path.join(self.save_path, f"{filename}.npy")
            np.save(filepath, depth_data_uint16)
        elif self.format_type == 'png':
            filepath = os.path.join(self.save_path, f"{filename}.png")
            cv2.imwrite(filepath, depth_data_uint16)

        self.frame_counter += 1
        return filepath


class VideoSegmentWriter:
    """视频分段写入器(支持按时长分段)"""

    def __init__(self, camera_name, data_type, width, height, fps, codec, save_path):
        self.camera_name = camera_name
        self.data_type = data_type
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.save_path = save_path
        self.writer = None
        self.current_segment_start = None
        self.segment_duration = VIDEO_CONFIG["segment"]["duration"]
        self.current_file_path = None
        self.video_format = VIDEO_CONFIG.get("video_format", "mp4")

        os.makedirs(save_path, exist_ok=True)

    def write_frame(self, frame):
        """写入帧,如果需要则创建新的视频段"""
        current_time = time.time()

        # 检查是否需要创建新的视频段
        if (self.writer is None or
                self.current_segment_start is None or
                current_time - self.current_segment_start >= self.segment_duration):

            # 关闭当前视频文件
            if self.writer is not None:
                self.writer.release()
                print(f"  ✓ 完成保存: {os.path.basename(self.current_file_path)}")

            # 创建新的视频文件
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.camera_name}_{self.data_type}_{timestamp_str}.{self.video_format}"
            self.current_file_path = os.path.join(self.save_path, filename)

            self.writer = cv2.VideoWriter(
                self.current_file_path,
                self.codec,
                self.fps,
                (self.width, self.height)
            )

            self.current_segment_start = current_time
            print(f"📹 开始新视频段: {os.path.basename(self.current_file_path)}")

        # 写入帧
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            if self.current_file_path:
                print(f"  ✓ 完成保存: {os.path.basename(self.current_file_path)}")
            self.writer = None
            self.current_segment_start = None


def format_str_to_enum(format_str):
    """将格式字符串转换为OBFormat枚举"""
    if not format_str:
        return None

    try:
        format_str = format_str.upper().strip()

        if hasattr(OBFormat, format_str):
            return getattr(OBFormat, format_str)

        format_aliases = {
            'MJPEG': 'MJPG',
            'JPEG': 'MJPG',
            'YUV': 'YUYV',
            'RGB': 'RGB888',
        }

        if format_str in format_aliases:
            alias = format_aliases[format_str]
            if hasattr(OBFormat, alias):
                return getattr(OBFormat, alias)

        return None

    except Exception:
        return None


class CameraRecorder:
    """单个相机录制管理器(支持持续运行,按需保存)"""

    def __init__(self, config, device_index):
        self.config = config
        self.device_index = device_index
        self.camera_name = config["name"]
        self.pipeline = None
        self.device = None
        self.is_initialized = False

        # 录制控制
        self.is_recording = False  # 是否正在录制(保存数据)
        self.current_id = None  # 当前录制的ID
        self.recording_start_time = None
        self.recording_duration = VIDEO_CONFIG["segment"]["duration"]

        # 采集模式
        self.capture_depth = config["capture_mode"]["depth"]
        self.capture_rgb = config["capture_mode"]["rgb"]

        # 滤波器
        self.temporal_filter = TemporalFilter(alpha=0.5)

        # 视频写入器
        self.rgb_writer = None
        self.depth_writer = None

        # 原始深度数据写入器
        self.raw_depth_writer = None
        self.raw_depth_enabled = VIDEO_CONFIG["raw_depth"]["enabled"] and self.capture_depth
        self.raw_depth_interval = VIDEO_CONFIG["raw_depth"]["save_interval"]

        # 统计信息
        self.frame_count = 0
        self.total_frames = 0

    def initialize(self, context, device_list):
        """初始化相机(只初始化一次)"""
        if self.is_initialized:
            print(f"⚠ {self.camera_name} 已经初始化")
            return True

        try:
            device_count = device_list.get_count()
            if self.device_index >= device_count:
                print(f"❌ 错误: 设备索引 {self.device_index} 超出范围,总设备数: {device_count}")
                return False

            self.device = device_list.get_device_by_index(self.device_index)
            device_info = self.device.get_device_info()

            print(f"\n{'=' * 60}")
            print(f"🎥 初始化相机: {self.camera_name}")
            print(f"  设备名称: {device_info.get_name()}")
            print(f"  序列号: {device_info.get_serial_number()}")
            print(f"  设备索引: {self.device_index}")

            config_obj = Config()
            self.pipeline = Pipeline(self.device)

            # 配置深度流
            if self.capture_depth:
                success = self._configure_depth_stream(config_obj)
                if not success:
                    print(f"  ⚠ 深度流配置失败")
                    self.capture_depth = False

            # 配置RGB流
            if self.capture_rgb:
                success = self._configure_rgb_stream(config_obj)
                if not success:
                    print(f"  ⚠ RGB流配置失败")
                    self.capture_rgb = False

            if not self.capture_depth and not self.capture_rgb:
                print("  ❌ 错误: 没有可用的传感器")
                return False

            # 启用帧对齐
            if self.capture_depth and self.capture_rgb and HAS_ALIGN_MODE:
                try:
                    if hasattr(config_obj, 'set_align_mode'):
                        config_obj.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)
                        print("  ✓ 帧对齐: 已启用")
                except Exception:
                    print(f"  ⚠ 帧对齐: 不可用")

            # 启动管道
            self.pipeline.start(config_obj)
            self.is_initialized = True

            print(f"\n✅ {self.camera_name} 初始化成功!")
            print(f"{'=' * 60}")
            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _configure_depth_stream(self, config_obj):
        """配置深度流"""
        depth_config = self.config["depth"]
        print(f"\n  📊 深度流配置:")
        print(f"    请求: {depth_config['width']}x{depth_config['height']} @ {depth_config['fps']}fps ({depth_config['format']})")

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list is None:
                print(f"    ❌ 无深度传感器")
                return False

            format_enum = format_str_to_enum(depth_config['format'])
            if format_enum is None:
                print(f"    ⚠ 格式 '{depth_config['format']}' 无效,使用默认格式")
                format_enum = OBFormat.Y16

            try:
                depth_profile = profile_list.get_video_stream_profile(
                    depth_config['width'],
                    depth_config['height'],
                    format_enum,
                    depth_config['fps']
                )

                if depth_profile:
                    actual_width = depth_profile.get_width()
                    actual_height = depth_profile.get_height()
                    actual_fps = depth_profile.get_fps()

                    print(f"    实际: {actual_width}x{actual_height} @ {actual_fps}fps")
                    print(f"    ✓ 深度流配置成功")

                    config_obj.enable_stream(depth_profile)

                    # 保存配置供后续创建写入器使用
                    self.depth_config = {
                        'width': actual_width,
                        'height': actual_height,
                        'fps': actual_fps
                    }

                    return True
                else:
                    print(f"    ❌ 配置不支持")
                    return False

            except Exception as e:
                print(f"    ❌ 配置失败: {e}")
                return False

        except Exception as e:
            print(f"    ❌ 错误: {e}")
            return False

    def _configure_rgb_stream(self, config_obj):
        """配置RGB流"""
        rgb_config = self.config["rgb"]
        print(f"\n  🎨 RGB流配置:")
        print(f"    请求: {rgb_config['width']}x{rgb_config['height']} @ {rgb_config['fps']}fps ({rgb_config['format']})")

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is None:
                print(f"    ❌ 无RGB传感器")
                return False

            format_enum = format_str_to_enum(rgb_config['format'])
            if format_enum is None:
                print(f"    ⚠ 格式 '{rgb_config['format']}' 无效,使用MJPG")
                format_enum = OBFormat.MJPG

            try:
                rgb_profile = profile_list.get_video_stream_profile(
                    rgb_config['width'],
                    rgb_config['height'],
                    format_enum,
                    rgb_config['fps']
                )

                if rgb_profile:
                    actual_width = rgb_profile.get_width()
                    actual_height = rgb_profile.get_height()
                    actual_fps = rgb_profile.get_fps()

                    print(f"    实际: {actual_width}x{actual_height} @ {actual_fps}fps")
                    print(f"    ✓ RGB流配置成功")

                    config_obj.enable_stream(rgb_profile)

                    # 保存配置供后续创建写入器使用
                    self.rgb_config = {
                        'width': actual_width,
                        'height': actual_height,
                        'fps': actual_fps
                    }

                    return True
                else:
                    print(f"    ❌ 配置不支持")
                    return False

            except Exception as e:
                print(f"    ❌ 配置失败: {e}")
                return False

        except Exception as e:
            print(f"    ❌ 错误: {e}")
            return False

    def start_recording_for_id(self, record_id, duration=None):
        """开始为指定ID录制"""
        # 如果正在录制其他ID,先停止
        if self.is_recording:
            self.stop_recording()

        self.current_id = record_id
        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_duration = duration if duration else VIDEO_CONFIG["segment"]["duration"]
        self.frame_count = 0

        # 重置滤波器
        self.temporal_filter.reset()

        # 创建视频写入器
        if self.capture_depth and hasattr(self, 'depth_config'):
            depth_save_path = os.path.join(
                VIDEO_CONFIG["segment"]["save_path"],
                self.camera_name,
                "depth",
                record_id
            )
            self.depth_writer = VideoSegmentWriter(
                self.camera_name,
                "depth",
                self.depth_config['width'],
                self.depth_config['height'],
                self.depth_config['fps'],
                VIDEO_CONFIG["video_codec"]["depth"],
                depth_save_path
            )

            # 创建原始深度数据写入器
            if self.raw_depth_enabled:
                raw_save_path = os.path.join(
                    VIDEO_CONFIG["segment"]["save_path"],
                    self.camera_name,
                    "depth_raw",
                    record_id
                )
                self.raw_depth_writer = RawDepthWriter(
                    self.camera_name,
                    raw_save_path,
                    VIDEO_CONFIG["raw_depth"]["format"]
                )

        if self.capture_rgb and hasattr(self, 'rgb_config'):
            rgb_save_path = os.path.join(
                VIDEO_CONFIG["segment"]["save_path"],
                self.camera_name,
                "rgb",
                record_id
            )
            self.rgb_writer = VideoSegmentWriter(
                self.camera_name,
                "rgb",
                self.rgb_config['width'],
                self.rgb_config['height'],
                self.rgb_config['fps'],
                VIDEO_CONFIG["video_codec"]["rgb"],
                rgb_save_path
            )

        print(f"\n🎬 {self.camera_name} 开始录制 ID: {record_id}, 预计时长: {self.recording_duration}秒")

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.depth_writer:
            self.depth_writer.release()
            self.depth_writer = None

        if self.rgb_writer:
            self.rgb_writer.release()
            self.rgb_writer = None

        if self.raw_depth_writer:
            print(f"  💾 保存的原始深度帧数: {self.raw_depth_writer.frame_counter}")
            self.raw_depth_writer = None

        elapsed = time.time() - self.recording_start_time if self.recording_start_time else 0
        print(f"\n⏹ {self.camera_name} 停止录制 ID: {self.current_id}")
        print(f"  录制帧数: {self.frame_count}, 时长: {elapsed:.2f}秒")

        self.current_id = None
        self.recording_start_time = None

    def capture_frame(self):
        """采集一帧(始终运行,但只在录制时保存)"""
        if not self.is_initialized:
            return False

        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return False

            self.total_frames += 1

            # 只有在录制时才保存数据
            if self.is_recording:
                self.frame_count += 1

                # 检查是否达到录制时长
                if self.recording_start_time:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed >= self.recording_duration:
                        print(f"\n⏰ {self.camera_name} 达到录制时长 ({self.recording_duration}秒), 自动停止")
                        self.stop_recording()
                        return True

                # 处理深度帧
                if self.capture_depth and self.depth_writer:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is not None:
                        depth_data_raw = self._process_depth_frame(depth_frame)
                        if depth_data_raw is not None:
                            # 保存原始深度数据
                            if self.raw_depth_writer and (self.frame_count % self.raw_depth_interval == 0):
                                self.raw_depth_writer.write_depth_frame(depth_data_raw)

                            # 转换为彩色图像保存视频
                            depth_normalized = cv2.normalize(
                                depth_data_raw, None, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U
                            )
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            self.depth_writer.write_frame(depth_colored)

                # 处理RGB帧
                if self.capture_rgb and self.rgb_writer:
                    color_frame = frames.get_color_frame()
                    if color_frame is not None:
                        rgb_data = self._process_rgb_frame(color_frame)
                        if rgb_data is not None:
                            bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                            self.rgb_writer.write_frame(bgr_data)

                # 每100帧打印一次统计
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.recording_start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{self.camera_name}] ID: {self.current_id}, 已录制: {self.frame_count} 帧, "
                          f"实际帧率: {fps:.2f} FPS")

            return True

        except Exception as e:
            return False

    def _process_depth_frame(self, depth_frame):
        """处理深度帧"""
        try:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))

            # 应用深度范围过滤
            depth_range = self.config["depth_range"]
            depth_data_mm = depth_data.astype(np.float32) * scale
            depth_data_filtered = np.where(
                (depth_data_mm > depth_range["min"]) & (depth_data_mm < depth_range["max"]),
                depth_data, 0
            ).astype(np.uint16)

            # 应用时间滤波
            depth_data_filtered = self.temporal_filter.process(depth_data_filtered)

            return depth_data_filtered

        except Exception:
            return None

    def _process_rgb_frame(self, color_frame):
        """处理RGB帧"""
        try:
            width = color_frame.get_width()
            height = color_frame.get_height()
            format = color_frame.get_format()
            raw_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

            color_data = None

            if format == OBFormat.MJPG:
                decoded = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    color_data = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            elif hasattr(OBFormat, 'RGB888') and format == OBFormat.RGB888:
                color_data = raw_data.reshape((height, width, 3))
            elif hasattr(OBFormat, 'RGB') and format == OBFormat.RGB:
                color_data = raw_data.reshape((height, width, 3))
            elif format == OBFormat.BGR:
                bgr_data = raw_data.reshape((height, width, 3))
                color_data = cv2.cvtColor(bgr_data, cv2.COLOR_BGR2RGB)
            elif format == OBFormat.YUYV:
                yuv_data = raw_data.reshape((height, width, 2))
                color_data = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2RGB_YUYV)
            else:
                decoded = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    color_data = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

            return color_data

        except Exception:
            return None

    def shutdown(self):
        """关闭相机"""
        if self.is_recording:
            self.stop_recording()

        if self.pipeline and self.is_initialized:
            self.pipeline.stop()
            self.is_initialized = False

        print(f"\n📊 {self.camera_name} 总统计: 总采集帧数: {self.total_frames}")


# ==================== Flask服务 ====================

class CameraService:
    """相机服务管理器"""

    def __init__(self):
        self.recorders = []
        self.context = None
        self.is_running = False
        self.capture_thread = None
        self.lock = Lock()

    def initialize(self):
        """初始化所有相机"""
        try:
            self.context = Context()
            device_list = self.context.query_devices()
            device_count = device_list.get_count()

            print("\n" + "=" * 70)
            print("🎬 奥比中光多相机采集系统 - Flask服务模式")
            print("=" * 70)
            print(f"检测到 {device_count} 个设备")

            if device_count == 0:
                print("\n❌ 错误: 未检测到任何相机设备")
                return False

            # 初始化每个启用的相机
            for cam_config in VIDEO_CONFIG["cameras"]:
                if not cam_config["enabled"]:
                    print(f"\n⏭ 跳过 {cam_config['name']} (未启用)")
                    continue

                if cam_config["device_index"] >= device_count:
                    print(f"\n⚠ 警告: {cam_config['name']} 的设备索引 {cam_config['device_index']} 超出范围")
                    continue

                recorder = CameraRecorder(cam_config, cam_config["device_index"])
                if recorder.initialize(self.context, device_list):
                    self.recorders.append(recorder)
                else:
                    print(f"\n⚠ 警告: {cam_config['name']} 初始化失败")

            if not self.recorders:
                print("\n❌ 错误: 没有成功初始化的相机")
                return False

            print("\n" + "=" * 70)
            print(f"✅ 成功初始化 {len(self.recorders)} 个相机")
            print("=" * 70)
            return True

        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_service(self):
        """启动采集服务(相机持续运行)"""
        if self.is_running:
            return

        self.is_running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("\n🎥 相机采集服务已启动(持续运行模式)")

    def _capture_loop(self):
        """采集循环(持续运行)"""
        while self.is_running:
            with self.lock:
                for recorder in self.recorders:
                    recorder.capture_frame()
            time.sleep(0.001)

    def start_recording(self, record_id, duration=None):
        """开始录制指定ID的数据"""
        with self.lock:
            for recorder in self.recorders:
                recorder.start_recording_for_id(record_id, duration)

    def stop_recording(self):
        """停止所有录制"""
        with self.lock:
            for recorder in self.recorders:
                if recorder.is_recording:
                    recorder.stop_recording()

    def get_status(self):
        """获取服务状态"""
        with self.lock:
            status = {
                "service_running": self.is_running,
                "cameras": []
            }
            for recorder in self.recorders:
                cam_status = {
                    "name": recorder.camera_name,
                    "is_recording": recorder.is_recording,
                    "current_id": recorder.current_id,
                    "frame_count": recorder.frame_count,
                    "total_frames": recorder.total_frames
                }
                status["cameras"].append(cam_status)
            return status

    def shutdown(self):
        """关闭服务"""
        print("\n⏹ 关闭相机服务...")
        self.is_running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2)

        with self.lock:
            for recorder in self.recorders:
                recorder.shutdown()

        print("✅ 相机服务已关闭")


# ==================== Flask应用 ====================

app = Flask(__name__)
camera_service = CameraService()


@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    """
    开始录制接口
    POST /api/start_recording
    Body: {"id": "用户ID", "duration": 60 (可选,默认使用配置的duration)}
    """
    try:
        data = request.get_json()
        if not data or 'id' not in data:
            return jsonify({
                "success": False,
                "message": "缺少必需参数: id"
            }), 400

        record_id = str(data['id']).strip()
        if not record_id:
            return jsonify({
                "success": False,
                "message": "ID不能为空"
            }), 400

        duration = data.get('duration', None)
        if duration is not None:
            try:
                duration = int(duration)
                if duration <= 0:
                    return jsonify({
                        "success": False,
                        "message": "duration必须大于0"
                    }), 400
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "message": "duration必须是整数"
                }), 400

        # 开始录制
        camera_service.start_recording(record_id, duration)

        return jsonify({
            "success": True,
            "message": f"开始录制 ID: {record_id}",
            "id": record_id,
            "duration": duration if duration else VIDEO_CONFIG["segment"]["duration"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500


@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    """
    停止录制接口
    POST /api/stop_recording
    """
    try:
        camera_service.stop_recording()
        return jsonify({
            "success": True,
            "message": "已停止所有录制"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """
    获取服务状态接口
    GET /api/status
    """
    try:
        status = camera_service.get_status()
        return jsonify({
            "success": True,
            "data": status
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"服务器错误: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    GET /api/health
    """
    return jsonify({
        "success": True,
        "message": "服务运行正常"
    })


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🚀 启动Flask相机采集服务")
    print("=" * 70)

    # 初始化相机服务
    if not camera_service.initialize():
        print("\n❌ 相机初始化失败,程序退出")
        return

    # 启动相机采集服务
    camera_service.start_service()

    print("\n" + "=" * 70)
    print("📡 Flask API服务启动中...")
    print(f"   地址: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print("\n可用接口:")
    print("  POST /api/start_recording - 开始录制 (参数: {\"id\": \"用户ID\", \"duration\": 60})")
    print("  POST /api/stop_recording  - 停止录制")
    print("  GET  /api/status          - 获取状态")
    print("  GET  /api/health          - 健康检查")
    print("=" * 70)

    try:
        # 启动Flask服务
        app.run(
            host=FLASK_CONFIG['host'],
            port=FLASK_CONFIG['port'],
            debug=FLASK_CONFIG['debug'],
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n⏹ 收到停止信号...")
    finally:
        camera_service.shutdown()


if __name__ == '__main__':
    main()
