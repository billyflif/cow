import os
import time
import cv2
import numpy as np
from datetime import datetime
from enum import Enum
from pyorbbecsdk import Config, OBSensorType, Pipeline, OBFormat, Context

# 尝试导入OBAlignMode
try:
    from pyorbbecsdk import OBAlignMode

    HAS_ALIGN_MODE = True
except ImportError:
    HAS_ALIGN_MODE = False
    print("警告: 当前SDK版本不支持OBAlignMode")

# ==================== 配置区域 ====================
# 在这里修改所有采集参数

VIDEO_CONFIG = {
    # 要使用的相机配置(可以配置多个相机)
    "cameras": [
        {
            "name": "相机1",  # 相机名称(用于文件命名)
            "device_index": 0,  # 设备索引(0表示第一个相机)
            "enabled":True,  # 是否启用此相机

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
                "format": "MJPG"  # MJPG格式兼容性最好
            },

            # 深度范围配置(单位:毫米)
            "depth_range": {
                "min": 20,
                "max": 60000
            }
        },
        # 如果需要第二个相机,取消下面的注释并修改配置
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
                "format": "MJPG"  # MJPG格式兼容性最好
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
        "rgb": "mp4v",  # RGB视频编码格式 (可选: 'mp4v', 'XVID', 'H264', 'avc1')
        "depth": "mp4v"  # 深度视频编码格式
    },

    # 视频文件格式配置
    "video_format": "mp4",  # 视频文件格式: 'mp4' 或 'avi'

    # 原始深度数据保存配置
    "raw_depth": {
        "enabled": True,  # 是否保存原始深度数据
        "format": "npy",  # 保存格式: 'npy' (numpy二进制) 或 'png' (16位PNG)
        "save_interval": 1,  # 每隔多少帧保存一次原始数据 (1表示每帧都保存)
    }
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
            # 保存为numpy二进制格式(.npy)
            filepath = os.path.join(self.save_path, f"{filename}.npy")
            np.save(filepath, depth_data_uint16)

        elif self.format_type == 'png':
            # 保存为16位PNG图像
            filepath = os.path.join(self.save_path, f"{filename}.png")
            cv2.imwrite(filepath, depth_data_uint16)

        self.frame_counter += 1
        return filepath


class VideoSegmentWriter:
    """视频分段写入器"""

    def __init__(self, camera_name, data_type, width, height, fps, codec, save_path):
        self.camera_name = camera_name
        self.data_type = data_type  # 'rgb' or 'depth'
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.save_path = save_path
        self.writer = None
        self.current_segment_start = None
        self.segment_duration = VIDEO_CONFIG["segment"]["duration"]
        self.current_file_path = None
        self.video_format = VIDEO_CONFIG.get("video_format", "mp4")  # 获取视频格式配置

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
            print(f"\n📹 开始新视频段: {os.path.basename(self.current_file_path)}")

        # 写入帧
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            if self.current_file_path:
                print(f"  ✓ 完成保存: {os.path.basename(self.current_file_path)}")


def format_str_to_enum(format_str):
    """将格式字符串转换为OBFormat枚举"""
    if not format_str:
        return None

    try:
        format_str = format_str.upper().strip()

        # 尝试直接获取
        if hasattr(OBFormat, format_str):
            return getattr(OBFormat, format_str)

        # 尝试常见的别名
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
    """单个相机录制管理器"""

    def __init__(self, config, device_index):
        self.config = config
        self.device_index = device_index
        self.camera_name = config["name"]
        self.pipeline = None
        self.device = None
        self.is_running = False

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
        self.start_time = None

    def initialize(self, context, device_list):
        """初始化相机"""
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

            # 启用帧对齐(如果两个流都启用)
            if self.capture_depth and self.capture_rgb and HAS_ALIGN_MODE:
                try:
                    if hasattr(config_obj, 'set_align_mode'):
                        config_obj.set_align_mode(OBAlignMode.ALIGN_D2C_HW_MODE)
                        print("  ✓ 帧对齐: 已启用")
                except Exception as e:
                    print(f"  ⚠ 帧对齐: 不可用")

            # 启动管道
            self.pipeline.start(config_obj)
            self.is_running = True
            self.start_time = time.time()

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
        print(
            f"    请求: {depth_config['width']}x{depth_config['height']} @ {depth_config['fps']}fps ({depth_config['format']})")

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if profile_list is None:
                print(f"    ❌ 无深度传感器")
                return False

            # 转换格式
            format_enum = format_str_to_enum(depth_config['format'])
            if format_enum is None:
                print(f"    ⚠ 格式 '{depth_config['format']}' 无效,使用默认格式")
                format_enum = OBFormat.Y16

            # 直接使用参数获取配置
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

                    # 创建深度视频写入器
                    save_path = os.path.join(
                        VIDEO_CONFIG["segment"]["save_path"],
                        self.camera_name,
                        "depth"
                    )
                    self.depth_writer = VideoSegmentWriter(
                        self.camera_name,
                        "depth",
                        actual_width,
                        actual_height,
                        actual_fps,
                        VIDEO_CONFIG["video_codec"]["depth"],
                        save_path
                    )

                    # 创建原始深度数据写入器
                    if self.raw_depth_enabled:
                        raw_save_path = os.path.join(
                            VIDEO_CONFIG["segment"]["save_path"],
                            self.camera_name,
                            "depth_raw"
                        )
                        self.raw_depth_writer = RawDepthWriter(
                            self.camera_name,
                            raw_save_path,
                            VIDEO_CONFIG["raw_depth"]["format"]
                        )

                    return True
                else:
                    print(f"    ❌ 配置不支持")
                    return False

            except Exception as e:
                print(f"    ❌ 配置失败: {e}")
                print(f"    💡 提示: 请检查相机是否支持该分辨率和帧率")
                return False

        except Exception as e:
            print(f"    ❌ 错误: {e}")
            return False

    def _configure_rgb_stream(self, config_obj):
        """配置RGB流"""
        rgb_config = self.config["rgb"]
        print(f"\n  🎨 RGB流配置:")
        print(
            f"    请求: {rgb_config['width']}x{rgb_config['height']} @ {rgb_config['fps']}fps ({rgb_config['format']})")

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is None:
                print(f"    ❌ 无RGB传感器")
                return False

            # 转换格式
            format_enum = format_str_to_enum(rgb_config['format'])
            if format_enum is None:
                print(f"    ⚠ 格式 '{rgb_config['format']}' 无效,使用MJPG")
                format_enum = OBFormat.MJPG

            # 直接使用参数获取配置
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

                    # 创建RGB视频写入器
                    save_path = os.path.join(
                        VIDEO_CONFIG["segment"]["save_path"],
                        self.camera_name,
                        "rgb"
                    )
                    self.rgb_writer = VideoSegmentWriter(
                        self.camera_name,
                        "rgb",
                        actual_width,
                        actual_height,
                        actual_fps,
                        VIDEO_CONFIG["video_codec"]["rgb"],
                        save_path
                    )
                    return True
                else:
                    print(f"    ❌ 配置不支持")
                    return False

            except Exception as e:
                print(f"    ❌ 配置失败: {e}")
                print(f"    💡 提示: 请检查相机是否支持该分辨率和帧率")
                print(f"    💡 建议: 对于1280x720分辨率,使用10fps或15fps")
                return False

        except Exception as e:
            print(f"    ❌ 错误: {e}")
            return False

    def capture_and_save(self):
        """采集并保存一帧"""
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return False

            self.frame_count += 1

            # 处理深度帧
            if self.capture_depth and self.depth_writer:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_data_raw = self._process_depth_frame(depth_frame)
                    if depth_data_raw is not None:
                        # 保存原始深度数据(uint16格式)
                        if self.raw_depth_writer and (self.frame_count % self.raw_depth_interval == 0):
                            self.raw_depth_writer.write_depth_frame(depth_data_raw)

                        # 将深度数据转换为彩色图像用于视频保存
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
                        # OpenCV使用BGR格式
                        bgr_data = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
                        self.rgb_writer.write_frame(bgr_data)

            # 每100帧打印一次统计信息
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                print(f"[{self.camera_name}] 已采集: {self.frame_count} 帧, "
                      f"实际帧率: {fps:.2f} FPS")

            return True

        except Exception as e:
            return False

    def _process_depth_frame(self, depth_frame):
        """处理深度帧,返回uint16格式的原始深度数据"""
        try:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            # 获取原始深度数据(uint16格式)
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
                if decoded is None:
                    print("  - [!!!!] WARNING: cv2.imdecode failed! Frame data might be corrupt.")
                elif decoded is not None:
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
                # 尝试解码
                decoded = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                if decoded is not None:
                    color_data = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

            return color_data

        except Exception:
            return None

    def stop(self):
        """停止录制"""
        self.is_running = False

        if self.depth_writer:
            self.depth_writer.release()
        if self.rgb_writer:
            self.rgb_writer.release()
        if self.pipeline:
            self.pipeline.stop()

        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 {self.camera_name} 录制统计:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  总时长: {elapsed:.2f}秒")
            print(f"  平均帧率: {avg_fps:.2f} FPS")
            if self.raw_depth_writer:
                print(f"  保存的原始深度帧数: {self.raw_depth_writer.frame_counter}")


# ==================== 主程序 ====================

class MultiCameraRecorder:
    """多相机录制管理器"""

    def __init__(self):
        self.recorders = []
        self.context = None
        self.is_running = False

    def initialize(self):
        """初始化所有相机"""
        try:
            self.context = Context()
            device_list = self.context.query_devices()
            device_count = device_list.get_count()

            print("\n" + "=" * 70)
            print("🎬 奥比中光多相机视频采集系统 v3.4 (支持MP4/AVI格式)")
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

    def start_recording(self):
        """开始录制"""
        self.is_running = True
        print("\n🎥 开始录制... (按 Ctrl+C 停止)")
        print(f"📁 视频保存路径: {os.path.abspath(VIDEO_CONFIG['segment']['save_path'])}")
        print(f"⏱ 视频分段时长: {VIDEO_CONFIG['segment']['duration']}秒")
        if VIDEO_CONFIG["raw_depth"]["enabled"]:
            print(f"💾 原始深度数据: 已启用 (格式: {VIDEO_CONFIG['raw_depth']['format']}, "
                  f"间隔: 每{VIDEO_CONFIG['raw_depth']['save_interval']}帧)")
        print("-" * 70)

        try:
            while self.is_running:
                # 从所有相机采集帧
                for recorder in self.recorders:
                    if recorder.is_running:
                        recorder.capture_and_save()

                # 小延迟避免CPU占用过高
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n⏹ 收到停止信号...")
        finally:
            self.stop()

    def stop(self):
        """停止所有录制"""
        print("\n⏹ 停止录制中...")
        self.is_running = False

        for recorder in self.recorders:
            recorder.stop()

        print("\n" + "=" * 70)
        print("✅ 录制已完成")
        print(f"📁 视频保存在: {os.path.abspath(VIDEO_CONFIG['segment']['save_path'])}")
        if VIDEO_CONFIG["raw_depth"]["enabled"]:
            print(f"💾 原始深度数据保存在: {os.path.abspath(VIDEO_CONFIG['segment']['save_path'])}/*/depth_raw/")
        print("=" * 70)


def print_config_summary():
    """打印配置摘要"""
    print("\n📋 当前配置:")
    print("-" * 70)

    enabled_cameras = [cam for cam in VIDEO_CONFIG["cameras"] if cam["enabled"]]

    if not enabled_cameras:
        print("  ⚠ 警告: 没有启用的相机")
        return

    for cam in enabled_cameras:
        print(f"\n🎥 相机: {cam['name']}")
        print(f"  设备索引: {cam['device_index']}")

        modes = []
        if cam["capture_mode"]["depth"]:
            modes.append("深度")
        if cam["capture_mode"]["rgb"]:
            modes.append("RGB")
        print(f"  采集模式: {' + '.join(modes)}")

        if cam["capture_mode"]["depth"]:
            d = cam["depth"]
            print(f"  深度配置: {d['width']}x{d['height']} @ {d['fps']}fps ({d['format']})")
            dr = cam["depth_range"]
            print(f"  深度范围: {dr['min']}-{dr['max']}mm")

        if cam["capture_mode"]["rgb"]:
            r = cam["rgb"]
            print(f"  RGB配置: {r['width']}x{r['height']} @ {r['fps']}fps ({r['format']})")

    print(f"\n⏱ 视频分段时长: {VIDEO_CONFIG['segment']['duration']}秒")
    print(f"📁 保存路径: {VIDEO_CONFIG['segment']['save_path']}")
    print(f"🎬 视频格式: {VIDEO_CONFIG.get('video_format', 'mp4').upper()}")
    print(f"🎬 视频编码: RGB={VIDEO_CONFIG['video_codec']['rgb']}, Depth={VIDEO_CONFIG['video_codec']['depth']}")

    if VIDEO_CONFIG["raw_depth"]["enabled"]:
        print(f"\n💾 原始深度数据保存:")
        print(f"  启用: 是")
        print(f"  格式: {VIDEO_CONFIG['raw_depth']['format']}")
        print(f"  保存间隔: 每{VIDEO_CONFIG['raw_depth']['save_interval']}帧")
    else:
        print(f"\n💾 原始深度数据保存: 否")

    print("-" * 70)


def main():
    """主函数"""
    # 打印配置信息
    print_config_summary()

    # 创建并启动录制器
    recorder = MultiCameraRecorder()
    if recorder.initialize():
        recorder.start_recording()
    else:
        print("\n❌ 系统初始化失败,程序退出")
        print("\n💡 提示:")
        print("  1. 检查相机是否正确连接")
        print("  2. 检查配置的分辨率、帧率和格式是否被相机支持")
        print("  3. 常见支持的配置:")
        print("     - 1280x720 @ 15fps (MJPG)")
        print("     - 1280x720 @ 10fps (MJPG)")
        print("     - 640x480 @ 30fps (MJPG/RGB888)")
        print("  4. 深度流通常使用: 640x480 @ 30fps (Y16)")


if __name__ == '__main__':
    main()