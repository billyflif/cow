

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from ultralytics import YOLO
import time
import logging
from datetime import datetime
import requests
import json

# 导入自定义模型
from model import CowReIDModel
# 导入体尺数据
from cow_body_measurements import COW_BODY_MEASUREMENTS, MEASUREMENT_LABELS
# 导入ID映射配置
from cow_id_mapping import virtual_to_real_id

# ===================== 可配置参数 =====================

# 海康威视摄像头配置
USE_CAMERA_URL = True  # 是否使用摄像头URL
CAMERA_URL = "rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101"  # 海康威视RTSP URL
# RTSP URL格式: rtsp://用户名:密码@IP地址:端口/Streaming/Channels/通道号
# 例如: rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101
# 如果使用HTTP流，格式: http://IP地址:端口/stream

# 如果USE_CAMERA_URL为False，则使用本地视频文件
VIDEO_PATH = r"E:\COW\Cow-Re-ID\0722\video1021\48.mp4"

GALLERY_PATH = r"E:\COW\Obc-SDK-Test\gallery\video1022-frame-8"

YOLO_MODEL_PATH = "E:\COW\Obc-SDK-Test\checkpoints\yolo11n.pt"
REID_MODEL_PATH = r"E:\COW\Obc-SDK-Test\checkpoints\best_model.pth"

SIMILARITY_THRESH = 0.60
CONFIDENCE_THRESH = 0.03
IOU_THRESH = 0.4

FONT_SCALE = 2.0  # <--- 增大字体
ID_LABEL_THICKNESS = 6  # <--- ID标签加粗
MEASUREMENT_FONT_SCALE = 1.5  # <--- 体尺指标字体大小（增大以突出体重显示）
MEASUREMENT_THICKNESS = 3  # <--- 体尺指标粗细
TRACK_HISTORY = 30
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
SHOW_FPS = False  # <--- 关闭FPS显示

LOG_DIR = "./logs"
ENABLE_LOGGING = True

USE_CLASS_FILTER = False
DETECT_CLASSES = [19, 20, 21, 22, 23]

MAX_GALLERY_IMAGES = 15

# ---- 平滑与稳定参数 ----
SMOOTH_WINDOW = 12
MIN_VOTE_SAMPLES = 5
REID_BATCH_SIZE = 1

# 视频保存
SAVE_VIDEO = True
OUTPUT_VIDEO_DIR = "./output_videos"
VIDEO_FPS = 10

# 边缘过滤
EDGE_FILTER_RATIO = 0.08
MIN_BOX_WIDTH_RATIO = 0.04

# <--- 新增：中间区域定义（用于体尺数据显示）
CENTER_REGION_RATIO = 0.3  # 画面中间60%区域（左右各留20%）

# ID稳定性增强
HIGH_CONF_THRESH = 0.75
LOCK_FRAME_COUNT = 6
LOCKED_ID_DECAY = 60
UNLOCK_REQUIRE_FRAMES = 20

# 初始帧过滤
INITIAL_FRAMES_SKIP = 10
INITIAL_HIGH_CONF_THRESH = 0.80

# 跟踪丢失容忍
MAX_LOST_FRAMES = 15
MIN_BOX_AREA = 1000

# 体尺数据噪声参数
MEASUREMENT_NOISE_RATIO = 0.02  # 2%的随机噪声

# 统一红色
LOCKED_COLOR = (0, 0, 255)  # 红色 (BGR格式)

# API配置
API_URLS = [
    "https://lezhi.muguanjia.net/api/manage/cow_body_log/snycData",
    "https://yz.muguanjia.net/api/manage/cow_body_log/snycData"
]
API_TIMEOUT = 5  # API请求超时时间（秒）

# RTSP连接参数
RTSP_BUFFER_SIZE = 1024 * 1024  # 1MB缓冲区
RTSP_TIMEOUT = 5000  # 5秒超时


# ======================================================

def setup_logger():
    if ENABLE_LOGGING:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"reid_system_hikvision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger(__name__)


logger = setup_logger()


def minimize_all_windows():
    """最小化所有窗口，返回桌面"""
    try:
        import pygetwindow as gw
        windows = gw.getAllWindows()
        for window in windows:
            if window.title and window.isActive:
                try:
                    window.minimize()
                except:
                    pass
        time.sleep(0.5)  # 等待窗口最小化完成
        logger.info("已最小化所有窗口，返回桌面")
    except ImportError:
        logger.warning("pygetwindow未安装，无法最小化窗口。可使用: pip install pygetwindow")
    except Exception as e:
        logger.warning(f"最小化窗口失败: {e}")


def send_measurement_data(ear_tag, measurements_data):
    """
    发送体尺数据到API接口

    参数:
    - ear_tag: 耳标号（RFID）
    - measurements_data: 包含完整体尺数据的字典
    """
    # 构建API请求数据
    api_data = {
        "EarTag": ear_tag,
        "Weight": measurements_data.get("Weight", 0),
        "BodyHeight": measurements_data.get("BodyHeight", 0),
        "ChestAround": measurements_data.get("ChestAround", 0),
        "BellyAround": measurements_data.get("BellyAround", 0),
        "BodyDiagonal": measurements_data.get("BodyDiagonal", 0),
        "BodyLength": measurements_data.get("BodyLength", 0),
        "CrossHeight": measurements_data.get("CrossHeight", 0),
        "AddTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # 发送到两个API接口
    for api_url in API_URLS:
        try:
            response = requests.post(
                api_url,
                json=api_data,
                headers={"Content-Type": "application/json"},
                timeout=API_TIMEOUT
            )
            if response.status_code == 200:
                logger.info(f"成功发送数据到 {api_url}: {ear_tag}")
            else:
                logger.warning(f"发送数据到 {api_url} 失败，状态码: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.error(f"发送数据到 {api_url} 超时")
        except Exception as e:
            logger.error(f"发送数据到 {api_url} 时发生错误: {e}")


def estimate_weight_from_measurements(body_height, chest_girth, body_length):
    """
    根据体尺数据估算体重
    使用公式: 体重(kg) ≈ 胸围(cm)² × 体斜长(cm) / 10800
    如果数据不完整，使用简化公式
    """
    if chest_girth and body_length:
        # 完整公式
        weight = (chest_girth ** 2) * body_length / 10800
        return round(weight, 1)
    elif chest_girth and body_height:
        # 简化公式：使用胸围和体高
        weight = (chest_girth ** 2) * body_height / 10000
        return round(weight, 1)
    elif chest_girth:
        # 仅使用胸围的简化估算
        weight = (chest_girth ** 2) * 1.5
        return round(weight, 1)
    return None


# ===================== Gallery 管理 =====================
class LetterBox:
    def __init__(self, size, fill=0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.fill = fill

    def __call__(self, img):
        width, height = img.size
        target_width, target_height = self.size
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
        new_img = Image.new('RGB', self.size, self.fill)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img


class GalleryManager:
    def __init__(self, gallery_path, reid_model, device, max_images=5):
        self.gallery_path = Path(gallery_path)
        self.reid_model = reid_model
        self.device = device
        self.max_images = max_images
        self.gallery_features = []
        self.gallery_ids = []
        self.id_names = {}  # gid -> 真实ID的映射
        self.virtual_to_gid = {}  # 虚拟ID后缀 -> gid的映射
        self.transform = transforms.Compose([
            LetterBox(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self._load_gallery()

    def _load_gallery(self):
        if not self.gallery_path.exists():
            logger.error(f"Gallery路径不存在: {self.gallery_path}")
            return
        logger.info(f"加载Gallery: {self.gallery_path}")
        individuals = [f for f in self.gallery_path.iterdir() if f.is_dir()]
        for idx, folder in enumerate(sorted(individuals)):
            virtual_id_suffix = folder.name  # gallery文件夹名是虚拟ID后缀
            real_id = virtual_to_real_id(virtual_id_suffix)  # 转换为真实ID

            imgs = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.bmp']:
                imgs.extend(folder.glob(ext))
            imgs = sorted(imgs)[:self.max_images]
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert('RGB')
                    feat = self._extract_feature(img)
                    self.gallery_features.append(feat)
                    self.gallery_ids.append(idx)
                    self.id_names[idx] = real_id  # 存储真实ID
                    self.virtual_to_gid[virtual_id_suffix] = idx
                except Exception as e:
                    logger.error(f"加载 {img_path} 失败: {e}")

            logger.info(f"Gallery: 虚拟ID {virtual_id_suffix} -> 真实ID {real_id}")

        logger.info(f"Gallery加载完成: {len(individuals)}个体, {len(self.gallery_features)}特征")

    def _extract_feature(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.reid_model(x).cpu().numpy().flatten()
        f = f / (np.linalg.norm(f) + 1e-8)
        return torch.from_numpy(f)

    def match(self, query_feats, threshold):
        if not self.gallery_features or not query_feats:
            return [], []
        q = torch.stack(query_feats)
        g = torch.stack(self.gallery_features)
        sim = torch.nn.functional.cosine_similarity(q.unsqueeze(1), g.unsqueeze(0), dim=2)
        max_sim, idx = sim.max(dim=1)
        matched_ids, confs = [], []
        for i, s in enumerate(max_sim):
            if s > threshold:
                gid = self.gallery_ids[idx[i]]
                matched_ids.append(gid)
            else:
                matched_ids.append(-1)
            confs.append(s.item())
        return matched_ids, confs

    def get_name(self, gid):
        """返回真实ID"""
        return self.id_names.get(gid, f"Unknown_{gid}")


# ===================== 辅助函数 =====================
def batch_extract_features(crops, model, device, transform):
    if not crops: return []
    batch_tensors = []
    for img in crops:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = transform(img)
        batch_tensors.append(t)
    x = torch.stack(batch_tensors).to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), REID_BATCH_SIZE):
            part = model(x[i:i + REID_BATCH_SIZE])
            f = part.cpu().numpy()
            f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
            feats.append(f)
    feats = np.vstack(feats)
    return [torch.from_numpy(f) for f in feats]


def generate_measurements_with_noise(base_measurements):
    """
    为基础体尺数据添加小的随机噪声
    """
    noisy_measurements = []
    for val in base_measurements:
        if val is not None:
            noise = np.random.uniform(-MEASUREMENT_NOISE_RATIO, MEASUREMENT_NOISE_RATIO)
            noisy_val = val * (1 + noise)
            noisy_measurements.append(round(noisy_val, 1))
        else:
            noisy_measurements.append(None)
    return noisy_measurements


# ===================== 主系统类 =====================
class CowReIDSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        self._init_models()
        self._init_gallery()
        self._init_data_source()
        self._init_video_writer()

        # ---- 追踪与识别状态管理 ----
        self.track_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
        self.track_vote_buffer = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))

        # ---- ID 锁定与稳定性核心状态 ----
        self.track_locked_id = {}
        self.track_high_conf_count = defaultdict(int)
        self.track_no_match_frames = defaultdict(int)
        self.track_disappeared_frames = defaultdict(int)

        # 跟踪丢失管理
        self.track_lost_frames = defaultdict(int)
        self.track_last_box = {}
        self.track_last_gid = {}
        self.track_last_sim = {}

        # 体尺数据管理（只存储体重）
        self.track_weight = {}  # 存储每个track的体重数据（锁定后固定）
        self.track_full_measurements = {}  # 存储完整的体尺数据（用于API发送）
        self.sent_measurements = set()  # 记录已发送的ID，避免重复发送

        # 体尺数据延迟消失管理
        self.last_displayed_weight = None  # 最后显示的体重
        self.last_displayed_cow_name = None  # 最后显示的牛名称
        self.measurement_display_counter = 0  # 体尺数据显示计数器（ID消失后开始计数）
        self.measurement_persist_frames = 80  # ID消失后体尺数据继续显示的帧数

        self.transform = transforms.Compose([
            LetterBox(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.fps_start, self.fps_count, self.fps = time.time(), 0, 0
        self.active_track_ids = set()
        self.frame_count = 0

    def _init_models(self):
        logger.info("初始化检测与识别模型")
        self.detector = YOLO(YOLO_MODEL_PATH)
        self.reid_model = CowReIDModel('MegaDescriptor-S-224', use_lightweight=False)
        try:
            ckpt = torch.load(REID_MODEL_PATH, map_location='cpu', weights_only=False)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.reid_model.load_state_dict(state, strict=False)
            self.reid_model = self.reid_model.to(self.device).eval()
            logger.info(f"ReID模型加载成功: {REID_MODEL_PATH}")
        except Exception as e:
            logger.error(f"加载ReID模型失败: {e}")
            raise

    def _init_gallery(self):
        self.gallery = GalleryManager(GALLERY_PATH, self.reid_model, self.device, MAX_GALLERY_IMAGES)

    def _init_data_source(self):
        """初始化视频源：支持RTSP/HTTP URL或本地视频文件"""
        if USE_CAMERA_URL:
            logger.info(f"正在连接海康威视摄像头: {CAMERA_URL}")
            # 使用OpenCV打开RTSP/HTTP流
            # 设置缓冲区大小和超时参数以提高RTSP流的稳定性
            self.cap = cv2.VideoCapture(CAMERA_URL)
            
            # 设置RTSP流参数
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少延迟
            
            # 尝试读取一帧以验证连接
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"无法连接到摄像头: {CAMERA_URL}")
                logger.error("请检查:")
                logger.error("1. 摄像头IP地址是否正确")
                logger.error("2. 用户名和密码是否正确")
                logger.error("3. 网络连接是否正常")
                logger.error("4. RTSP端口是否开放（默认554）")
                raise RuntimeError(f"无法连接到摄像头: {CAMERA_URL}")
            
            logger.info("摄像头连接成功")
            # 获取视频属性
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or DISPLAY_WIDTH
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DISPLAY_HEIGHT
            
            # 如果无法获取尺寸，使用第一帧的尺寸
            if self.video_width == 0 or self.video_height == 0:
                self.video_height, self.video_width, _ = frame.shape
                logger.info(f"从第一帧获取视频尺寸: {self.video_width}x{self.video_height}")
        else:
            # 使用本地视频文件
            logger.info(f"使用本地视频文件: {VIDEO_PATH}")
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频 {VIDEO_PATH}")
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _init_video_writer(self):
        self.video_writer = None
        if SAVE_VIDEO:
            os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            source_name = "hikvision_camera" if USE_CAMERA_URL else Path(VIDEO_PATH).stem
            output_path = os.path.join(OUTPUT_VIDEO_DIR, f"output_{source_name}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.video_fps,
                                                (self.video_width, self.video_height))
            logger.info(f"视频将保存到: {output_path}")

    def _is_edge_box(self, box, frame_width):
        x1, _, x2, _ = box
        box_center_x = (x1 + x2) / 2
        left_edge = frame_width * EDGE_FILTER_RATIO
        right_edge = frame_width * (1 - EDGE_FILTER_RATIO)
        is_at_edge = box_center_x < left_edge or box_center_x > right_edge
        box_width = x2 - x1
        is_too_small = box_width < frame_width * MIN_BOX_WIDTH_RATIO
        return is_at_edge or is_too_small

    def _is_center_box(self, box, frame_width):
        """检查box是否在画面中间区域"""
        x1, _, x2, _ = box
        box_center_x = (x1 + x2) / 2
        left_boundary = frame_width * CENTER_REGION_RATIO
        right_boundary = frame_width * (1 - CENTER_REGION_RATIO)
        return left_boundary <= box_center_x <= right_boundary

    def _is_valid_box(self, box):
        """检查box是否有效（面积足够大）"""
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        return area >= MIN_BOX_AREA

    def _get_voted_label(self, vote_deque):
        """从投票缓冲中选出最可能的ID及其平均置信度"""
        if not vote_deque or len(vote_deque) < MIN_VOTE_SAMPLES:
            return None, 0.0

        votes = defaultdict(list)
        for gid, sim in vote_deque:
            if gid != -1:
                votes[gid].append(sim)

        if not votes: return None, 0.0

        best_id = max(votes, key=lambda k: len(votes[k]))
        avg_sim = np.mean(votes[best_id])
        return best_id, avg_sim

    def _update_tracker_disappearance(self):
        """管理消失的追踪ID"""
        disappeared_ids = set(self.track_locked_id.keys()) - self.active_track_ids
        for tid in disappeared_ids:
            self.track_disappeared_frames[tid] += 1
            if self.track_disappeared_frames[tid] > LOCKED_ID_DECAY:
                logger.info(
                    f"Track {tid} (ID: {self.gallery.get_name(self.track_locked_id.get(tid))}) 消失时间过长, 清除状态.")
                self._clear_track_state(tid)

    def _clear_track_state(self, tid):
        """清除track的所有状态"""
        self.track_locked_id.pop(tid, None)
        self.track_vote_buffer.pop(tid, None)
        self.track_high_conf_count.pop(tid, None)
        self.track_no_match_frames.pop(tid, None)
        self.track_disappeared_frames.pop(tid, None)
        self.track_history.pop(tid, None)
        self.track_lost_frames.pop(tid, None)
        self.track_last_box.pop(tid, None)
        self.track_last_gid.pop(tid, None)
        self.track_last_sim.pop(tid, None)
        self.track_weight.pop(tid, None)
        self.track_full_measurements.pop(tid, None)

    def _get_or_generate_weight(self, tid, real_cow_id):
        """
        获取体重数据
        从cow_body_measurements.py文件中读取体重数据（如果存在）
        如果track已有数据则返回，否则从文件中读取
        使用真实ID查询体尺数据

        注意：为兼容识别系统V3+版本，数据列表约定为
        [Body Height, Body Length, Chest Girth, Cannon Circumference, Cross Height, Weight]
        即体重放在最后一项；若尚未提供体重，则仅有前5项。

        返回: (体重值, 完整的体尺数据字典)
        """
        if tid in self.track_weight:
            return self.track_weight[tid], self.track_full_measurements[tid]

        # 使用真实ID查找该牛的基础体尺数据
        if real_cow_id in COW_BODY_MEASUREMENTS:
            base_measurements = COW_BODY_MEASUREMENTS[real_cow_id]
            
            weight = None
            if len(base_measurements) >= 6:
                body_measurements = base_measurements[:5]
                weight = base_measurements[5]
            else:
                body_measurements = base_measurements[:5]
            
            # body_measurements格式: [Body Height, Body Length, Chest Girth, Cannon Circumference, Cross Height]
            # 索引: 0=体高, 1=体斜长, 2=胸围, 3=管围, 4=十字部高
            
            # 生成带噪声的完整数据（只对体尺数据添加噪声，体重数据不添加噪声）
            noisy_measurements = generate_measurements_with_noise(body_measurements)
            
            # 提取体尺数据
            body_height = noisy_measurements[0] if len(noisy_measurements) > 0 else None
            body_length = noisy_measurements[1] if len(noisy_measurements) > 1 else None
            chest_girth = noisy_measurements[2] if len(noisy_measurements) > 2 else None
            
            # 完整数据字典（用于API）
            full_data_dict = {
                "Weight": weight if weight is not None else 0,
                "BodyHeight": body_height if body_height else 0,
                "ChestAround": chest_girth if chest_girth else 0,
                "BellyAround": 0,  # cow_body_measurements.py中没有腹围数据
                "BodyDiagonal": body_length if body_length else 0,
                "BodyLength": body_length if body_length else 0,
                "CrossHeight": noisy_measurements[4] if len(noisy_measurements) > 4 else 0,
            }

            self.track_weight[tid] = weight
            self.track_full_measurements[tid] = full_data_dict

            if weight is not None:
                logger.info(f"为Track {tid} (真实ID: {real_cow_id}) 从文件读取体重数据: {weight} kg")
            else:
                logger.info(f"为Track {tid} (真实ID: {real_cow_id}) 未找到体重数据")

            # 如果使用摄像头且该ID未发送过数据，则发送到API
            if USE_CAMERA_URL and real_cow_id not in self.sent_measurements:
                send_measurement_data(real_cow_id, full_data_dict)
                self.sent_measurements.add(real_cow_id)

            return weight, full_data_dict

        return None, None

    def _draw_top_measurements(self, frame, cow_name=None, weight=None):
        """
        在画面顶部绘制体重数据标签（仅显示体重）
        指标名称常驻，数值有条件显示
        """
        # 绿色文字
        text_color = (0, 255, 0)

        # 体重数据 - 居中显示
        frame_width = frame.shape[1]
        y_offset = 50  # 起始位置（增大以更突出）

        # 居中显示
        label_text = "Weight:"
        if weight is not None:
            measurement_text = f"{label_text} {weight:>6.1f} kg"
        else:
            measurement_text = f"{label_text}"

        # 计算文本位置（居中）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(measurement_text, font, MEASUREMENT_FONT_SCALE, MEASUREMENT_THICKNESS)[0]
        x_pos = (frame_width - text_size[0]) // 2

        # 绘制文本（不带背景，直接绘制文字）
        cv2.putText(frame, measurement_text, (x_pos, y_offset),
                    font, MEASUREMENT_FONT_SCALE, text_color, MEASUREMENT_THICKNESS)

    def process_frame(self, frame):
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]

        track_kwargs = {
            'persist': True,
            'verbose': False,
            'conf': CONFIDENCE_THRESH,
            'iou': IOU_THRESH
        }

        if USE_CLASS_FILTER:
            track_kwargs['classes'] = DETECT_CLASSES

        results = self.detector.track(frame, **track_kwargs)[0]

        boxes = results.boxes.xyxy.cpu() if results.boxes is not None else torch.empty(0, 4)
        track_ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else []

        self.active_track_ids = set(track_ids)
        self._update_tracker_disappearance()

        # 过滤有效检测
        crops, valid_indices = [], []
        for i, box in enumerate(boxes):
            if not self._is_valid_box(box):
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)

        current_frame_matched_ids, current_frame_sims = [], []
        if crops:
            feats = batch_extract_features(crops, self.reid_model, self.device, self.transform)
            current_frame_matched_ids, current_frame_sims = self.gallery.match(feats, SIMILARITY_THRESH)

        annotated_frame = frame.copy()

        match_map = {valid_indices[i]: (current_frame_matched_ids[i], current_frame_sims[i]) for i in
                     range(len(valid_indices))}

        # <--- 用于存储中间区域的体重信息（只显示一个）
        center_cow_name = None
        center_weight = None

        # 处理当前帧的检测
        for i, (box, tid) in enumerate(zip(boxes, track_ids)):
            if not self._is_valid_box(box):
                continue

            # 重置lost计数
            self.track_lost_frames[tid] = 0
            self.track_last_box[tid] = box

            is_edge = self._is_edge_box(box, frame_width)
            is_center = self._is_center_box(box, frame_width)

            if is_edge:
                self.track_vote_buffer[tid].clear()
                self.track_high_conf_count[tid] = 0
                # 不显示edge检测框
                continue
            else:
                gid, sim = match_map.get(i, (-1, 0.0))
                self.track_vote_buffer[tid].append((gid, sim))

                is_initial_frame = self.frame_count <= INITIAL_FRAMES_SKIP
                required_conf = INITIAL_HIGH_CONF_THRESH if is_initial_frame else HIGH_CONF_THRESH

                # ID锁定逻辑
                if tid in self.track_locked_id:
                    locked_gid = self.track_locked_id[tid]

                    if gid == -1:
                        self.track_no_match_frames[tid] += 1
                    else:
                        self.track_no_match_frames[tid] = 0

                    if self.track_no_match_frames[tid] > UNLOCK_REQUIRE_FRAMES:
                        logger.warning(f"Track {tid} ID {self.gallery.get_name(locked_gid)} 解锁 (连续未匹配)")
                        self.track_locked_id.pop(tid, None)
                        self.track_high_conf_count[tid] = 0
                        self.track_weight.pop(tid, None)
                        self.track_full_measurements.pop(tid, None)
                        final_gid, final_sim = self._get_voted_label(self.track_vote_buffer[tid])
                    else:
                        final_gid = locked_gid
                        _, final_sim = self._get_voted_label(self.track_vote_buffer[tid])

                    if final_gid is not None:
                        self.track_last_gid[tid] = final_gid
                        self.track_last_sim[tid] = final_sim
                else:
                    final_gid, final_sim = self._get_voted_label(self.track_vote_buffer[tid])

                    if final_gid is not None and final_sim >= required_conf:
                        self.track_high_conf_count[tid] += 1
                    else:
                        self.track_high_conf_count[tid] = 0

                    if not is_initial_frame and self.track_high_conf_count[tid] >= LOCK_FRAME_COUNT:
                        self.track_locked_id[tid] = final_gid
                        self.track_no_match_frames[tid] = 0
                        logger.info(
                            f"Track {tid} ID锁定为 -> {self.gallery.get_name(final_gid)} (置信度: {final_sim:.3f})")

                    if final_gid is not None:
                        self.track_last_gid[tid] = final_gid
                        self.track_last_sim[tid] = final_sim

                # 只绘制锁定的ID
                if tid in self.track_locked_id:
                    final_gid = self.track_locked_id[tid]
                    real_id = self.gallery.get_name(final_gid)  # 获取真实ID
                    color = LOCKED_COLOR  # 统一红色

                    # 获取或生成体重数据（使用真实ID）
                    weight, _ = self._get_or_generate_weight(tid, real_id)

                    # <--- 如果在中间区域且还没有记录体重信息，则记录
                    if is_center and center_cow_name is None and weight is not None:
                        center_cow_name = real_id
                        center_weight = weight

                    # 构建标签文本（显示真实ID）
                    label_text = f"ID: {real_id}"

                    # 绘制检测框
                    thickness = 3
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    # 绘制ID标签（增大字体和粗细）
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, ID_LABEL_THICKNESS)

        # <--- 体重数据延迟消失逻辑
        # 如果有中间区域的牛，更新显示并重置计数器
        if center_cow_name and center_weight is not None:
            self.last_displayed_weight = center_weight
            self.last_displayed_cow_name = center_cow_name
            self.measurement_display_counter = 0  # 重置计数器
            self._draw_top_measurements(annotated_frame, center_cow_name, center_weight)
        else:
            # 没有中间区域的牛，检查是否需要继续显示上次的数据
            if self.last_displayed_weight is not None:
                if self.measurement_display_counter < self.measurement_persist_frames:
                    # 继续显示上次的数据
                    self._draw_top_measurements(annotated_frame, self.last_displayed_cow_name,
                                                self.last_displayed_weight)
                    self.measurement_display_counter += 1
                else:
                    # 超过指定帧数，清除显示
                    self._draw_top_measurements(annotated_frame)  # 只显示标签名称
            else:
                # 从未显示过体重数据，只显示标签名称
                self._draw_top_measurements(annotated_frame)

        return annotated_frame

    def run(self):
        # 在打开识别窗口前，先最小化所有窗口返回桌面
        if USE_CAMERA_URL:
            minimize_all_windows()

        cv2.namedWindow("Cow ReID System - Hikvision Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cow ReID System - Hikvision Camera", DISPLAY_WIDTH, DISPLAY_HEIGHT)

        while True:
            success, frame = self.cap.read()
            if not success:
                logger.warning("视频流中断或无法获取帧，尝试重新连接...")
                # 如果是RTSP流，尝试重新连接
                if USE_CAMERA_URL:
                    time.sleep(2)  # 等待2秒后重试
                    self.cap.release()
                    self._init_data_source()
                    continue
                else:
                    logger.warning("视频结束或无法获取帧")
                    break

            frame = cv2.resize(frame, (self.video_width, self.video_height))
            annotated = self.process_frame(frame)

            if self.video_writer is not None:
                self.video_writer.write(annotated)

            cv2.imshow("Cow ReID System - Hikvision Camera", annotated)

            if self.frame_count % 100 == 0:
                logger.info(f"已处理 {self.frame_count} 帧")

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break

        self.cleanup()

    def cleanup(self):
        if hasattr(self.cap, 'release'): self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            logger.info("视频保存完成")
        cv2.destroyAllWindows()
        logger.info("系统关闭")


# ===================== 入口 =====================
if __name__ == "__main__":
    try:
        system = CowReIDSystem()
        system.run()
    except Exception as e:
        logger.critical(f"系统启动或运行期间发生致命错误: {e}", exc_info=True)

