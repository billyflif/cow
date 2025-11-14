#!/usr/bin/env python3
"""
相机标定参数管理
用于体尺测量的相机内参获取和管理
"""

import json
import numpy as np
from pyorbbecsdk import *


class CameraCalibration:
    """管理相机标定参数"""
    
    def __init__(self):
        self.calibration_data = None
        
    def get_camera_intrinsics(self, device, sensor_type=OBSensorType.DEPTH_SENSOR):
        """
        从设备获取相机内参
        
        Args:
            device: Orbbec设备对象
            sensor_type: 传感器类型（深度或彩色）
            
        Returns:
            dict: 包含相机内参的字典
        """
        try:
            # 获取相机内参
            if hasattr(device, 'get_calibration_camera_params'):
                params = device.get_calibration_camera_params()
                
                if sensor_type == OBSensorType.DEPTH_SENSOR:
                    intrinsics = params.depth_intrinsic
                else:
                    intrinsics = params.rgb_intrinsic
                
                return {
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'cx': intrinsics.cx,
                    'cy': intrinsics.cy,
                    'width': intrinsics.width,
                    'height': intrinsics.height,
                    'k1': intrinsics.k1,
                    'k2': intrinsics.k2,
                    'k3': intrinsics.k3,
                    'k4': intrinsics.k4,
                    'k5': intrinsics.k5,
                    'k6': intrinsics.k6,
                    'p1': intrinsics.p1,
                    'p2': intrinsics.p2
                }
            else:
                # 如果SDK不支持，返回默认值
                print("Warning: Camera intrinsics not available from device, using defaults")
                return self.get_default_intrinsics(sensor_type)
                
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
            return self.get_default_intrinsics(sensor_type)
    
    def get_default_intrinsics(self, sensor_type):
        """
        获取默认相机内参（典型值）
        实际应用中应该进行相机标定获取准确值
        """
        if sensor_type == OBSensorType.DEPTH_SENSOR:
            return {
                'fx': 570.0,
                'fy': 570.0,
                'cx': 320.0,
                'cy': 240.0,
                'width': 640,
                'height': 480,
                'k1': 0.0, 'k2': 0.0, 'k3': 0.0,
                'k4': 0.0, 'k5': 0.0, 'k6': 0.0,
                'p1': 0.0, 'p2': 0.0
            }
        else:  # RGB
            return {
                'fx': 520.0,
                'fy': 520.0,
                'cx': 320.0,
                'cy': 240.0,
                'width': 640,
                'height': 480,
                'k1': 0.0, 'k2': 0.0, 'k3': 0.0,
                'k4': 0.0, 'k5': 0.0, 'k6': 0.0,
                'p1': 0.0, 'p2': 0.0
            }
    
    def save_calibration(self, calibration_data, filename="camera_calibration.json"):
        """保存标定数据到文件"""
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename="camera_calibration.json"):
        """从文件加载标定数据"""
        try:
            with open(filename, 'r') as f:
                self.calibration_data = json.load(f)
            print(f"Calibration loaded from {filename}")
            return self.calibration_data
        except FileNotFoundError:
            print(f"Calibration file {filename} not found")
            return None
    
    def pixel_to_3d_point(self, x, y, depth, intrinsics):
        """
        将像素坐标转换为3D点
        
        Args:
            x, y: 像素坐标
            depth: 深度值（毫米）
            intrinsics: 相机内参字典
            
        Returns:
            tuple: (X, Y, Z) 3D坐标（毫米）
        """
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # 应用针孔相机模型
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        
        return (X, Y, Z)
    
    def calculate_real_dimensions(self, roi, depth_data, intrinsics):
        """
        计算ROI区域的真实物理尺寸
        
        Args:
            roi: (x, y, width, height) 像素坐标系中的ROI
            depth_data: 深度图数据
            intrinsics: 相机内参
            
        Returns:
            dict: 物理尺寸信息
        """
        x, y, w, h = roi
        
        # 提取ROI深度数据
        roi_depth = depth_data[y:y+h, x:x+w]
        valid_mask = roi_depth > 0
        
        if not np.any(valid_mask):
            return None
        
        # 获取ROI四个角的3D坐标
        corners_2d = [
            (x, y),           # 左上
            (x + w, y),       # 右上
            (x, y + h),       # 左下
            (x + w, y + h)    # 右下
        ]
        
        corners_3d = []
        for px, py in corners_2d:
            # 使用该点附近的平均深度
            local_depth = self._get_local_depth(depth_data, px, py, window=5)
            if local_depth > 0:
                point_3d = self.pixel_to_3d_point(px, py, local_depth, intrinsics)
                corners_3d.append(point_3d)
        
        if len(corners_3d) < 4:
            # 使用简化方法
            mean_depth = np.mean(roi_depth[valid_mask])
            width_mm = (w * mean_depth) / intrinsics['fx']
            height_mm = (h * mean_depth) / intrinsics['fy']
        else:
            # 计算实际尺寸
            width_mm = np.linalg.norm(np.array(corners_3d[1]) - np.array(corners_3d[0]))
            height_mm = np.linalg.norm(np.array(corners_3d[2]) - np.array(corners_3d[0]))
        
        # 深度范围
        depth_min = np.min(roi_depth[valid_mask])
        depth_max = np.max(roi_depth[valid_mask])
        depth_range = depth_max - depth_min
        
        return {
            'width_mm': width_mm,
            'height_mm': height_mm,
            'depth_range_mm': depth_range,
            'mean_depth_mm': np.mean(roi_depth[valid_mask]),
            'valid_pixels': np.sum(valid_mask),
            'coverage': np.sum(valid_mask) / (w * h)
        }
    
    def _get_local_depth(self, depth_data, x, y, window=5):
        """获取局部区域的平均深度"""
        h, w = depth_data.shape
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))
        
        # 定义窗口边界
        x1 = max(0, x - window // 2)
        x2 = min(w, x + window // 2 + 1)
        y1 = max(0, y - window // 2)
        y2 = min(h, y + window // 2 + 1)
        
        # 获取局部深度
        local_region = depth_data[y1:y2, x1:x2]
        valid_depths = local_region[local_region > 0]
        
        if valid_depths.size > 0:
            return np.median(valid_depths)  # 使用中值减少噪声影响
        else:
            return 0


# 使用示例
if __name__ == "__main__":
    # 创建标定管理器
    calib = CameraCalibration()
    
    # 示例：使用默认内参计算尺寸
    intrinsics = calib.get_default_intrinsics(OBSensorType.DEPTH_SENSOR)
    print("默认深度相机内参：")
    print(f"  焦距: fx={intrinsics['fx']}, fy={intrinsics['fy']}")
    print(f"  主点: cx={intrinsics['cx']}, cy={intrinsics['cy']}")
    
    # 示例：像素到3D转换
    x, y = 320, 240  # 图像中心
    depth = 1000  # 1米
    point_3d = calib.pixel_to_3d_point(x, y, depth, intrinsics)
    print(f"\n像素({x}, {y})在深度{depth}mm处对应的3D点: {point_3d}")
    
    # 保存标定数据
    calib.save_calibration(intrinsics)
