#!/usr/bin/env python3
"""
体尺测量专用示例
演示如何使用同步捕获接口进行精确的物体尺寸测量
"""

import requests
import numpy as np
import cv2
from datetime import datetime


class BodySizeMeasurement:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        
    def measure_static_object(self):
        """静态物体测量 - 使用多帧平均提高精度"""
        print("=== 静态物体测量模式 ===")
        
        # 1. 检查相机状态
        health = requests.get(f"{self.api_url}/health").json()
        if not health.get('camera_running'):
            print("错误：相机未运行")
            return None
        
        # 2. 捕获多帧进行平均
        print("正在捕获10帧深度数据...")
        response = requests.post(
            f"{self.api_url}/capture_sync",
            json={
                "num_frames": 10,
                "capture_type": "depth"  # 只需要深度数据
            }
        )
        
        if not response.json().get('success'):
            print(f"捕获失败: {response.json().get('error')}")
            return None
        
        frames = response.json()['frames']
        print(f"成功捕获 {len(frames)} 帧")
        
        # 3. 加载所有深度数据并计算平均
        depth_arrays = []
        for frame in frames:
            npy_path = frame['depth']['saved_paths']['raw_data_path']
            depth_data = np.load(npy_path)
            depth_arrays.append(depth_data)
        
        # 计算平均深度图
        averaged_depth = np.mean(depth_arrays, axis=0).astype(np.uint16)
        
        # 4. 应用额外的空间滤波
        filtered_depth = cv2.medianBlur(averaged_depth, 5)
        
        print("深度数据处理完成")
        return filtered_depth
    
    def measure_moving_object(self):
        """动态物体测量 - 单帧快速捕获"""
        print("=== 动态物体测量模式 ===")
        
        # 捕获单帧，避免运动模糊
        response = requests.post(
            f"{self.api_url}/capture_sync",
            json={
                "num_frames": 1,
                "capture_type": "depth"
            }
        )
        
        if not response.json().get('success'):
            print(f"捕获失败: {response.json().get('error')}")
            return None
        
        frame = response.json()['frames'][0]
        npy_path = frame['depth']['saved_paths']['raw_data_path']
        depth_data = np.load(npy_path)
        
        print(f"捕获时间: {datetime.fromtimestamp(frame['timestamp'])}")
        print(f"中心点距离: {frame['depth']['statistics']['center_distance']}mm")
        
        return depth_data
    
    def calculate_object_size(self, depth_data, roi=None):
        """
        计算物体尺寸
        
        Args:
            depth_data: 深度图数据
            roi: 感兴趣区域 (x, y, width, height)，None表示自动检测
        
        Returns:
            dict: 包含宽度、高度、深度等测量结果
        """
        if roi is None:
            # 自动检测物体（简单示例：找最大连通区域）
            # 实际应用中应该使用更复杂的物体检测算法
            valid_mask = (depth_data > 0) & (depth_data < 5000)
            
            # 使用形态学操作清理噪声
            kernel = np.ones((5, 5), np.uint8)
            valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # 找轮廓
            contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("未检测到物体")
                return None
            
            # 找最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            roi = (x, y, w, h)
        
        x, y, w, h = roi
        
        # 提取ROI区域的深度数据
        roi_depth = depth_data[y:y+h, x:x+w]
        valid_depths = roi_depth[roi_depth > 0]
        
        if valid_depths.size == 0:
            print("ROI区域无有效深度数据")
            return None
        
        # 计算统计信息
        mean_depth = np.mean(valid_depths)
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        
        # 假设已知相机内参（需要根据实际相机调整）
        # 这里使用典型值，实际应用应该从相机获取
        fx = 570.0  # 焦距x
        fy = 570.0  # 焦距y
        cx = depth_data.shape[1] / 2  # 主点x
        cy = depth_data.shape[0] / 2  # 主点y
        
        # 计算物理尺寸（单位：毫米）
        # 使用平均深度进行计算
        physical_width = (w * mean_depth) / fx
        physical_height = (h * mean_depth) / fy
        physical_depth = max_depth - min_depth
        
        result = {
            'roi': roi,
            'pixel_size': {'width': w, 'height': h},
            'physical_size': {
                'width': physical_width,
                'height': physical_height,
                'depth': physical_depth
            },
            'depth_stats': {
                'mean': mean_depth,
                'min': min_depth,
                'max': max_depth
            },
            'unit': 'mm'
        }
        
        return result
    
    def visualize_measurement(self, depth_data, measurement_result):
        """可视化测量结果"""
        # 创建彩色深度图
        depth_colormap = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_JET)
        
        if measurement_result:
            # 绘制ROI
            x, y, w, h = measurement_result['roi']
            cv2.rectangle(depth_colormap, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 添加测量信息
            size = measurement_result['physical_size']
            text = f"W:{size['width']:.1f} H:{size['height']:.1f} D:{size['depth']:.1f}mm"
            cv2.putText(depth_colormap, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return depth_colormap


def main():
    # 创建测量对象
    measurement = BodySizeMeasurement()
    
    # 示例1：静态物体测量
    print("\n示例1：静态物体精确测量")
    depth_data = measurement.measure_static_object()
    
    if depth_data is not None:
        # 计算物体尺寸
        result = measurement.calculate_object_size(depth_data)
        
        if result:
            print("\n测量结果：")
            print(f"物体尺寸（毫米）：")
            print(f"  宽度: {result['physical_size']['width']:.1f} mm")
            print(f"  高度: {result['physical_size']['height']:.1f} mm")
            print(f"  厚度: {result['physical_size']['depth']:.1f} mm")
            
            # 可视化
            vis_image = measurement.visualize_measurement(depth_data, result)
            cv2.imwrite("measurement_result.png", vis_image)
            print("\n可视化结果已保存到 measurement_result.png")
    
    # 示例2：动态物体测量
    print("\n\n示例2：动态物体快速测量")
    depth_data = measurement.measure_moving_object()
    
    if depth_data is not None:
        # 指定ROI进行测量（实际应用中可能来自物体追踪）
        h, w = depth_data.shape
        roi = (w//4, h//4, w//2, h//2)  # 中心区域
        
        result = measurement.calculate_object_size(depth_data, roi)
        if result:
            print(f"\n指定区域测量结果：")
            print(f"  宽度: {result['physical_size']['width']:.1f} mm")
            print(f"  高度: {result['physical_size']['height']:.1f} mm")


if __name__ == "__main__":
    main()
