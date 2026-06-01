#!/usr/bin/env python3
"""测试 ToolManager 基本功能"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from config import TEST_DATA_DIR
from tools import tool_manager

def test_ocr():
    """测试OCR检测"""
    print("测试OCR检测...")
    image_path = str(TEST_DATA_DIR / "82856411003183.png")
    if not os.path.exists(image_path):
        print(f"图像文件不存在: {image_path}")
        return
    try:
        result = tool_manager.ocr_detect(image_path)
        print(f"OCR检测成功，返回 {len(result)} 个文本")
        for i, item in enumerate(result[:3]):  # 只打印前三个
            print(f"  [{i}] 文本: {item.get('text', 'N/A')}, 边界框: {item.get('box', 'N/A')}")
    except Exception as e:
        print(f"OCR检测失败: {e}")

def test_axis_detect():
    """测试坐标轴关键点检测"""
    print("\n测试坐标轴关键点检测...")
    image_path = str(TEST_DATA_DIR / "82856411003183.png")
    try:
        x_candidates, y_candidates, temp_path = tool_manager.axis_detect(image_path, "/tmp")
        print(f"检测到 X轴候选点: {len(x_candidates)} 个")
        print(f"检测到 Y轴候选点: {len(y_candidates)} 个")
        print(f"可视化结果保存至: {temp_path}")
    except Exception as e:
        print(f"坐标轴检测失败: {e}")

def test_tool_manager_init():
    """测试工具管理器初始化"""
    print("测试工具管理器初始化...")
    try:
        tool_manager.initialize()
        print("工具管理器初始化成功")
        print(f"OCR工具: {tool_manager.ocr_tool is not None}")
        print(f"KPD轴工具: {tool_manager.kpd_axis is not None}")
        print(f"KPD饼图工具: {tool_manager.kpd_pie is not None}")
        print(f"YOLO垂直柱工具: {tool_manager.seg_vertical_bar is not None}")
        print(f"YOLO水平条工具: {tool_manager.seg_horizontal_bar is not None}")
        print(f"Auxiline工具: {tool_manager.auxiline_tool is not None}")
    except Exception as e:
        print(f"工具管理器初始化失败: {e}")

if __name__ == "__main__":
    test_tool_manager_init()
    test_ocr()
    test_axis_detect()