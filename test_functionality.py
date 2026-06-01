#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("测试工具功能...")

from config import TEST_DATA_DIR
from tools import tool_manager

# 使用第一个可用的图像
image_dir = str(TEST_DATA_DIR)
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print("没有找到测试图像")
    sys.exit(0)

test_image = os.path.join(image_dir, image_files[0])
print(f"使用测试图像: {test_image}")

print("\n1. 测试 OCR 检测...")
try:
    results = tool_manager.ocr_detect(test_image)
    print(f"OCR 结果数量: {len(results)}")
    for i, r in enumerate(results[:3]):  # 显示前三个结果
        print(f"  结果 {i}: 文本='{r.get('text', '')}', 框={r.get('box', [])}")
except Exception as e:
    print(f"OCR 检测失败: {e}")

print("\n2. 测试坐标轴检测...")
try:
    x_candidates, y_candidates, temp_path = tool_manager.axis_detect(test_image, "/tmp/test_output")
    print(f"X 轴候选点: {len(x_candidates)}")
    print(f"Y 轴候选点: {len(y_candidates)}")
    print(f"临时路径: {temp_path}")
except Exception as e:
    print(f"坐标轴检测失败: {e}")

print("\n3. 测试饼图检测...")
try:
    labels, output_path = tool_manager.pie_detect(test_image, "/tmp/test_output")
    print(f"饼图标签: {labels}")
    print(f"输出路径: {output_path}")
except Exception as e:
    print(f"饼图检测失败: {e}")

print("\n4. 测试柱状图检测...")
try:
    # 假设是垂直柱状图
    mock_category_coords = [100, 200, 300, 400, 500]
    def mock_value_mapper(x):
        return x * 0.1
    
    results, output_path = tool_manager.bar_detect(
        test_image, 
        "vertical_bar",
        mock_category_coords,
        "/tmp/test_output",
        mock_value_mapper
    )
    print(f"柱状图结果: {results}")
    print(f"输出路径: {output_path}")
except Exception as e:
    print(f"柱状图检测失败: {e}")

print("\n5. 测试辅助线检测...")
try:
    mock_x_ticks = [100, 200, 300, 400]
    mock_text_bboxes = [[50, 50, 100, 100], [150, 50, 100, 100]]
    def mock_pixel_to_value(pixel):
        return pixel * 0.5
    
    results, output_path = tool_manager.auxiline_detect(
        test_image,
        mock_x_ticks,
        mock_text_bboxes,
        mock_pixel_to_value,
        "/tmp/test_output/auxiline.png"
    )
    print(f"辅助线结果: {results}")
    print(f"输出路径: {output_path}")
except Exception as e:
    print(f"辅助线检测失败: {e}")

print("\n测试完成！")