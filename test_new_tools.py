#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("测试新的工具包装器...")

# 测试各个工具的直接导入
print("\n1. 测试 YOLOTool 导入...")
try:
    from tools.tool_yolo import YOLOTool
    print("✓ YOLOTool 导入成功")
except Exception as e:
    print(f"✗ YOLOTool 导入失败: {e}")

print("\n2. 测试 KPDTool 导入...")
try:
    from tools.tool_kpd import KPDTool
    print("✓ KPDTool 导入成功")
except Exception as e:
    print(f"✗ KPDTool 导入失败: {e}")

print("\n3. 测试 OCRTool 导入...")
try:
    from tools.tool_ocr import OCRTool
    print("✓ OCRTool 导入成功")
except Exception as e:
    print(f"✗ OCRTool 导入失败: {e}")

print("\n4. 测试 AuxilineTool 导入...")
try:
    from tools.tool_auxiline import AuxilineTool
    print("✓ AuxilineTool 导入成功")
except Exception as e:
    print(f"✗ AuxilineTool 导入失败: {e}")

print("\n5. 测试 ToolManager 导入和初始化...")
try:
    from tools import ToolManager, tool_manager
    print("✓ ToolManager 导入成功")
    
    tm = ToolManager()
    print("✓ ToolManager 实例创建成功")
    
    tm.initialize()
    print("✓ ToolManager 初始化成功")
    
    print(f"\n工具状态:")
    print(f"  OCR 工具: {tm.ocr_tool is not None}")
    print(f"  KPD 轴工具: {tm.kpd_axis is not None}")
    print(f"  KPD 饼图工具: {tm.kpd_pie is not None}")
    print(f"  垂直柱工具: {tm.seg_vertical_bar is not None}")
    print(f"  水平条工具: {tm.seg_horizontal_bar is not None}")
    print(f"  Auxiline 工具: {tm.auxiline_tool is not None}")
    
except Exception as e:
    print(f"✗ ToolManager 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成！")