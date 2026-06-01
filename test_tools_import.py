#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("测试 CPAgent_adk/tools 导入...")
try:
    from tools import ToolManager, tool_manager
    print("✓ ToolManager 导入成功")
    
    # 测试工具初始化
    tm = ToolManager()
    print("ToolManager 实例创建成功")
    
    # 尝试初始化工具（可能会加载模型）
    print("尝试初始化工具...")
    tm.initialize()
    print("工具初始化完成")
    
    # 检查各个工具是否已加载
    print(f"OCR 工具: {tm.ocr_tool is not None}")
    print(f"KPD 轴工具: {tm.kpd_axis is not None}")
    print(f"KPD 饼图工具: {tm.kpd_pie is not None}")
    print(f"垂直柱工具: {tm.seg_vertical_bar is not None}")
    print(f"水平条工具: {tm.seg_horizontal_bar is not None}")
    print(f"Auxiline 工具: {tm.auxiline_tool is not None}")
    
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()