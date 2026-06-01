import sys
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from config import CPAGENT_TOOLS_DIR

# 尝试从 CPAgent 导入原始 OCRTool
try:
    sys.path.append(str(CPAGENT_TOOLS_DIR))
    from tool_ocr import OCRTool as BaseOCRTool
    HAS_OCR = True
except ImportError as e:
    print(f"[CPAgent_adk] 无法导入原始 OCRTool: {e}")
    HAS_OCR = False
except Exception as e:
    print(f"[CPAgent_adk] OCRTool 导入过程中发生错误: {e}")
    HAS_OCR = False

class OCRTool:
    """CPAgent_adk 的 OCR 工具包装器"""
    
    def __init__(self):
        self.ocr = None
        self._initialized = False
        
    def _try_initialize(self):
        """尝试初始化 OCR 引擎"""
        if self._initialized:
            return
            
        if HAS_OCR:
            try:
                self.ocr = BaseOCRTool()
                self._initialized = True
                print("[CPAgent_adk.OCRTool] 使用原始 OCRTool")
                return
            except Exception as e:
                print(f"[CPAgent_adk.OCRTool] 原始 OCRTool 初始化失败: {e}")
        
        # 备用方案：提供一个简单的模拟 OCR
        print("[CPAgent_adk.OCRTool] 使用模拟 OCR 实现")
        self.ocr = None
        self._initialized = True
    
    def text_detect(self, image_path: str) -> List[Dict[str, Any]]:
        """OCR 文本检测
        
        Args:
            image_path: 图像路径
            
        Returns:
            包含文本和边界框的字典列表
        """
        self._try_initialize()
        
        # 如果原始 OCR 可用，使用它
        if self.ocr is not None and HAS_OCR:
            try:
                return self.ocr.text_detect(image_path)
            except Exception as e:
                print(f"[CPAgent_adk.OCRTool] OCR 检测失败: {e}")
                # 回退到模拟实现
        
        # 模拟实现：返回空列表或简单检测
        print(f"[CPAgent_adk.OCRTool] 模拟 OCR 检测: {image_path}")
        
        # 这里可以添加简单的 OpenCV 文本检测作为备用
        # 但为了简化，先返回空列表
        return []
    
    def extract_axis_labels(self, ocr_results: List[Dict[str, Any]], image_size: tuple) -> Dict[str, List]:
        """提取坐标轴标签（模拟实现）
        
        Args:
            ocr_results: OCR 检测结果
            image_size: (width, height)
            
        Returns:
            包含 x_axis 和 y_axis 标签的字典
        """
        # 简单模拟：根据位置判断轴标签
        x_axis = []
        y_axis = []
        
        for result in ocr_results:
            box = result['box']  # [cx, cy, w, h]
            text = result['text']
            
            # 简单启发式：靠近底部的可能是 x 轴，靠近左侧的可能是 y 轴
            height = image_size[1]
            width = image_size[0]
            
            if box[1] > height * 0.7:  # cy 靠近底部
                x_axis.append({'box': box, 'text': text})
            elif box[0] < width * 0.3:  # cx 靠近左侧
                y_axis.append({'box': box, 'text': text})
        
        return {"x_axis": x_axis, "y_axis": y_axis}

if __name__ == "__main__":
    # 简单测试
    tool = OCRTool()
    print(f"工具初始化: {tool._initialized}")
    print(f"HAS_OCR: {HAS_OCR}")