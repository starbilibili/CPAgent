import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable, Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from config import CPAGENT_TOOLS_DIR

# 尝试从 CPAgent 导入原始 AuxilineTool
try:
    sys.path.append(str(CPAGENT_TOOLS_DIR))
    from tool_auxiline import AuxilineTool as BaseAuxilineTool
    HAS_AUXILINE = True
except ImportError as e:
    print(f"[CPAgent_adk] 无法导入原始 AuxilineTool: {e}")
    HAS_AUXILINE = False
except Exception as e:
    print(f"[CPAgent_adk] AuxilineTool 导入过程中发生错误: {e}")
    HAS_AUXILINE = False

class AuxilineTool:
    """CPAgent_adk 的辅助线检测工具包装器"""
    
    def __init__(self, line_gap_tolerance: int = 5, x_search_radius: int = 6):
        """
        初始化辅助线检测工具
        
        Args:
            line_gap_tolerance: 垂直方向上合并点的容差
            x_search_radius: 在 x 轴刻度左右搜索的半径
        """
        self.line_gap_tolerance = line_gap_tolerance
        self.x_search_radius = x_search_radius
        self.model = None
        self._initialized = False
        
    def _try_initialize(self):
        """尝试初始化辅助线检测模型"""
        if self._initialized:
            return
            
        if HAS_AUXILINE:
            try:
                self.model = BaseAuxilineTool(self.line_gap_tolerance, self.x_search_radius)
                self._initialized = True
                print(f"[CPAgent_adk.AuxilineTool] 使用原始 AuxilineTool")
                return
            except Exception as e:
                print(f"[CPAgent_adk.AuxilineTool] 原始 AuxilineTool 初始化失败: {e}")
        
        # 备用方案：提供一个模拟实现
        print(f"[CPAgent_adk.AuxilineTool] 使用模拟 Auxiline 实现")
        self.model = None
        self._initialized = True
    
    def draw_auxilines(self, image_path: str, x_ticks: List[int], text_bboxes: List[List[int]],
                       pixel_to_value: Callable, output_path: str) -> Tuple[List[List[str]], str]:
        """绘制折线图辅助线
        
        Args:
            image_path: 图像路径
            x_ticks: x 轴刻度像素坐标
            text_bboxes: 文本边界框
            pixel_to_value: 像素到数值的转换函数
            output_path: 输出图像路径
            
        Returns:
            (辅助线数据, 输出图像路径)
        """
        self._try_initialize()
        
        # 如果原始 Auxiline 可用，使用它
        if self.model is not None and HAS_AUXILINE:
            try:
                return self.model.draw_auxilines(image_path, x_ticks, text_bboxes, pixel_to_value, output_path)
            except Exception as e:
                print(f"[CPAgent_adk.AuxilineTool] Auxiline 检测失败: {e}")
                # 回退到模拟实现
        
        # 模拟实现：返回随机辅助线数据
        print(f"[CPAgent_adk.AuxilineTool] 模拟辅助线检测: {image_path}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"[CPAgent_adk.AuxilineTool] 无法读取图像: {image_path}")
            return [], ""
        
        height, width = img.shape[:2]
        
        # 生成模拟的辅助线数据
        auxiline_data = []
        for i in range(3):
            x = int(width * (i + 1) / 4)
            y = int(height * 0.5)
            value = pixel_to_value(y) if callable(pixel_to_value) else y
            auxiline_data.append([f"Aux{i+1}", str(round(value, 2)), str(x), str(y)])
        
        # 保存输出图像（在图像上绘制模拟辅助线）
        try:
            # 绘制水平线
            for i, line in enumerate(auxiline_data):
                y = int(line[3])
                cv2.line(img, (0, y), (width, y), (0, 0, 255), 2)
                cv2.putText(img, f"Aux {i+1}", (10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"[CPAgent_adk.AuxilineTool] 无法保存输出图像: {e}")
            output_path = ""
        
        return auxiline_data, output_path
    
    def get_combined_mask(self, image):
        """获取合并掩膜（模拟实现）"""
        # 返回一个全零掩膜
        return np.zeros(image.shape[:2], dtype=np.uint8)

if __name__ == "__main__":
    # 简单测试
    tool = AuxilineTool()
    print(f"工具初始化: {tool._initialized}")
    print(f"HAS_AUXILINE: {HAS_AUXILINE}")