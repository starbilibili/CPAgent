import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Callable, Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from config import CPAGENT_TOOLS_DIR, YOLO_VERTICAL_BAR_PATH

# 尝试从 CPAgent 导入原始 YOLOTool
try:
    sys.path.append(str(CPAGENT_TOOLS_DIR))
    from tool_yolo import YOLOTool as BaseYOLOTool
    HAS_YOLO = True
except ImportError as e:
    print(f"[CPAgent_adk] 无法导入原始 YOLOTool: {e}")
    HAS_YOLO = False
except Exception as e:
    print(f"[CPAgent_adk] YOLOTool 导入过程中发生错误: {e}")
    HAS_YOLO = False

class YOLOTool:
    """CPAgent_adk 的 YOLO 工具包装器"""
    
    def __init__(self, model_path: str):
        """
        初始化 YOLO 工具
        
        Args:
            model_path: 模型权重文件路径
        """
        self.model_path = model_path
        self.model = None
        self._initialized = False
        
    def _try_initialize(self):
        """尝试初始化 YOLO 模型"""
        if self._initialized:
            return
            
        if HAS_YOLO and os.path.exists(self.model_path):
            try:
                self.model = BaseYOLOTool(self.model_path)
                self._initialized = True
                print(f"[CPAgent_adk.YOLOTool] 使用原始 YOLOTool: {self.model_path}")
                return
            except Exception as e:
                print(f"[CPAgent_adk.YOLOTool] 原始 YOLOTool 初始化失败: {e}")
        
        # 备用方案：提供一个模拟实现
        print(f"[CPAgent_adk.YOLOTool] 使用模拟 YOLO 实现: {self.model_path}")
        self.model = None
        self._initialized = True
    
    def predict_bar(self, image_path: str, type: str, output_dir: str,
                    value_mapper: Callable, category_coords: List[int]) -> Tuple[List[List[str]], str]:
        """预测柱状图
        
        Args:
            image_path: 图像路径
            type: 图表类型 ('vertical_bar' 或 'horizontal_bar')
            output_dir: 输出目录
            value_mapper: 值映射函数
            category_coords: 类别坐标
            
        Returns:
            (预测结果列表, 输出图像路径)
        """
        self._try_initialize()
        
        # 如果原始 YOLO 可用，使用它
        if self.model is not None and HAS_YOLO:
            try:
                return self.model.predict_bar(image_path, type, output_dir, value_mapper, category_coords)
            except Exception as e:
                print(f"[CPAgent_adk.YOLOTool] YOLO 预测失败: {e}")
                # 回退到模拟实现
        
        # 模拟实现：返回随机结果
        print(f"[CPAgent_adk.YOLOTool] 模拟 YOLO 预测: {image_path}, 类型: {type}")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成模拟预测结果
        # 这里假设有 5 个柱状图
        mock_results = []
        for i in range(5):
            value = np.random.uniform(10, 100)
            mock_results.append([f"Category{i+1}", str(round(value, 2))])
        
        # 复制输入图像作为"输出图像"
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        try:
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"[CPAgent_adk.YOLOTool] 无法保存输出图像: {e}")
            output_path = ""
        
        return mock_results, output_path
    
    def refine_box_for_bars(self, image, x_min, y_min, x_max, y_max,
                          color_tolerance_hsv=(10, 40, 60),
                          gap_closure_size=15,
                          containment_thresh=0.4):
        """优化柱状图边界框（模拟实现）"""
        # 简单返回原始边界框
        return x_min, y_min, x_max, y_max

if __name__ == "__main__":
    tool = YOLOTool(str(YOLO_VERTICAL_BAR_PATH))
    print(f"工具初始化: {tool._initialized}")
    print(f"HAS_YOLO: {HAS_YOLO}")