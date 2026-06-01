import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from config import CPAGENT_TOOLS_DIR, KPD_AXIS_PATH

# 尝试从 CPAgent 导入原始 KPDTool
try:
    sys.path.append(str(CPAGENT_TOOLS_DIR))
    from tool_kpd import KPDTool as BaseKPDTool
    HAS_KPD = True
except ImportError as e:
    print(f"[CPAgent_adk] 无法导入原始 KPDTool: {e}")
    HAS_KPD = False
except Exception as e:
    print(f"[CPAgent_adk] KPDTool 导入过程中发生错误: {e}")
    HAS_KPD = False

class KPDTool:
    """CPAgent_adk 的关键点检测工具包装器"""
    
    def __init__(self, model_weight_path: str):
        """
        初始化 KPD 工具
        
        Args:
            model_weight_path: 模型权重文件路径
        """
        self.model_weight_path = model_weight_path
        self.model = None
        self.device = "cuda"  # 假设有 GPU
        self._initialized = False
        
    def _try_initialize(self):
        """尝试初始化 KPD 模型"""
        if self._initialized:
            return
            
        if HAS_KPD and os.path.exists(self.model_weight_path):
            try:
                self.model = BaseKPDTool(self.model_weight_path)
                self._initialized = True
                print(f"[CPAgent_adk.KPDTool] 使用原始 KPDTool: {self.model_weight_path}")
                return
            except Exception as e:
                print(f"[CPAgent_adk.KPDTool] 原始 KPDTool 初始化失败: {e}")
        
        # 备用方案：提供一个模拟实现
        print(f"[CPAgent_adk.KPDTool] 使用模拟 KPD 实现: {self.model_weight_path}")
        self.model = None
        self._initialized = True
    
    def axis_detect(self, image_path: str, output_dir: str) -> Tuple[List, List, str]:
        """坐标轴关键点检测
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            
        Returns:
            (x_candidates, y_candidates, temp_path)
        """
        self._try_initialize()
        
        # 如果原始 KPD 可用，使用它
        if self.model is not None and HAS_KPD:
            try:
                return self.model.axis_detect(image_path, output_dir)
            except Exception as e:
                print(f"[CPAgent_adk.KPDTool] KPD 轴检测失败: {e}")
                # 回退到模拟实现
        
        # 模拟实现：返回随机关键点
        print(f"[CPAgent_adk.KPDTool] 模拟轴检测: {image_path}")
        
        # 读取图像获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"[CPAgent_adk.KPDTool] 无法读取图像: {image_path}")
            return [], [], ""
        
        height, width = img.shape[:2]
        
        # 生成模拟的 x 轴关键点（沿底部）
        x_candidates = []
        for i in range(5):
            x = int(width * (i + 1) / 6)
            y = int(height * 0.9)
            x_candidates.append([x, y])
        
        # 生成模拟的 y 轴关键点（沿左侧）
        y_candidates = []
        for i in range(5):
            x = int(width * 0.1)
            y = int(height * (i + 1) / 6)
            y_candidates.append([x, y])
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模拟结果图像
        output_path = os.path.join(output_dir, f"kpd_{os.path.basename(image_path)}")
        try:
            # 在图像上绘制关键点
            for pt in x_candidates:
                cv2.circle(img, tuple(pt), 5, (0, 255, 0), -1)
            for pt in y_candidates:
                cv2.circle(img, tuple(pt), 5, (255, 0, 0), -1)
            cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"[CPAgent_adk.KPDTool] 无法保存输出图像: {e}")
            output_path = ""
        
        return x_candidates, y_candidates, output_path
    
    def pie_detect(self, image_path: str, output_dir: str) -> Tuple[List[str], str]:
        """饼图关键点检测
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            
        Returns:
            (labels, output_path)
        """
        self._try_initialize()
        
        # 如果原始 KPD 可用，使用它
        if self.model is not None and HAS_KPD:
            try:
                return self.model.pie_detect(image_path, output_dir)
            except Exception as e:
                print(f"[CPAgent_adk.KPDTool] KPD 饼图检测失败: {e}")
                # 回退到模拟实现
        
        # 模拟实现：返回随机标签
        print(f"[CPAgent_adk.KPDTool] 模拟饼图检测: {image_path}")
        
        mock_labels = ["A: 25%", "B: 30%", "C: 20%", "D: 15%", "E: 10%"]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模拟结果图像
        output_path = os.path.join(output_dir, f"pie_{os.path.basename(image_path)}")
        try:
            img = cv2.imread(image_path)
            if img is not None:
                cv2.imwrite(output_path, img)
        except Exception as e:
            print(f"[CPAgent_adk.KPDTool] 无法保存输出图像: {e}")
            output_path = ""
        
        return mock_labels, output_path

if __name__ == "__main__":
    tool = KPDTool(str(KPD_AXIS_PATH))
    print(f"工具初始化: {tool._initialized}")
    print(f"HAS_KPD: {HAS_KPD}")