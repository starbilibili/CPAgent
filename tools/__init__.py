import sys
import os
import cv2
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# 导入 CPAgent_adk 自己的工具实现
try:
    from .tool_yolo import YOLOTool
except ImportError as e:
    print(f"[CPAgent_adk.tools] 无法导入 YOLOTool: {e}")
    # 定义空类作为备用
    class YOLOTool:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .tool_kpd import KPDTool
except ImportError as e:
    print(f"[CPAgent_adk.tools] 无法导入 KPDTool: {e}")
    # 定义空类作为备用
    class KPDTool:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .tool_ocr import OCRTool
except ImportError as e:
    print(f"[CPAgent_adk.tools] 无法导入 OCRTool: {e}")
    # 定义空类作为备用
    class OCRTool:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .tool_auxiline import AuxilineTool
except ImportError as e:
    print(f"[CPAgent_adk.tools] 无法导入 AuxilineTool: {e}")
    # 定义空类作为备用
    class AuxilineTool:
        def __init__(self, *args, **kwargs):
            pass

from config import (
    CPAGENT_TOOLS_DIR,
    KPD_AXIS_PATH,
    KPD_PIE_PATH,
    YOLO_VERTICAL_BAR_PATH,
    YOLO_HORIZONTAL_BAR_PATH,
)

class ToolManager:
    """工具管理器，统一初始化和管理所有工具"""
    def __init__(self):
        self.ocr_tool = None
        self.kpd_axis = None
        self.kpd_pie = None
        self.seg_vertical_bar = None
        self.seg_horizontal_bar = None
        self.auxiline_tool = None
        self._initialized = False
    
    def initialize(self):
        """延迟初始化工具，避免启动时加载所有模型"""
        if self._initialized:
            return
        
        print("[ToolManager] 初始化工具...")
        try:
            self.ocr_tool = OCRTool()
            print("[ToolManager] OCRTool 初始化完成")
        except Exception as e:
            print(f"[ToolManager] OCRTool 初始化失败: {e}")
        
        try:
            self.kpd_axis = KPDTool(str(KPD_AXIS_PATH))
            print("[ToolManager] KPDTool (轴) 初始化完成")
        except Exception as e:
            print(f"[ToolManager] KPDTool (轴) 初始化失败: {e}")
        
        try:
            self.kpd_pie = KPDTool(str(KPD_PIE_PATH))
            print("[ToolManager] KPDTool (饼图) 初始化完成")
        except Exception as e:
            print(f"[ToolManager] KPDTool (饼图) 初始化失败: {e}")
        
        try:
            self.seg_vertical_bar = YOLOTool(str(YOLO_VERTICAL_BAR_PATH))
            print("[ToolManager] YOLOTool (垂直柱) 初始化完成")
        except Exception as e:
            print(f"[ToolManager] YOLOTool (垂直柱) 初始化失败: {e}")
        
        try:
            self.seg_horizontal_bar = YOLOTool(str(YOLO_HORIZONTAL_BAR_PATH))
            print("[ToolManager] YOLOTool (水平条) 初始化完成")
        except Exception as e:
            print(f"[ToolManager] YOLOTool (水平条) 初始化失败: {e}")
        
        try:
            self.auxiline_tool = AuxilineTool()
            print("[ToolManager] AuxilineTool 初始化完成")
        except Exception as e:
            print(f"[ToolManager] AuxilineTool 初始化失败: {e}")
        
        self._initialized = True
    
    # OCR 相关函数
    def ocr_detect(self, image_path: str) -> List[Dict[str, Any]]:
        """OCR 检测，返回文本和边界框"""
        self.initialize()
        if self.ocr_tool is None:
            raise RuntimeError("OCRTool 未初始化")
        return self.ocr_tool.text_detect(image_path)
    
    # 关键点检测相关函数
    def axis_detect(self, image_path: str, output_dir: str) -> Tuple[List, List, str]:
        """坐标轴关键点检测"""
        self.initialize()
        if self.kpd_axis is None:
            raise RuntimeError("KPDTool (轴) 未初始化")
        return self.kpd_axis.axis_detect(image_path, output_dir)
    
    def pie_detect(self, image_path: str, output_dir: str) -> Tuple[List[str], str]:
        """饼图关键点检测"""
        self.initialize()
        if self.kpd_pie is None:
            raise RuntimeError("KPDTool (饼图) 未初始化")
        return self.kpd_pie.pie_detect(image_path, output_dir)
    
    # 柱状图检测相关函数
    def bar_detect(self, image_path: str, chart_type: str, category_coords: List[int], 
                   output_dir: str, value_mapper: callable) -> Tuple[List[List[str]], str]:
        """柱状图检测"""
        self.initialize()
        if chart_type == "vertical_bar":
            tool = self.seg_vertical_bar
        elif chart_type == "horizontal_bar":
            tool = self.seg_horizontal_bar
        else:
            raise ValueError(f"不支持的柱状图类型: {chart_type}")
        
        if tool is None:
            raise RuntimeError(f"YOLOTool ({chart_type}) 未初始化")
        
        return tool.predict_bar(image_path, type=chart_type, output_dir=output_dir, 
                                value_mapper=value_mapper, category_coords=category_coords)
    
    # 折线图辅助线检测
    def auxiline_detect(self, image_path: str, x_ticks: List[int], text_bboxes: List[List[int]], 
                        pixel_to_value: callable, output_path: str) -> Tuple[List[List[str]], str]:
        """折线图辅助线检测"""
        self.initialize()
        if self.auxiline_tool is None:
            raise RuntimeError("AuxilineTool 未初始化")
        
        return self.auxiline_tool.draw_auxilines(image_path, x_ticks, text_bboxes, 
                                                 pixel_to_value, output_path)
    
    # 坐标轴回归模型（复用 CPAgent 中的逻辑）
    def get_axis_regression_model(self, image_path: str, analysis_json: Dict[str, Any], 
                                  tool_manager=None) -> Tuple[float, float, int, List[int]]:
        """
        获取坐标轴像素坐标-数值坐标回归模型
        返回值: slope, intercept, flag_val, category_coords
        """
        from sklearn.linear_model import LinearRegression, RANSACRegressor
        import re
        
        x_info = analysis_json.get("x_axis", {})
        y_info = analysis_json.get("y_axis", {})
        x_ticks = x_info.get("ticks", [])
        y_ticks = y_info.get("ticks", [])
        chart_type = analysis_json.get("chart_type", "unknown")
        
        x_type = x_info.get("type", "unknown")
        y_type = y_info.get("type", "unknown")
        
        numerical_axis = {} # 存放数值轴的元数据
        category_axis = {}  # 存放类别轴的元数据
        flag_val = 1        # 默认 Y 轴为数值轴 (1)
        
        # 确定坐标轴角色与方向
        if x_type == "numerical" and y_type != "numerical":
            numerical_axis = x_info
            category_axis = y_info
            flag_val = 0 # X轴数值关注 cx
        elif y_type == "numerical" and x_type != "numerical":
            numerical_axis = y_info
            category_axis = x_info
            flag_val = 1 # Y轴数值关注 cy
        else:
            # 回退逻辑
            if chart_type == "horizontal_bar":
                numerical_axis = x_info
                category_axis = y_info
                flag_val = 0
            else:
                # vertical_bar, line, 默认情况
                numerical_axis = y_info
                category_axis = x_info
                flag_val = 1

        flag_cat = 1 - flag_val
        
        # 收集回归用的数据点 [(value, pixel), ...]
        points_for_regression = []
        print("[get_axis_regression_model] 尝试利用KPDTool识别坐标轴关键点")
        
        # 使用工具管理器进行轴检测
        if tool_manager is None:
            tool_manager = self
        
        try:
            x_candidates, y_candidates, temp_path = tool_manager.axis_detect(image_path, "/tmp")
            axis_res = {}
            if x_candidates and y_candidates:
                # 构建axis_res结构
                axis_res = {
                    "x_axis": {"ticks": [[str(i), pt] for i, pt in enumerate(x_candidates)]},
                    "y_axis": {"ticks": [[str(i), pt] for i, pt in enumerate(y_candidates)]}
                }
        except Exception as e:
            print(f"[get_axis_regression_model] KPDTool轴检测失败: {e}")
            axis_res = {}
        
        value_axis_key = "y_axis" if flag_val == 1 else "x_axis"
        if axis_res and value_axis_key in axis_res:
            value_ticks = axis_res[value_axis_key].get("ticks", [])
            if len(value_ticks) >= len(numerical_axis.get("ticks", [])):
                numerical_axis['ticks'] = value_ticks
            else:
                print("[get_axis_regression_model] KPDTool无法识别数值轴关键点，通过文本框识别")
            value_ticks = numerical_axis.get("ticks", [])
            for item in value_ticks:
                text = str(item[0])
                bbox = item[1]
                try:
                    # bbox 格式为 [cx, cy, w, h]
                    pixel_val = bbox[flag_val] 
                    clean_label = re.sub(r'[^\d\.\-]', '', text)
                    if not clean_label or clean_label in ['.', '-']: 
                        continue
                    val = float(clean_label)
                    points_for_regression.append((val, pixel_val))
                except (ValueError, IndexError):
                    continue
        
        # 提取类别轴坐标 (category_coords)
        category_coords = []
        cat_axis_key = "y_axis" if flag_val == 0 else "x_axis"
        if axis_res and cat_axis_key in axis_res:
            cat_ticks = axis_res[cat_axis_key].get("ticks", [])
            if len(cat_ticks) >= len(category_axis.get("ticks", [])):
                category_axis['ticks'] = cat_ticks
            else:
                print("[get_axis_regression_model] KPDTool无法识别类别轴关键点，通过文本框识别")
            cat_ticks = category_axis.get("ticks", [])
            for item in cat_ticks:
                bbox = item[1]
                category_coords.append(int(bbox[flag_cat]))
        
        # 执行回归拟合
        if len(points_for_regression) < 2:
            print(f"[get_axis_regression_model] 有效数值轴刻度点不足 ({len(points_for_regression)})")
            raise ValueError("Not enough points for regression")
        
        pixel_data = np.array([p[1] for p in points_for_regression]).reshape(-1, 1)
        value_data = np.array([p[0] for p in points_for_regression])
        
        slope = 0
        intercept = 0
        
        # 优先使用 RANSAC 处理离群点
        if len(points_for_regression) >= 3:
            try:
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    min_samples=2,
                    residual_threshold=10.0,
                    random_state=42
                )
                ransac.fit(pixel_data, value_data)
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
                inliers = np.sum(ransac.inlier_mask_)
                print(f"[get_axis_regression_model] RANSAC 拟合成功 (Inliers: {inliers}/{len(points_for_regression)})")
            except Exception as e:
                print(f"[get_axis_regression_model] RANSAC 失败: {e}, 回退到普通线性回归")
                lm = LinearRegression().fit(pixel_data, value_data)
                slope = lm.coef_[0]
                intercept = lm.intercept_
        else:
            lm = LinearRegression().fit(pixel_data, value_data)
            slope = lm.coef_[0]
            intercept = lm.intercept_
            
        return slope, intercept, flag_val, category_coords


# 全局工具管理器实例
tool_manager = ToolManager()