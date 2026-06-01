import os
import sys
import json
import re
import cv2
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# 导入本地模块
sys.path.append(os.path.dirname(__file__))
from llm import create_llm, BaseLLM
from tools import tool_manager, ToolManager
from prompt import INSTRUCTION_V1, GLOBAL_PROMPT, get_prompt, TOOL_DESCRIPTIONS
from config import ensure_temp_dir

TEMP_PATH = str(ensure_temp_dir())

# 尝试导入谷歌 ADK
try:
    from google.adk.agents.llm_agent import Agent
    from google.adk.tools import tool
    ADK_AVAILABLE = True
except ImportError:
    print("警告：google.adk 不可用，将使用自定义 Agent 实现")
    ADK_AVAILABLE = False
    # 定义占位符
    class Agent:
        def __init__(self, **kwargs):
            pass
    def tool(func):
        return func

class ChartParseAgentADK:
    """
    基于谷歌 ADK 框架的图表解析 Agent
    整合了 CPAgent 的核心功能，适配 ADK 接口
    """
    def __init__(self,
                 model_type: str = "qwen-vl",
                 model_config: Optional[Dict] = None,
                 use_adk: bool = True):
        """
        初始化 Agent

        Args:
            model_type: LLM 类型，目前支持 "qwen-vl"
            model_config: 模型配置参数，如 model_path
            use_adk: 是否使用 ADK 框架（如果可用）
        """
        self.model_type = model_type
        self.model_config = model_config or {}
        self.use_adk = use_adk and ADK_AVAILABLE
        
        # 初始化 LLM
        print(f"[ChartParseAgentADK] 初始化 {model_type} LLM...")
        self.llm = create_llm(model_type, **self.model_config)
        
        # 初始化工具管理器
        self.tool_manager = tool_manager
        
        # 初始化计数器
        self.counter = {"bar": 0, "pie": 0, "line": 0, "has_text": 0, "total": 0}
        
        # 创建 ADK Agent（如果使用 ADK）
        if self.use_adk:
            self._create_adk_agent()
        else:
            self.adk_agent = None
        
        print("[ChartParseAgentADK] 初始化完成")
    
    def _create_adk_agent(self):
        """创建 ADK Agent 实例"""
        try:
            # 注册工具
            tools = self._register_adk_tools()
            
            # 创建 Agent
            self.adk_agent = Agent(
                model=self.llm,  # ADK 可能需要特定的模型格式
                name="chart_parse_agent",
                description="专业的图表解析与分析助手",
                instruction=INSTRUCTION_V1,
                tools=tools
            )
            print("[ChartParseAgentADK] ADK Agent 创建成功")
        except Exception as e:
            print(f"[ChartParseAgentADK] 创建 ADK Agent 失败: {e}")
            print("将回退到自定义实现")
            self.use_adk = False
            self.adk_agent = None
    
    def _register_adk_tools(self):
        """注册工具到 ADK（如果可用）"""
        if not ADK_AVAILABLE:
            return []
        
        # 使用 @tool 装饰器定义工具函数
        tools = []
        
        @tool
        def ocr_detect(image_path: str) -> List[Dict[str, Any]]:
            """对输入的图像进行OCR文本检测"""
            return self.tool_manager.ocr_detect(image_path)
        
        @tool
        def axis_detect(image_path: str, output_dir: str = TEMP_PATH) -> Tuple[List, List, str]:
            """检测图表图像中的坐标轴关键点"""
            return self.tool_manager.axis_detect(image_path, output_dir)
        
        @tool
        def pie_detect(image_path: str, output_dir: str = TEMP_PATH) -> Tuple[List[str], str]:
            """检测饼图图像中的扇区关键点"""
            return self.tool_manager.pie_detect(image_path, output_dir)
        
        @tool
        def bar_detect(image_path: str, chart_type: str, category_coords: List[int],
                      output_dir: str = TEMP_PATH, value_mapper: Optional[callable] = None):
            """检测柱状图图像中的柱子位置"""
            return self.tool_manager.bar_detect(image_path, chart_type, category_coords,
                                               output_dir, value_mapper)
        
        @tool
        def auxiline_detect(image_path: str, x_ticks: List[int], text_bboxes: List[List[int]],
                           output_path: str = TEMP_PATH, pixel_to_value: Optional[callable] = None):
            """检测折线图图像中辅助线与折线的交点"""
            return self.tool_manager.auxiline_detect(image_path, x_ticks, text_bboxes,
                                                    pixel_to_value, output_path)
        
        # 添加工具到列表
        tools.extend([ocr_detect, axis_detect, pie_detect, bar_detect, auxiline_detect])
        return tools
    
    # ========== 核心功能方法（兼容原有 CPAgent 接口） ==========
    
    def ocr(self, image_path: str) -> List[Dict[str, Any]]:
        """OCR 检测"""
        return self.tool_manager.ocr_detect(image_path)
    
    def chart_analysis(self, ocr_result_str: str, img_path: str) -> Tuple[Dict[str, Any], str]:
        """
        图表分析（类型分类 + 文本分类 + 坐标轴分析）
        返回分析结果 JSON 和原始输出字符串
        """
        prompt = get_prompt("chart_analysis", {"【OCR_RESULT】": ocr_result_str})
        
        # 使用 LLM 调用
        response = self.llm(prompt, img_path)
        
        # 解析 JSON
        try:
            cleaned_res = response.strip()
            if "```json" in cleaned_res:
                cleaned_res = re.search(r"```json\s*([\s\S]*)\s*```", cleaned_res).group(1)
            elif "```" in cleaned_res:
                cleaned_res = cleaned_res.replace("```", "")
            
            analysis_json = json.loads(cleaned_res)
            return analysis_json, cleaned_res
        except Exception as e:
            print(f"[Chart Analysis] JSON 解析失败: {e}")
            print(f"[Chart Analysis] 原始输出: {response}")
            # 返回空结构防止崩溃
            return {
                "chart_type": "unknown",
                "x_axis": {"type": "unknown", "ticks": []},
                "y_axis": {"type": "unknown", "ticks": []}
            }, response
    
    def axis_detect_with_prompt(self, image_path: str, x_candidates: List, y_candidates: List) -> Dict[str, Any]:
        """
        坐标轴关键点检测（带 prompt 优化）
        基于现有 CPAgent 的 axis_detect 逻辑
        """
        try:
            x_candidates, y_candidates, temp_path = self.tool_manager.axis_detect(image_path, TEMP_PATH)
            if x_candidates is None and y_candidates is None:
                raise Exception("未检测到关键点")
            
            prompt = get_prompt("axis_detect", {
                "【X_KEYPOINTS】": json.dumps(x_candidates),
                "【Y_KEYPOINTS】": json.dumps(y_candidates)
            })
            
            response = self.llm(prompt, temp_path)
            
            # 解析结果
            match = re.search(r"```(?:json)?\s*([\s\S]*)\s*```", response, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
            else:
                result = {}
            
            # 可视化结果
            img = cv2.imread(temp_path)
            for item in result.get("x_axis", {}).get("ticks", []):
                x, y = item[1]
                cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            for item in result.get("y_axis", {}).get("ticks", []):
                x, y = item[1]
                cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.imwrite(temp_path, img)
            
            return result
        except Exception as e:
            print(f"[axis_detect_with_prompt] 错误: {e}")
            return {}
    
    def get_axis_regression_model(self, image_path: str, analysis_json: Dict[str, Any]) -> Tuple[float, float, int, List[int]]:
        """
        获取坐标轴像素坐标-数值坐标回归模型
        复用 tools 模块中的实现
        """
        return self.tool_manager.get_axis_regression_model(image_path, analysis_json)
    
    def generate_table(self, text: str, image_path: str) -> str:
        """生成数据表格"""
        prompt = get_prompt("chart2table", {"【TEXT】": text})
        response = self.llm(prompt, image_path)
        return response
    
    def conver_md2csv(self, md_string: str, output_path: str):
        """将 Markdown 表格转换为 CSV"""
        import csv
        
        if not md_string:
            print("[conver_md2csv] 输入字符串为空")
            return

        # 清洗字符串：去除 ```markdown 或 ``` 代码块包裹
        pattern = r"```(?:markdown)?\s*(.*?)\s*```"
        match = re.search(pattern, md_string, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
        else:
            content = md_string.strip()

        lines = content.split('\n')
        table_data = []

        for line in lines:
            line = line.strip()
            if not line or "|" not in line:
                continue

            cells = line.split('|')

            if line.startswith("|"):
                cells = cells[1:]
            if line.endswith("|"):
                cells = cells[:-1]

            clean_cells = [c.strip() for c in cells]

            # 跳过分隔线
            is_separator = all(all(char in '-: ' for char in cell) if cell else True for cell in clean_cells)
            if is_separator:
                continue

            table_data.append(clean_cells)

        if not table_data:
            print("[conver_md2csv] 未提取到有效的表格数据")
            return

        try:
            with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(table_data)
            print(f"[conver_md2csv] 成功转换并保存: {output_path}")
        except Exception as e:
            print(f"[conver_md2csv] 保存CSV失败: {e}")
    
    def chart2table(self, image_path: str, meta_path: Optional[str] = None) -> Tuple[str, str]:
        """
        主流程：图表转表格
        返回表格字符串和原始分析字符串
        """
        self.counter["total"] += 1
        
        # 1. OCR 检测
        ocr_result_list = self.ocr(image_path)
        ocr_result_str = json.dumps(ocr_result_list)
        
        # 2. 图表分析
        if meta_path and os.path.exists(meta_path):
            try:
                analysis_json = json.load(open(meta_path, "r", encoding="utf-8"))
                raw_analysis_str = json.dumps(analysis_json)
            except Exception:
                analysis_json, raw_analysis_str = self.chart_analysis(ocr_result_str, image_path)
        else:
            analysis_json, raw_analysis_str = self.chart_analysis(ocr_result_str, image_path)
        
        chart_type = analysis_json.get("chart_type", "unknown")
        print(f"[Chart2Table] 图表分析完成。类型: {chart_type}")
        
        # 3. 分支处理
        if chart_type == "has_text":
            print("[Chart2Table] 检测到有文本图表，直接生成表格")
            self.counter["has_text"] += 1
            table = self.generate_table(raw_analysis_str, image_path)
        
        elif chart_type == "pie":
            try:
                self.counter["pie"] += 1
                datapoint_texts, temp_path = self.tool_manager.pie_detect(image_path, TEMP_PATH)
                print(f"[Chart2Table] 检测到饼图，调用KPDTool，中间结果保存至:{temp_path}")
                
                # 将检测到的数据点注入 JSON
                if "other_text" not in analysis_json:
                    analysis_json["other_text"] = {}
                analysis_json["other_text"]["Datapoint"] = datapoint_texts
                
                final_context = json.dumps(analysis_json, ensure_ascii=False)
                table = self.generate_table(final_context, temp_path)
            except Exception as e:
                print(f"[Chart2Table Logic Error] {e} - 降级处理")
                table = self.generate_table(raw_analysis_str, image_path)
            
        else:
            # 垂直柱状图、水平条形图、折线图 -> 需要线性回归
            try:
                if chart_type == "line":
                    self.counter["line"] += 1
                elif "bar" in chart_type:
                    self.counter["bar"] += 1
                
                # 坐标轴解析
                slope, intercept, flag_val, category_coords = self.get_axis_regression_model(image_path, analysis_json)
                def pixel_to_value(px):
                    return slope * px + intercept
                
                # 调用特定工具
                datapoint_texts = []
                temp_path = image_path
                
                if "bar" in chart_type:
                    detect_type = "horizontal_bar" if chart_type == "horizontal_bar" else "vertical_bar"
                    datapoint_texts, temp_path = self.tool_manager.bar_detect(
                        image_path,
                        chart_type=detect_type,
                        category_coords=category_coords,
                        output_dir=TEMP_PATH,
                        value_mapper=pixel_to_value
                    )
                    print(f"[Chart2Table] 检测到bar chart, 调用YOLOTool，中间结果保存至:{temp_path}")
                elif chart_type == "line":
                    text_bboxes = [item['box'] for item in ocr_result_list]
                    datapoint_texts, temp_path = self.tool_manager.auxiline_detect(
                        image_path=image_path,
                        x_ticks=category_coords,
                        text_bboxes=text_bboxes,
                        pixel_to_value=pixel_to_value,
                        output_path=TEMP_PATH
                    )
                    print(f"[Chart2Table] 检测到line chart, 调用AuxilineTool，中间结果保存至:{temp_path}")
                
                # 将计算出的数值注入 JSON
                if "other_text" not in analysis_json:
                    analysis_json["other_text"] = {}
                analysis_json["other_text"]["Datapoint"] = datapoint_texts
                
                final_context = json.dumps(analysis_json, ensure_ascii=False)
                table = self.generate_table(final_context, temp_path)

            except Exception as e:
                print(f"[Chart2Table Logic Error] {e} - 降级处理")
                table = self.generate_table(raw_analysis_str, image_path)
        
        return table, raw_analysis_str
    
    def run(self, query: str, image_path: Optional[str] = None) -> str:
        """
        运行 Agent（ADK 或自定义）
        
        Args:
            query: 用户查询
            image_path: 可选图像路径
            
        Returns:
            Agent 的回复
        """
        if self.use_adk and self.adk_agent:
            # 使用 ADK Agent
            try:
                # ADK Agent 可能支持多模态输入
                response = self.adk_agent.run(query, image=image_path)
                return response
            except Exception as e:
                print(f"[ADK Agent] 运行失败: {e}")
                print("回退到自定义实现")
                self.use_adk = False
        
        # 自定义实现
        if "图表转表格" in query or "chart2table" in query.lower():
            if not image_path:
                return "请提供图像路径以进行图表转表格分析"
            table, _ = self.chart2table(image_path)
            return table
        elif "图表分析" in query or "chart analysis" in query.lower():
            if not image_path:
                return "请提供图像路径以进行图表分析"
            ocr_result = self.ocr(image_path)
            analysis_json, _ = self.chart_analysis(json.dumps(ocr_result), image_path)
            return json.dumps(analysis_json, ensure_ascii=False, indent=2)
        else:
            # 通用问答
            if image_path:
                response = self.llm(query, image_path)
            else:
                response = self.llm(query)
            return response


# 全局 Agent 实例
def create_agent(model_type: str = "qwen-vl", model_config: Optional[Dict] = None,
                 use_adk: bool = True, **kwargs) -> ChartParseAgentADK:
    """创建 Agent 实例的工厂函数"""
    config = dict(model_config or {})
    config.update(kwargs)
    return ChartParseAgentADK(model_type=model_type, model_config=config, use_adk=use_adk)

if __name__ == "__main__":
    # 测试代码
    agent = create_agent(model_path=os.environ.get("QWEN_VL_MODEL_PATH"))
    print("ChartParseAgentADK 初始化成功")
    print(f"模型类型: {agent.model_type}")
    print(f"使用 ADK: {agent.use_adk}")