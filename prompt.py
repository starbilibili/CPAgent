import os
from typing import Dict, List, Optional

# 提示词文件根目录
PROMPT_ROOT = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str, subdir: str = "ch") -> str:
    """加载提示词文件"""
    filepath = os.path.join(PROMPT_ROOT, subdir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"提示词文件不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# 预定义的提示词常量
PROMPT_CHART_ANALYSIS = load_prompt("chart_analysis")
PROMPT_CHART2TABLE = load_prompt("chart2table")
PROMPT_AXIS_DETECT = load_prompt("axis_detect")
PROMPT_CHART_CLS = load_prompt("chart_cls")
PROMPT_TEXT_CLS = load_prompt("text_cls")
PROMPT_CHART2TABLE_HAS_TEXT = load_prompt("chart2table_has_text")

# 全局基础提示词 - 定义 Agent 的角色和能力
GLOBAL_PROMPT = """
你是一个专业的图表理解与分析助手，专门处理各种类型的图表（柱状图、折线图、饼图等）。
你具备以下能力：
1. 视觉理解：能够分析图表图像，识别图表元素、坐标轴、数据点等
2. 文本识别：能够处理OCR检测结果，补全遗漏文本，进行语义分类
3. 数据分析：能够从图表中提取数值信息，重建数据表格
4. 工具使用：可以根据需要使用各种专业工具进行图表分析

你的任务是帮助用户完成图表相关的分析任务，包括但不限于：
- 图表类型识别与坐标轴分析
- 文本补全与语义分类
- 数据表格重建
- 图表内容解读

请根据用户的具体需求，选择合适的工具和方法进行分析。
输出结果时，请严格遵守指定的格式要求（如JSON、Markdown表格等）。
"""

# 整合指令版本1 - 结合现有提示词，形成完整的图表分析流程
INSTRUCTION_V1 = f"""
{GLOBAL_PROMPT}

## 核心分析流程与工具调用

### 第一阶段：图表类型识别与初步分析
**目标：** 识别图表类型和基本结构信息

**工具调用：**
- 必须首先调用 `ocr_detect` 工具获取图表中的文本信息
- 根据OCR结果和图像内容，分析并返回图表类型（chart_type）和是否包含文本标注（has_text）
- 输出包含：chart_type、has_text、title、legend、axis_labels 等基本信息

{PROMPT_CHART_ANALYSIS}


### 第二阶段：根据图表类型调用相应工具

#### 2.1 如果是饼图（pie chart）
**工具调用顺序：**
1. 调用 `pie_detect` 工具检测饼图扇区，获取各扇区的百分比占比
2. 结合OCR识别的标签信息，生成数据表格

**条件：** chart_type 为 "pie"


#### 2.2 如果是有文本标注的图表（has_text = true）
**工具调用顺序：**
1. 调用 `ocr_detect` 工具（第一阶段已调用，直接使用结果）
2. 无需调用其他检测工具，直接根据OCR文本分析数据
3. 使用文本分类工具 `text_cls` 对识别到的文本进行语义分类（类别名称、数值、单位等）
4. 根据分类结果整理数据，生成Markdown表格

**条件：** has_text 为 true，适用于文本信息完整的图表

{PROMPT_CHART2TABLE_HAS_TEXT}


#### 2.3 如果是柱状图或折线图（需要坐标轴回归）
**适用图表类型：**
- 垂直柱状图 (vertical_bar)
- 水平条形图 (horizontal_bar)
- 折线图 (line)

**工具调用顺序（必须按以下顺序执行）：**

**步骤1：坐标轴检测**
- 调用 `axis_detect` 工具，获取X轴和Y轴的关键点（刻度位置）信息

**步骤2：建立坐标轴回归模型**
- 调用 `get_axis_regression_model` 工具
- 输入：image_path 和 analysis_json（包含图表分析结果）
- 输出：像素坐标到数值坐标的映射函数（pixel_to_value）

**步骤3：根据图表类型调用数据提取工具**

**对于柱状图（vertical_bar 或 horizontal_bar）：**
- 调用 `bar_detect` 工具
- 输入参数：
  - image_path：图表图像路径
  - chart_type：柱状图类型（vertical_bar 或 horizontal_bar）
  - category_coords：类别轴的坐标位置列表（从 axis_detect 结果获取）
  - value_mapper：像素到数值的映射函数（从 get_axis_regression_model 获取）
  - output_dir：输出可视化结果的目录（可选）
- 输出：每个柱子的类别和对应的数值

**对于折线图（line）：**
- 调用 `auxiline_detect` 工具
- 输入参数：
  - image_path：图表图像路径
  - x_ticks：X轴刻度位置列表（从 axis_detect 结果获取）
  - text_bboxes：文本边界框列表（从 ocr_detect 结果获取）
  - pixel_to_value：像素到数值的映射函数（从 get_axis_regression_model 获取）
  - output_path：输出可视化结果的路径（可选）
- 输出：每个折线数据点的X值和Y值

**步骤4：整合结果生成表格**
- 将检测工具输出的数据点整理为Markdown表格格式

**重要提示：**
- 这三类图表必须先完成步骤1和步骤2，才能执行步骤3
- 步骤2的输出（value_mapper）是步骤3的必需输入
- 不要跳过任何步骤，否则无法正确提取数据


### 工具调用决策流程图

```
开始
 ↓
调用 ocr_detect 进行初步分析
 ↓
判断图表类型
 ↓
├─ 饼图 (pie)
│   └─ 调用 pie_detect → 生成表格
│
├─ has_text = true
│   └─ 调用 text_cls → 直接生成表格
│
└─ 柱状图/折线图
    ├─ 调用 axis_detect（必须）
    ├─ 调用 get_axis_regression_model（必须）
    ├─ 垂直/水平柱状图 → 调用 bar_detect
    ├─ 折线图 → 调用 auxiline_detect
    └─ 整合结果 → 生成表格
```


## 工具调用规则总结

| 工具名称 | 调用条件 | 依赖工具 | 输出说明 |
|---------|---------|---------|---------|
| ocr_detect | 所有图表分析的第一步 | 无 | 文本列表和边界框 |
| axis_detect | 柱状图、折线图 | ocr_detect | X/Y轴关键点位置 |
| get_axis_regression_model | 柱状图、折线图 | axis_detect, ocr_detect | 坐标映射函数 |
| bar_detect | 垂直/水平柱状图 | axis_detect, get_axis_regression_model | 柱子类别和数值 |
| auxiline_detect | 折线图 | axis_detect, get_axis_regression_model, ocr_detect | 数据点坐标 |
| pie_detect | 饼图 | ocr_detect | 扇区占比 |
| text_cls | has_text=true | ocr_detect | 文本语义分类 |


## 输出要求
- 所有输出必须放在 `<answer></answer>` 标签中
- JSON格式必须严格符合指定的结构
- Markdown表格应简洁清晰，包含表头和数据行
- 避免输出无关的解释或说明
- 工具调用结果应包含调用参数和返回数据，便于调试
"""

# 工具描述字典 - 用于 ADK Agent 的工具注册
TOOL_DESCRIPTIONS = {
    "ocr_detect": {
        "name": "ocr_detect",
        "description": "对输入的图像进行OCR文本检测，返回检测到的文本列表，每个文本包含内容和边界框信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测的图像文件路径"
                }
            },
            "required": ["image_path"]
        }
    },
    "axis_detect": {
        "name": "axis_detect",
        "description": "检测图表图像中的坐标轴关键点（刻度位置），返回X轴和Y轴的关键点坐标列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测的图像文件路径"
                },
                "output_dir": {
                    "type": "string",
                    "description": "输出可视化结果的目录"
                }
            },
            "required": ["image_path"]
        }
    },
    "pie_detect": {
        "name": "pie_detect",
        "description": "检测饼图图像中的扇区关键点，计算各扇区的百分比占比。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测的图像文件路径"
                },
                "output_dir": {
                    "type": "string",
                    "description": "输出可视化结果的目录"
                }
            },
            "required": ["image_path"]
        }
    },
    "bar_detect": {
        "name": "bar_detect",
        "description": "检测柱状图图像中的柱子位置，根据坐标轴回归模型计算柱子的数值。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测的图像文件路径"
                },
                "chart_type": {
                    "type": "string",
                    "enum": ["vertical_bar", "horizontal_bar"],
                    "description": "柱状图类型：垂直柱状图或水平条形图"
                },
                "category_coords": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "类别轴的坐标位置列表"
                },
                "output_dir": {
                    "type": "string",
                    "description": "输出可视化结果的目录"
                },
                "value_mapper": {
                    "type": "object",
                    "description": "像素到数值的映射函数"
                }
            },
            "required": ["image_path", "chart_type"]
        }
    },
    "auxiline_detect": {
        "name": "auxiline_detect",
        "description": "检测折线图图像中辅助线与折线的交点，根据坐标轴回归模型计算交点的数值。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测的图像文件路径"
                },
                "x_ticks": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "X轴刻度位置列表"
                },
                "text_bboxes": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "integer"}},
                    "description": "文本边界框列表"
                },
                "output_path": {
                    "type": "string",
                    "description": "输出可视化结果的路径"
                },
                "pixel_to_value": {
                    "type": "object",
                    "description": "像素到数值的映射函数"
                }
            },
            "required": ["image_path", "x_ticks"]
        }
    },
    "get_axis_regression_model": {
        "name": "get_axis_regression_model",
        "description": "基于图表分析结果和坐标轴关键点检测，建立像素坐标到数值坐标的线性回归模型。",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "图表图像文件路径"
                },
                "analysis_json": {
                    "type": "object",
                    "description": "图表分析结果的JSON对象"
                }
            },
            "required": ["image_path", "analysis_json"]
        }
    }
}

# 工具提示词模板
TOOL_PROMPTS = {
    "chart_analysis": PROMPT_CHART_ANALYSIS,
    "chart2table": PROMPT_CHART2TABLE,
    "axis_detect": PROMPT_AXIS_DETECT,
    "chart_cls": PROMPT_CHART_CLS,
    "text_cls": PROMPT_TEXT_CLS,
    "chart2table_has_text": PROMPT_CHART2TABLE_HAS_TEXT
}

def get_prompt(key: str, replacements: Optional[Dict[str, str]] = None) -> str:
    """获取提示词，支持变量替换"""
    if key not in TOOL_PROMPTS:
        raise KeyError(f"未知的提示词键: {key}")
    
    prompt = TOOL_PROMPTS[key]
    if replacements:
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
    
    return prompt

# 导出常用提示词
__all__ = [
    "GLOBAL_PROMPT",
    "INSTRUCTION_V1",
    "TOOL_DESCRIPTIONS",
    "TOOL_PROMPTS",
    "get_prompt",
    "load_prompt"
]