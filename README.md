# CPAgent ADK

基于 Google ADK 框架的图表解析 Agent，整合 [CPAgent](../CPAgent) 的核心能力，支持将图表图像自动解析为结构化数据表格。

## 功能概述

- **图表类型识别**：柱状图、折线图、饼图、带文本标注图表等
- **OCR 文本检测**：提取图表中的标题、图例、坐标轴刻度等文本
- **关键点检测**：坐标轴刻度、饼图扇区、柱状图元素、折线数据点
- **坐标轴回归**：建立像素坐标到数值的线性映射
- **表格生成**：输出 Markdown 表格，支持转换为 CSV
- **多模态 LLM**：Qwen-VL 本地推理
- **ADK 兼容**：可选接入 Google ADK Agent 框架，不可用时自动回退到自定义实现

## 项目结构

```
CPAgent_adk/
├── config.py             # 模型路径与目录配置（统一管理）
├── agent.py              # 主 Agent 入口（ChartParseAgentADK）
├── llm.py                # LLM 封装（Qwen-VL）
├── prompt.py             # 提示词加载与 ADK 指令
├── tools/                # 工具管理器与各检测工具
│   ├── __init__.py       # ToolManager 统一调度
│   ├── tool_ocr.py       # OCR 包装器
│   ├── tool_kpd.py       # 关键点检测（坐标轴 / 饼图）
│   ├── tool_yolo.py      # 柱状图分割检测
│   └── tool_auxiline.py  # 折线图辅助线检测
├── prompts/              # 中英文提示词模板
│   ├── ch/               # 中文提示词
│   └── en/               # 英文提示词
├── requirements.txt
└── test_*.py             # 单元测试脚本
```

> **注意**：根目录下的 `tools.py` 与 `tools/` 包内容重复，实际运行时 Python 优先加载 `tools/` 包，`tools.py` 可视为遗留文件。

## 环境要求

- Python 3.10+
- CUDA（推荐，用于 KPD / YOLO 模型推理）
- 依赖 CPAgent 预训练权重，位于：
  ```
  ../CPAgent/tools/checkpoint/
  ├── kpd_axis.pt
  ├── kpd_pie.pt
  ├── yolo_vertical_bar.pt
  └── yolo_horizontal_bar.pt
  ```

## 安装

```bash
cd /data5/home/xiechenyu2023/project/ChartQA/CPAgent_adk
pip install -r requirements.txt

# CPAgent 工具额外依赖（若尚未安装）
pip install paddleocr paddlepaddle ultralytics pillow
```

## 配置

### 模型路径一览

所有模型加载路径集中在 `config.py` 管理，可通过环境变量覆盖默认值。

| 模型 / 工具 | 默认路径 | 环境变量 | 说明 |
|-------------|----------|----------|------|
| **Qwen-VL** | `/lustre/home/xiechenyu2023/saved_model/qwen3/Qwen3-VL-8B-Instruct` | `QWEN_VL_MODEL_PATH` | 多模态 LLM，用于图表分析与表格生成 |
| **KPD 坐标轴** | `../CPAgent/tools/checkpoint/kpd_axis.pt` | `KPD_AXIS_PATH` | 坐标轴刻度关键点检测 |
| **KPD 饼图** | `../CPAgent/tools/checkpoint/kpd_pie.pt` | `KPD_PIE_PATH` | 饼图扇区关键点检测 |
| **YOLO 垂直柱** | `../CPAgent/tools/checkpoint/yolo_vertical_bar.pt` | `YOLO_VERTICAL_BAR_PATH` | 垂直柱状图分割 |
| **YOLO 水平条** | `../CPAgent/tools/checkpoint/yolo_horizontal_bar.pt` | `YOLO_HORIZONTAL_BAR_PATH` | 水平条形图分割 |
| **PaddleOCR** | 自动下载 | — | OCR 文本检测，首次运行自动拉取 |
| **Auxiline** | 无需权重 | — | 折线图辅助线检测（纯算法） |

目录级环境变量：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `CPAGENT_ROOT` | `../CPAgent` | CPAgent 项目根目录 |
| `CPAGENT_TOOLS_DIR` | `{CPAGENT_ROOT}/tools` | CPAgent 工具源码目录 |
| `CPAGENT_CHECKPOINT_DIR` | `{CPAGENT_TOOLS_DIR}/checkpoint` | 所有 `.pt` 权重所在目录 |
| `CPAGENT_ADK_TEMP_DIR` | `./temp_data` | 中间结果输出目录 |
| `CPAGENT_TEST_DATA_DIR` | `{CPAGENT_ROOT}/data` | 测试图像目录 |

检查模型路径是否就绪：

```bash
python config.py
```

### 初始化 Agent

```bash
# .env 或 shell 环境变量
QWEN_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
# 可选：统一修改权重目录
CPAGENT_CHECKPOINT_DIR=/path/to/checkpoint
```

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")
```

## 快速开始

### 图表转表格（主流程）

```python
from agent import create_agent

agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

image_path = "/path/to/chart.png"
table_md, analysis_json = agent.chart2table(image_path)

print(table_md)

# 可选：保存为 CSV
agent.conver_md2csv(table_md, "output.csv")
```

### 通过 run 接口交互

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

# 图表转表格
result = agent.run("图表转表格", image_path="/path/to/chart.png")

# 图表分析
result = agent.run("图表分析", image_path="/path/to/chart.png")

# 通用问答
result = agent.run("这张图表的主要趋势是什么？", image_path="/path/to/chart.png")
```

### 分步调用

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

# 1. OCR
ocr_results = agent.ocr(image_path)

# 2. 图表分析
analysis, raw = agent.chart_analysis(json.dumps(ocr_results), image_path)

# 3. 坐标轴回归
slope, intercept, flag, cat_coords = agent.get_axis_regression_model(image_path, analysis)
```

## 处理流程

```
输入图表图像
    │
    ▼
OCR 文本检测
    │
    ▼
LLM 图表分析（类型分类 + 坐标轴属性 + 文本语义分类）
    │
    ├── has_text ──────────► 直接生成 Markdown 表格
    │
    ├── pie ───────────────► KPD 饼图检测 ──► 生成表格
    │
    └── bar / line ────────► 坐标轴关键点检测
                              │
                              ▼
                         线性回归（像素→数值）
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              YOLO 柱检测          辅助线折线检测
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         生成 Markdown 表格
```

## 支持的图表类型

| chart_type       | 说明           | 使用工具                    |
|------------------|----------------|-----------------------------|
| `has_text`       | 元素上直接标注数值 | OCR + LLM                  |
| `pie`            | 饼图 / 环形图   | KPDTool (pie)              |
| `vertical_bar`   | 垂直柱状图      | KPDTool + YOLOTool         |
| `horizontal_bar` | 水平条形图      | KPDTool + YOLOTool         |
| `line`           | 折线图          | KPDTool + AuxilineTool     |

## 测试

```bash
# 模块导入测试
python test_import.py

# 工具导入测试
python test_tools_import.py

# 工具功能测试（需要测试图像和模型权重）
python test_functionality.py
```

## ADK 模式

当安装了 `google-adk` 且环境可用时，Agent 会自动注册以下工具到 ADK：

- `ocr_detect` — OCR 文本检测
- `axis_detect` — 坐标轴关键点检测
- `pie_detect` — 饼图扇区检测
- `bar_detect` — 柱状图检测
- `auxiline_detect` — 折线辅助线检测

ADK 不可用时，系统自动回退到 `ChartParseAgentADK` 的内置流程，不影响核心功能。

## 与 CPAgent 的关系

本项目是 [CPAgent](../CPAgent) 的 ADK 适配版本：

- 复用 CPAgent 的预训练模型权重和原始工具实现
- `tools/` 下的各工具为 CPAgent 工具的包装器，原始工具不可用时提供模拟回退
- 核心 `chart2table` 流程与 CPAgent 的 `ChartParseAgent` 保持一致

## 已知限制

- 暂不支持堆叠柱状图（`stacked bar`）等复合图表类型
- 模型权重路径可通过 `config.py` 或环境变量配置，详见「配置 → 模型路径一览」
- 工具包装器在原始 CPAgent 工具不可用时回退到模拟实现，返回随机/占位数据，仅用于开发调试
