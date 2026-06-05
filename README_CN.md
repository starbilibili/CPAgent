# CPAgent ADK

A chart parsing Agent based on the Google ADK framework, integrating the core capabilities of CPAgent to support automatic parsing of chart images into structured data tables.

## Feature Overview

- **Chart Type Recognition**: Bar charts, line charts, pie charts, charts with text annotations, etc.
- **OCR Text Detection**: Extracts text from charts such as titles, legends, and axis ticks.
- **Keypoint Detection**: Axis ticks, pie chart sectors, bar chart elements, and line data points.
- **Axis Regression**: Establishes a linear mapping from pixel coordinates to numerical values.
- **Table Generation**: Outputs Markdown tables, with support for CSV conversion.
- **Multimodal LLM**: Local inference using Qwen-VL.
- **ADK Compatibility**: Optional integration with the Google ADK Agent framework; automatically falls back to a custom implementation if unavailable.

## Project Structure

```text
CPAgent_adk/
├── config.py             # Model paths and directory configuration (centralized management)
├── agent.py              # Main Agent entry point (ChartParseAgentADK)
├── llm.py                # LLM wrapper (Qwen-VL)
├── prompt.py             # Prompt loading and ADK instructions
├── tools/                # Tool manager and individual detection tools
│   ├── __init__.py       # Unified dispatch via ToolManager
│   ├── tool_ocr.py       # OCR wrapper
│   ├── tool_kpd.py       # Keypoint detection (axis / pie chart)
│   ├── tool_yolo.py      # Bar chart segmentation detection
│   └── tool_auxiline.py  # Line chart auxiliary line detection
├── prompts/              # Chinese and English prompt templates
│   ├── ch/               # Chinese prompts
│   └── en/               # English prompts
├── requirements.txt
└── test_*.py             # Unit test scripts
```

**Note**: The `tools.py` file in the root directory duplicates the contents of the `tools/` package. During runtime, Python will prioritize loading the `tools/` package. The `tools.py` file can be considered a legacy artifact.

## Environment Requirements

- Python 3.10+
- CUDA (Recommended, for KPD / YOLO model inference)
- Requires CPAgent pre-trained weights. Download link: https://huggingface.co/seven-night/CPAgent

## Installation

```bash
cd /data5/home/xiechenyu2023/project/ChartQA/CPAgent_adk
pip install -r requirements.txt

# Additional dependencies for CPAgent tools (if not already installed)
pip install paddleocr paddlepaddle ultralytics pillow
```

## Configuration

### Model Paths Overview

All model loading paths are centrally managed in `config.py` and can be overridden via environment variables.

| Model / Tool | Default Path | Environment Variable | Description |
| ------ |------ |------ |------ |
| **Qwen-VL** | `/lustre/home/xiechenyu2023/saved_model/qwen3/Qwen3-VL-8B-Instruct` | `QWEN_VL_MODEL_PATH` | Multimodal LLM for chart analysis and table generation |
| **KPD Axis** | `../CPAgent/tools/checkpoint/kpd_axis.pt` | `KPD_AXIS_PATH` | Axis tick keypoint detection |
| **KPD Pie** | `../CPAgent/tools/checkpoint/kpd_pie.pt` | `KPD_PIE_PATH` | Pie chart sector keypoint detection |
| **YOLO Vertical Bar** | `../CPAgent/tools/checkpoint/yolo_vertical_bar.pt` | `YOLO_VERTICAL_BAR_PATH` | Vertical bar chart segmentation |
| **YOLO Horizontal Bar** | `../CPAgent/tools/checkpoint/yolo_horizontal_bar.pt` | `YOLO_HORIZONTAL_BAR_PATH` | Horizontal bar chart segmentation |
| **PaddleOCR** | Auto-download | — | OCR text detection; automatically fetched on first run |
| **Auxiline** | No weights required | — | Line chart auxiliary line detection (pure algorithm) |

Directory-level Environment Variables:

| Environment Variable | Default Value | Description |
| ------ |------ |------ |
| `CPAGENT_ROOT` | `../CPAgent` | Root directory of the CPAgent project |
| `CPAGENT_TOOLS_DIR` | `{CPAGENT_ROOT}/tools` | CPAgent tool source code directory |
| `CPAGENT_CHECKPOINT_DIR` | `{CPAGENT_TOOLS_DIR}/checkpoint` | Directory containing all `.pt` weight files |
| `CPAGENT_ADK_TEMP_DIR` | `./temp_data` | Output directory for intermediate results |
| `CPAGENT_TEST_DATA_DIR` | `{CPAGENT_ROOT}/data` | Test image directory |

Check if model paths are ready:

```bash
python config.py
```

### Initialize Agent

```bash
# .env or shell environment variables
QWEN_VL_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
# Optional: uniformly modify the weights directory
CPAGENT_CHECKPOINT_DIR=/path/to/checkpoint
```

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")
```

## Quick Start

### Chart-to-Table (Main Pipeline)

```python
from agent import create_agent

agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

image_path = "/path/to/chart.png"
table_md, analysis_json = agent.chart2table(image_path)

print(table_md)

# Optional: Save as CSV
agent.conver_md2csv(table_md, "output.csv")
```

### Interactive via `run` Interface

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

# Chart-to-Table
result = agent.run("Convert chart to table", image_path="/path/to/chart.png")

# Chart Analysis
result = agent.run("Analyze chart", image_path="/path/to/chart.png")

# General Question Answering
result = agent.run("What is the main trend in this chart?", image_path="/path/to/chart.png")
```

### Step-by-step Invocation

```python
agent = create_agent(model_path="/path/to/Qwen3-VL-8B-Instruct")

# 1. OCR
ocr_results = agent.ocr(image_path)

# 2. Chart Analysis
analysis, raw = agent.chart_analysis(json.dumps(ocr_results), image_path)

# 3. Axis Regression
slope, intercept, flag, cat_coords = agent.get_axis_regression_model(image_path, analysis)
```

## Processing Pipeline

```text
Input Chart Image
    │
    ▼
OCR Text Detection
    │
    ▼
LLM Chart Analysis (Type Classification + Axis Properties + Text Semantic Classification)
    │
    ├── has_text ──────────► Directly Generate Markdown Table
    │
    ├── pie ───────────────► KPD Pie Detection ──► Generate Table
    │
    └── bar / line ────────► Axis Keypoint Detection
                              │
                              ▼
                         Linear Regression (Pixel → Value)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              YOLO Bar Detection   Auxiline Detection
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         Generate Markdown Table
```

## Supported Chart Types

| chart_type | Description | Tools Used |
| ------ |------ |------ |
| `has_text` | Values directly annotated on elements | OCR + LLM |
| `pie` | Pie / Donut Chart | KPDTool (pie) |
| `vertical_bar` | Vertical Bar Chart | KPDTool + YOLOTool |
| `horizontal_bar` | Horizontal Bar Chart | KPDTool + YOLOTool |
| `line` | Line Chart | KPDTool + AuxilineTool |

## Testing

```bash
# Module import test
python test_import.py

# Tool import test
python test_tools_import.py

# Tool functionality test (requires test images and model weights)
python test_functionality.py
```

## ADK Mode

When `google-adk` is installed and available, the Agent will automatically register the following tools with ADK:

- `ocr_detect` — OCR text detection
- `axis_detect` — Axis keypoint detection
- `pie_detect` — Pie chart sector detection
- `bar_detect` — Bar chart detection
- `auxiline_detect` — Line chart auxiliary line detection

If ADK is unavailable, the system automatically falls back to the built-in pipeline of `ChartParseAgentADK`, ensuring core functionalities remain unaffected.

## Relationship with CPAgent

This project is an ADK-adapted version of CPAgent:

- Reuses CPAgent's pre-trained model weights and original tool implementations.
- The tools under `tools/` act as wrappers for CPAgent tools, providing simulated fallbacks when original tools are unavailable.
- The core `chart2table` pipeline remains consistent with CPAgent's `ChartParseAgent`.

## Known Limitations

- Compound chart types such as stacked bar charts (`stacked bar`) are currently not supported.
- Model weight paths can be configured via `config.py` or environment variables; see "Configuration → Model Paths Overview".
- When original CPAgent tools are unavailable, tool wrappers fall back to simulated implementations that return random/placeholder data. This is strictly for development and debugging purposes.
