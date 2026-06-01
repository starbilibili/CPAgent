"""
CPAgent ADK 路径与模型权重配置

所有模型加载路径集中在此文件管理，支持通过环境变量覆盖默认值。
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 项目目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CHARTQA_ROOT = PROJECT_ROOT.parent
CPAGENT_ROOT = Path(os.environ.get("CPAGENT_ROOT", str(CHARTQA_ROOT / "CPAgent")))
CPAGENT_TOOLS_DIR = Path(os.environ.get("CPAGENT_TOOLS_DIR", str(CPAGENT_ROOT / "tools")))
CPAGENT_CHECKPOINT_DIR = Path(
    os.environ.get("CPAGENT_CHECKPOINT_DIR", str(CPAGENT_TOOLS_DIR / "checkpoint"))
)
TEMP_DIR = Path(os.environ.get("CPAGENT_ADK_TEMP_DIR", str(PROJECT_ROOT / "temp_data")))
TEST_DATA_DIR = Path(os.environ.get("CPAGENT_TEST_DATA_DIR", str(CPAGENT_ROOT / "data")))

# ---------------------------------------------------------------------------
# LLM 模型（Qwen-VL）
# ---------------------------------------------------------------------------
QWEN_VL_MODEL_PATH = Path(
    os.environ.get(
        "QWEN_VL_MODEL_PATH",
        "/lustre/home/xiechenyu2023/saved_model/qwen3/Qwen3-VL-8B-Instruct",
    )
)

# ---------------------------------------------------------------------------
# 视觉工具模型权重（.pt）
# ---------------------------------------------------------------------------
KPD_AXIS_PATH = Path(os.environ.get("KPD_AXIS_PATH", str(CPAGENT_CHECKPOINT_DIR / "kpd_axis.pt")))
KPD_PIE_PATH = Path(os.environ.get("KPD_PIE_PATH", str(CPAGENT_CHECKPOINT_DIR / "kpd_pie.pt")))
YOLO_VERTICAL_BAR_PATH = Path(
    os.environ.get("YOLO_VERTICAL_BAR_PATH", str(CPAGENT_CHECKPOINT_DIR / "yolo_vertical_bar.pt"))
)
YOLO_HORIZONTAL_BAR_PATH = Path(
    os.environ.get("YOLO_HORIZONTAL_BAR_PATH", str(CPAGENT_CHECKPOINT_DIR / "yolo_horizontal_bar.pt"))
)

# ---------------------------------------------------------------------------
# 模型路径注册表（便于查阅与校验）
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict] = {
    "qwen_vl": {
        "description": "Qwen3-VL 多模态大模型，用于图表分析、表格生成等 LLM 任务",
        "path": QWEN_VL_MODEL_PATH,
        "type": "huggingface",
        "required": True,
        "env_var": "QWEN_VL_MODEL_PATH",
        "used_by": ["llm.QwenVLLL", "agent.ChartParseAgentADK"],
    },
    "kpd_axis": {
        "description": "KPD 坐标轴关键点检测模型",
        "path": KPD_AXIS_PATH,
        "type": "checkpoint",
        "required": True,
        "env_var": "KPD_AXIS_PATH",
        "used_by": ["tools.KPDTool (axis)", "ToolManager.axis_detect"],
    },
    "kpd_pie": {
        "description": "KPD 饼图扇区关键点检测模型",
        "path": KPD_PIE_PATH,
        "type": "checkpoint",
        "required": True,
        "env_var": "KPD_PIE_PATH",
        "used_by": ["tools.KPDTool (pie)", "ToolManager.pie_detect"],
    },
    "yolo_vertical_bar": {
        "description": "YOLO 垂直柱状图分割模型",
        "path": YOLO_VERTICAL_BAR_PATH,
        "type": "checkpoint",
        "required": True,
        "env_var": "YOLO_VERTICAL_BAR_PATH",
        "used_by": ["tools.YOLOTool", "ToolManager.bar_detect (vertical_bar)"],
    },
    "yolo_horizontal_bar": {
        "description": "YOLO 水平条形图分割模型",
        "path": YOLO_HORIZONTAL_BAR_PATH,
        "type": "checkpoint",
        "required": True,
        "env_var": "YOLO_HORIZONTAL_BAR_PATH",
        "used_by": ["tools.YOLOTool", "ToolManager.bar_detect (horizontal_bar)"],
    },
    "paddleocr": {
        "description": "PaddleOCR 文本检测模型（首次运行时自动下载，无需手动配置 .pt 路径）",
        "path": None,
        "type": "auto_download",
        "required": True,
        "env_var": None,
        "used_by": ["tools.OCRTool", "ToolManager.ocr_detect"],
    },
    "auxiline": {
        "description": "折线图辅助线检测（纯算法实现，无需加载模型权重）",
        "path": None,
        "type": "algorithm",
        "required": False,
        "env_var": None,
        "used_by": ["tools.AuxilineTool", "ToolManager.auxiline_detect"],
    },
}


def ensure_temp_dir() -> Path:
    """确保临时输出目录存在"""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_DIR


def check_model_paths(required_only: bool = True) -> Dict[str, bool]:
    """
    检查模型路径是否存在

    Returns:
        {model_name: exists} 字典；auto_download / algorithm 类型始终为 True
    """
    results = {}
    for name, info in MODEL_REGISTRY.items():
        if info["type"] in ("auto_download", "algorithm"):
            results[name] = True
            continue
        if required_only and not info["required"]:
            continue
        path = info["path"]
        results[name] = path is not None and Path(path).exists()
    return results


def print_model_paths(missing_only: bool = False) -> None:
    """打印模型路径清单"""
    print("=" * 60)
    print("CPAgent ADK 模型路径配置")
    print("=" * 60)
    for name, info in MODEL_REGISTRY.items():
        path = info["path"]
        exists = info["type"] in ("auto_download", "algorithm") or (path and Path(path).exists())
        if missing_only and exists:
            continue
        status = "✓" if exists else "✗"
        path_str = str(path) if path else f"({info['type']})"
        print(f"\n[{status}] {name}")
        print(f"    说明: {info['description']}")
        print(f"    路径: {path_str}")
        if info["env_var"]:
            print(f"    环境变量: {info['env_var']}")
        print(f"    使用模块: {', '.join(info['used_by'])}")
    print("\n" + "=" * 60)
    print(f"CPAgent 工具目录: {CPAGENT_TOOLS_DIR}")
    print(f"权重目录:         {CPAGENT_CHECKPOINT_DIR}")
    print(f"临时目录:         {TEMP_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_model_paths()
