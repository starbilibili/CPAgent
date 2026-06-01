#!/usr/bin/env python3
"""测试 ChartParseAgentADK 模块导入"""

import sys
import os

print("=== 测试模块导入 ===")

try:
    from llm import create_llm, QwenVLLL
    print("✓ llm 模块导入成功")
except Exception as e:
    print(f"✗ llm 模块导入失败: {e}")

try:
    from tools import tool_manager
    print("✓ tools 模块导入成功")
except Exception as e:
    print(f"✗ tools 模块导入失败: {e}")

try:
    from prompt import INSTRUCTION_V1, GLOBAL_PROMPT, get_prompt
    print("✓ prompt 模块导入成功")
except Exception as e:
    print(f"✗ prompt 模块导入失败: {e}")

try:
    from agent import ChartParseAgentADK, create_agent
    print("✓ agent 模块导入成功")
    
    # 测试创建实例
    agent = create_agent()
    print(f"✓ Agent 实例创建成功 (模型类型: {agent.model_type})")
    
except Exception as e:
    print(f"✗ agent 模块导入失败: {e}")

print("=== 导入测试完成 ===")