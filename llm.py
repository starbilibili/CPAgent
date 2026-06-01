import os
from typing import Optional

from config import QWEN_VL_MODEL_PATH


class BaseLLM:
    """LLM 基类，定义统一接口，便于替换不同模型"""

    def __init__(self, model_name: str = None, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def call(self, prompt: str, image: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    def __call__(self, prompt: str, image: Optional[str] = None, **kwargs) -> str:
        return self.call(prompt, image, **kwargs)


class QwenVLLL(BaseLLM):
    """Qwen-VL 模型封装，基于 CPAgent 中的 Qwen3VL 调用方式"""

    def __init__(self, model_path: str = None, **kwargs):
        import torch

        model_path = model_path or str(QWEN_VL_MODEL_PATH)
        super().__init__(model_path, **kwargs)
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """加载 Qwen3VL 模型"""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            import torch

            print(f"[QwenVLLL] 加载模型: {self.model_path}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype="auto", device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception as e:
            print(f"[QwenVLLL] 加载模型失败: {e}")
            raise

    def construct_messages(self, prompt: str, image_path: str):
        """构造 Qwen3VL 的消息格式"""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def call(self, prompt: str, image: Optional[str] = None, **kwargs) -> str:
        import torch

        if not image:
            inputs = self.processor(prompt, return_tensors="pt", padding=True).to(self.device)
        else:
            messages = self.construct_messages(prompt, image)
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, **kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw = output_text[0]
            if "<answer>" in raw and "</answer>" in raw:
                return raw.split("<answer>")[1].split("</answer>")[0]
            return raw


def create_llm(llm_type: str = "qwen-vl", **kwargs) -> BaseLLM:
    """
    创建 LLM 实例

    Args:
        llm_type: 目前仅支持 "qwen-vl"
        **kwargs: 传递给 QwenVLLL 的参数，如 model_path
    """
    if llm_type.lower() == "qwen-vl":
        return QwenVLLL(**kwargs)
    raise ValueError(f"Unsupported LLM type: {llm_type}")
