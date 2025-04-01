import asyncio
import gc
import os
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator, Generator, Dict, Optional, Union

import numpy as np
import soundfile as sf
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from models.thread_safe_base import ThreadSafeModelBase

from typing import List, Tuple, Union
from pysbd import Segmenter


def split_text_into_sentences(
    paragraphs: List[str], is_chinese: bool = True
) -> List[Union[Tuple[str], List[str]]]:
    """
    将文本切分为段落和句子

    Args:
        text: 需要切分的文本
        is_chinese: 是否为中文文本

    Returns:
        切分后的段落和句子列表
    """
    if is_chinese:
        segmenter = Segmenter(language="zh", clean=True)
    else:
        segmenter = Segmenter(language="en", clean=True)

    return [segmenter.segment(para) for para in paragraphs]


KOKORA_SPEAKER_LIST = [
    "af_maple",
    "af_sol",
    "bf_vale",
    "zf_001",
    "zf_002",
    "zf_003",
    "zf_004",
    "zf_005",
    "zf_006",
    "zf_007",
    "zf_008",
    "zf_017",
    "zf_018",
    "zf_019",
    "zf_021",
    "zf_022",
    "zf_023",
    "zf_024",
    "zf_026",
    "zf_027",
    "zf_028",
    "zf_032",
    "zf_036",
    "zf_038",
    "zf_039",
    "zf_040",
    "zf_042",
    "zf_043",
    "zf_044",
    "zf_046",
    "zf_047",
    "zf_048",
    "zf_049",
    "zf_051",
    "zf_059",
    "zf_060",
    "zf_067",
    "zf_070",
    "zf_071",
    "zf_072",
    "zf_073",
    "zf_074",
    "zf_075",
    "zf_076",
    "zf_077",
    "zf_078",
    "zf_079",
    "zf_083",
    "zf_084",
    "zf_085",
    "zf_086",
    "zf_087",
    "zf_088",
    "zf_090",
    "zf_092",
    "zf_093",
    "zf_094",
    "zf_099",
    "zm_009",
    "zm_010",
    "zm_011",
    "zm_012",
    "zm_013",
    "zm_014",
    "zm_015",
    "zm_016",
    "zm_020",
    "zm_025",
    "zm_029",
    "zm_030",
    "zm_031",
    "zm_033",
    "zm_034",
    "zm_035",
    "zm_037",
    "zm_041",
    "zm_045",
    "zm_050",
    "zm_052",
    "zm_053",
    "zm_054",
    "zm_055",
    "zm_056",
    "zm_057",
    "zm_058",
    "zm_061",
    "zm_062",
    "zm_063",
    "zm_064",
    "zm_065",
    "zm_066",
    "zm_068",
    "zm_069",
    "zm_080",
    "zm_081",
    "zm_082",
    "zm_089",
    "zm_091",
    "zm_095",
    "zm_096",
    "zm_097",
    "zm_098",
    "zm_100",
]


class KokoraTTS(ThreadSafeModelBase):
    """
    线程安全的Kokoro文本转语音模型

    继承自ThreadSafeModelBase，提供以下功能：
    - 异步流式生成
    - 同步生成
    - 线程安全的模型加载和卸载
    """

    def __init__(
        self,
        model_params: Dict[str, Any] = None,
        device: str = None,
        thread_safe: bool = True,
    ):
        """
        初始化Kokoro TTS系统

        参数:
            model_params: 模型参数字典，支持以下参数:
                - model_dir: Kokoro模型的仓库ID，默认为"hexgrad/Kokoro-82M-v1.1-zh"
            device: 模型运行设备，默认自动选择
            thread_safe: 是否启用线程安全机制，默认为True
        """
        # 设置默认参数
        default_params = {
            "model_dir": "hexgrad/Kokoro-82M-v1.1-zh",
            "language": "zh",
        }
        self.language_map = {
            "zh": "z",
            "en": "a",
        }

        # 合并默认参数和用户参数
        model_params = {**default_params, **(model_params or {})}

        # 如果设备未指定，自动选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 调用父类初始化
        super().__init__(model_params, device, thread_safe)

        # 错误跟踪
        self.error_occurred = False

        # 模型和流水线
        self.model = None
        self.kpipeline = None
        self.en_pipeline = None
        self._sample_rate = 24000

    def _load_model(self) -> None:
        """实际加载模型的方法，实现父类抽象方法"""
        self.logger.info(f"加载Kokoro TTS模型: {self.model_params['model_dir']}")

        try:
            # 初始化模型
            self.model = (
                KModel(repo_id=self.model_params["model_dir"]).to(self.device).eval()
            )

            lang_code = self.language_map[self.model_params["language"]]
            # 初始化英文流水线（用于处理英文音素）

            self.en_pipeline = KPipeline(
                lang_code="a", repo_id=self.model_params["model_dir"], model=False
            )

            # 定义英文音素处理函数
            def en_callable(text):
                if text == "Kokoro":
                    return "kˈOkəɹO"
                return next(self.en_pipeline(text)).phonemes

            self.kpipeline = KPipeline(
                lang_code=lang_code,
                repo_id=self.model_params["model_dir"],
                model=self.model,
                en_callable=en_callable,
            )

            self.logger.info("Kokoro模型和流水线初始化完成")

        except Exception as e:
            self.logger.error(f"Kokoro模型初始化失败: {e}")
            self.error_occurred = True
            raise e

    def _unload_model(self) -> None:
        """实际卸载模型的方法，实现父类抽象方法"""
        if self.model is not None:
            self.logger.info("释放Kokoro TTS模型资源")

            # 删除模型和流水线
            del self.model
            del self.kpipeline
            del self.en_pipeline

            self.model = None
            self.kpipeline = None
            self.en_pipeline = None

            # 手动触发垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _infer(self, infer_params: Dict[str, Any]) -> Any:
        """
        实际推理的方法，实现父类抽象方法

        参数:
            infer_params: 推理参数，必须包含:
                - text: 要合成的文本
                - voice: 说话人ID

        可选参数:
            - speed: 语速，默认1.0
            - silence_duration: 段落间的静音持续时间（秒），默认0.1
            - language: 语言，默认"zh"

        返回:
            生成器，每次返回一个音频块
        """
        text = infer_params.get("text")
        if text is None:
            raise ValueError("必须提供text参数")

        voice = infer_params.get("voice", "zf_001")
        speed = infer_params.get("speed", 1.0)
        silence_duration = infer_params.get("silence_duration", 0.1)
        language = infer_params.get("language", "zh")

        return self.generate_speech(text, voice, speed, silence_duration, language)

    def generate_speech(
        self,
        text: str,
        voice: str = "zf_001",
        speed: float = 1.0,
        silence_duration: float = 0.1,
        language: str = "zh",
    ):
        """
        生成完整的语音音频

        参数:
            text: 要处理的文本
            voice: 声音ID，默认使用初始化时指定的voice
            speed: 语速，默认1.0
            silence_duration: 段落间的静音持续时间（秒），默认0.1
            language: 语言，默认"zh"

        返回:
            生成器，每次返回一个音频块
        """
        # 分段处理长文本
        paragraphs = [p for p in text.split("\n") if p.strip()]
        text_segments = split_text_into_sentences(
            paragraphs, is_chinese=language == "zh"
        )
        for i, paragraph in enumerate(text_segments):
            for i, (gs, ps, audio) in enumerate(
                self.kpipeline(paragraph, voice=voice, speed=speed)
            ):
                # gs => graphemes/text
                # ps => phonemes
                # audio => audio
                yield audio.cpu().numpy()

            if i != len(text_segments) - 1:
                yield self._get_silence(silence_duration)

    def _get_silence(
        self, duration_seconds: float
    ) -> np.ndarray:
        """生成特定长度的静音"""
        return np.zeros(int(self.sample_rate * duration_seconds))

    @property
    def speaker_list(self):
        """获取支持的说话人列表"""
        return KOKORA_SPEAKER_LIST
    
    @property
    def sample_rate(self):
        """获取采样率"""
        return self._sample_rate


if __name__ == "__main__":
    tts = KokoraTTS()
    print(tts.speaker_list)
