from functools import lru_cache
import json
import torch
import time
import gc
import torchaudio
import os
from typing import Dict, Any, Generator, Tuple

from models.thread_safe_base import ThreadSafeModelBase
from third_party import CosyVoice2, load_wav, set_all_random_seed


class CosyVoiceTTS(ThreadSafeModelBase):
    """
    线程安全的CosyVoice语音合成模型

    继承自ThreadSafeModelBase，自动具备线程安全的特性
    """

    def __init__(
        self,
        model_params: Dict[str, Any] = None,
        device: str = None,
        thread_safe: bool = True,
    ):
        """
        初始化语音合成模型

        参数:
            model_params: 模型参数字典，支持以下参数:
                - model_dir: 模型名称或本地磁盘中的模型路径，默认为"iic/CosyVoice2-0.5B"
                - load_jit: 是否使用JIT加载模型，默认为False
                - load_trt: 是否使用TensorRT加载模型，默认为False
                - fp16: 是否使用半精度浮点数，默认为False
            device: 模型运行设备，默认自动选择
            thread_safe: 是否启用线程安全机制，默认为True
        """
        # 设置默认参数
        default_params = {
            "model_dir": "iic/CosyVoice2-0.5B",
            "load_jit": False,
            "load_trt": False,
            "fp16": False,
        }

        # 合并默认参数和用户参数
        model_params = {**default_params, **(model_params or {})}

        # 如果设备未指定，自动选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 调用父类初始化
        super().__init__(model_params, device, thread_safe)

        self.file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asset"
        )
        self.speaker_json = json.load(
            open(os.path.join(self.file_path, "speaker.json"), "r", encoding="utf-8")
        )
        self.prompt_sr = 16_000

    def _load_model(self) -> None:
        """实际加载模型的方法，实现父类抽象方法"""
        self.logger.info(f"加载语音合成模型 {self.model_params['model_dir']}")

        # 初始化模型
        self.model = CosyVoice2(
            self.model_params["model_dir"],
            load_jit=self.model_params["load_jit"],
            load_trt=self.model_params["load_trt"],
            fp16=self.model_params["fp16"],
        )

        self.logger.info("语音合成模型加载完成")

    def _infer(self, infer_params: Dict[str, Any]) -> Any:
        """
        实际推理的方法，实现父类抽象方法

        参数:
            infer_params: 推理参数，必须包含:
                - text: 要合成的文本
                - voice: 说话人ID
                - mode: 合成模式，可选值: "zero_shot", "cross_lingual", "instruct"
            可选参数:
                - instruct: 指令模式下的指令，如"用四川话说这句话"
                - stream: 是否流式输出，默认False
                - speed: 合成速度，默认1.0

        返回:
            始终返回音频数据生成器
        """
        # 解析参数
        text = infer_params.get("text")
        if text is None:
            raise ValueError("必须提供text参数")

        voice = infer_params.get("voice")
        if voice is None:
            raise ValueError("必须提供voice参数")

        mode = infer_params.get("mode", "zero_shot")
        instruct = infer_params.get("instruct")
        stream = infer_params.get("stream", False)
        speed = infer_params.get("speed", 1.0)

        # 根据模式选择不同的合成方法
        if mode == "zero_shot":
            return self.synthesize_zero_shot(text, voice, stream, speed)
        elif mode == "cross_lingual":
            return self.synthesize_cross_lingual(
                text, voice, stream, speed
            )
        elif mode == "instruct":
            if instruct is None:
                raise ValueError("instruct模式需要提供instruct参数")
            return self.synthesize_instruct(
                text, voice, instruct, stream, speed
            )
        else:
            raise ValueError(
                f"不支持的合成模式: {mode}，可选值为: zero_shot, cross_lingual, instruct"
            )

    def _unload_model(self) -> None:
        """实际卸载模型的方法，实现父类抽象方法"""
        if self.model is not None:
            self.logger.info("释放语音合成模型资源")

            # 删除模型
            del self.model
            self.model = None

            # 手动触发垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    @lru_cache(maxsize=16)
    def _load_prompt(self, voice: str) -> Tuple[str, torch.Tensor]:
        """
        加载提示音频或文本，并返回对应的音频数据

        参数:
            voice: 说话人ID

        返回:
            元组 (提示文本, 提示音频)
        """
        if voice in self.speaker_json:
            prompt_path = self.speaker_json[voice]["path"]
            prompt_text = self.speaker_json[voice]["prompt_text"]
            prompt_speech = load_wav(
                os.path.join(self.file_path, prompt_path), self.prompt_sr
            )
            return prompt_text, prompt_speech
        else:
            raise ValueError(f"说话人ID {voice} 不存在")

    def synthesize_zero_shot(
        self, text, voice, stream=False, speed=1.0
    ) -> Generator[torch.Tensor, None, None]:
        """
        零样本语音合成

        Args:
            text: 要合成的文本
            voice: 说话人ID
            stream: 是否流式输出
            speed: 合成速度

        Returns:
            语音数据生成器
        """
        # 确保模型已加载
        if not self.is_loaded:
            self.load()

        prompt_text, prompt_speech = self._load_prompt(voice)
        for i, result in enumerate(
            self.model.inference_zero_shot(
                text, prompt_text, prompt_speech, stream=stream, speed=speed
            )
        ):
            yield result["tts_speech"]

    def synthesize_cross_lingual(
        self, text, voice, stream=False, speed=1.0
    ) -> Generator[torch.Tensor, None, None]:
        """
        跨语言语音合成

        Args:
            text: 要合成的文本
            voice: 说话人ID
            stream: 是否流式输出
            speed: 合成速度

        Returns:
            语音数据生成器
        """
        # 确保模型已加载
        if not self.is_loaded:
            self.load()

        prompt_text, prompt_speech = self._load_prompt(voice)
        for i, result in enumerate(
            self.model.inference_cross_lingual(
                text, prompt_speech, stream=stream, speed=speed
            )
        ):
            yield result["tts_speech"]

    def synthesize_instruct(
        self, text, voice, instruct, stream=False, speed=1.0
    ) -> Generator[torch.Tensor, None, None]:
        """
        指令语音合成

        Args:
            text: 要合成的文本
            voice: 说话人ID
            instruct: 合成指令，如"用四川话说这句话"
            stream: 是否流式输出
            speed: 合成速度

        Returns:
            语音数据生成器
        """
        # 确保模型已加载
        if not self.is_loaded:
            self.load()

        prompt_text, prompt_speech = self._load_prompt(voice)
        for i, result in enumerate(
            self.model.inference_instruct2(
                text, instruct, prompt_speech, stream=stream, speed=speed
            )
        ):
            yield result["tts_speech"]

    def synthesize(
        self,
        text,
        voice,
        mode="zero_shot",
        instruct=None,
        stream=False,
        speed=1.0,
    ):
        """
        通用语音合成接口（便捷方法）

        参数:
            text: 要合成的文本
            voice: 说话人ID
            mode: 合成模式，可选值："zero_shot", "cross_lingual", "instruct"
            instruct: 指令模式下的指令，如"用四川话说这句话"
            stream: 是否流式输出，默认False
            speed: 合成速度，默认1.0

        返回:
            音频数据生成器
        """
        infer_params = {
            "text": text,
            "voice": voice,
            "mode": mode,
            "instruct": instruct,
            "stream": stream,
            "speed": speed,
        }
        return self.infer(infer_params)

    def save_audio(self, audio_data, file_path, sample_rate=None):
        """
        保存音频数据到文件

        参数:
            audio_data: 音频数据
            file_path: 文件路径
            sample_rate: 采样率，默认使用模型采样率
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        torchaudio.save(file_path, audio_data, sample_rate)
        self.logger.info(f"已保存音频到: {file_path}")

    @property
    def sample_rate(self):
        """获取模型采样率"""
        # 确保模型已加载
        if not self.is_loaded:
            self.load()
        return self.model.sample_rate

    @property
    def status(self):
        """获取模型状态信息"""
        status = {
            "loaded": self.is_loaded,
            "model_id": self.model_id,
            "model_dir": self.model_params.get("model_dir"),
            "device": self.device,
            "thread_safe": self.thread_safe,
            "speakers_count": len(self.speaker_json)
            if hasattr(self, "speaker_json")
            else 0,
        }
        return status

    @property
    def speaker_list(self):
        """获取支持的说话人列表"""
        speakers = []
        for speaker in self.speaker_json:
            speakers.append({
                "id": speaker,
                "name": self.speaker_json[speaker]["name"],
                "gender": self.speaker_json[speaker]["gender"], 
            })
        return speakers
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        set_all_random_seed(seed)


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    import numpy as np
    from pathlib import Path

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CosyVoice TTS 流式合成测试")
    parser.add_argument(
        "--text",
        type=str,
        default="这是一个测试语音合成的示例文本，让我们看看流式合成的效果如何。",
        help="要合成的文本",
    )
    parser.add_argument("--speaker", type=str, default=None, help="说话人ID")
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "cross_lingual", "instruct"],
        help="合成模式",
    )
    parser.add_argument(
        "--instruct", type=str, default="用开心的语气说这句话", help="指令模式下的指令"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="语速")
    parser.add_argument("--output", type=str, default="output.wav", help="输出文件路径")
    args = parser.parse_args()

    # 初始化语音合成模型
    tts = CosyVoiceTTS()

    try:
        # 加载模型
        tts.load()

        # 如果未指定说话人，使用第一个可用的说话人
        if args.speaker is None:
            speakers = tts.speaker_list
            if not speakers:
                raise ValueError("找不到可用的说话人")
            # 获取第一个说话人的ID
            first_speaker = speakers[0]
            speaker_id = tts.speaker_json[first_speaker][0]["id"]
            voice = f"{first_speaker}-{speaker_id}"
            print(f"未指定说话人，使用默认说话人: {voice}")
        else:
            voice = args.speaker

        print(f"正在使用流式方式合成文本: '{args.text}'")
        print(f"说话人: {voice}")
        print(f"合成模式: {args.mode}")
        if args.mode == "instruct":
            print(f"指令: {args.instruct}")
        print(f"语速: {args.speed}")

        # 准备输出目录
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 流式合成并保存
        start_time = time.time()
        audio_chunks = []

        # 使用通用接口进行合成
        speech_generator = tts.synthesize(
            text=args.text,
            voice=voice,
            mode=args.mode,
            instruct=args.instruct if args.mode == "instruct" else None,
            stream=True,  # 启用流式输出
            speed=args.speed,
        )

        # 模拟实时处理流式输出
        for i, chunk in enumerate(speech_generator):
            # 将张量转换为NumPy数组
            chunk_np = chunk.cpu().numpy()
            audio_chunks.append(chunk_np)
            # 显示进度
            print(f"接收到第 {i + 1} 个音频片段，长度: {len(chunk_np)} 样本")
            # 模拟实时处理，比如这里可以播放或处理每个片段
            # 这里只是演示，实际应用中可能需要更复杂的处理

        # 合并所有音频片段
        full_audio = np.concatenate(audio_chunks, axis=1)

        # 保存合并后的音频
        sf.write(args.output, full_audio.T, tts.sample_rate)

        elapsed_time = time.time() - start_time
        print(f"合成完成! 耗时: {elapsed_time:.2f}秒")
        print(f"音频已保存到: {output_path.absolute()}")
        print(
            f"采样率: {tts.sample_rate}Hz, 音频长度: {full_audio.shape[1] / tts.sample_rate:.2f}秒"
        )

    except Exception as e:
        print(f"合成过程中发生错误: {e}")

    finally:
        # 释放资源
        tts.unload()
        print("已释放TTS资源")
