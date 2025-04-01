import asyncio
import queue
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

import numpy as np
import pyaudio
import soundfile as sf
import torch
from kokoro import KModel, KPipeline

from loguru import logger
# 获取日志记录器


class StreamingTTS:
    """
    流式文本转语音服务类

    支持:
    - 异步流式处理和传输
    - 边处理边播放
    - 随时终止处理和播放
    """

    def __init__(
        self,
        repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh",
        voice: str = "zf_001",
        sample_rate: int = 24000,
        device: str = None,
        chunk_size: int = 4800,  # 约200ms的音频
        buffer_size: int = 128,  # 缓冲区可容纳多少个音频块
        silence_duration: float = 0.1,  # 段落间停顿时间（秒）
    ):
        """
        初始化流式TTS系统

        Args:
            repo_id: Kokoro模型的仓库ID
            voice: 声音ID
            sample_rate: 采样率
            device: 运行设备，None则自动选择
            chunk_size: 每个音频块的大小（样本数）
            buffer_size: 缓冲区大小（块数）
            silence_duration: 段落间的静音持续时间（秒）
        """
        self.repo_id = repo_id
        self.voice = voice
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.silence_duration = silence_duration

        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"初始化TTS系统，使用设备: {self.device}")

        # 初始化模型和流水线
        self.model = None
        self.zh_pipeline = None
        self.en_pipeline = None
        self._init_model()

        # 音频处理状态
        self.is_playing = False
        self.should_stop = False
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.error_occurred = False
        self.AUDIO_END_SIGNAL = "<AUDIO_END_SIGNAL>"

        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 播放器状态
        self.player_task = None
        self.play_stream = None
        self.pyaudio_instance = None

        # 添加信号处理，确保能够正确处理中断
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """处理系统信号，确保在程序终止时清理资源"""
        logger.info(f"接收到信号: {sig}，正在关闭TTS系统...")
        self.close()

    def _init_model(self):
        """初始化Kokoro模型和流水线"""
        try:
            logger.info(f"加载Kokoro模型: {self.repo_id}")
            self.model = KModel(repo_id=self.repo_id).to(self.device).eval()

            # 初始化英文流水线（用于处理英文音素）
            self.en_pipeline = KPipeline(
                lang_code="a", repo_id=self.repo_id, model=False
            )

            # 定义英文音素处理函数
            def en_callable(text):
                if text == "Kokoro":
                    return "kˈOkəɹO"
                return next(self.en_pipeline(text)).phonemes

            # 定义语速调整函数
            def speed_callable(len_ps):
                speed = 0.8
                if len_ps <= 83:
                    speed = 1
                elif len_ps < 183:
                    speed = 1 - (len_ps - 83) / 500
                return speed * 1.2  # 可根据需要调整整体语速

            # 初始化中文流水线
            self.zh_pipeline = KPipeline(
                lang_code="z",
                repo_id=self.repo_id,
                model=self.model,
                en_callable=en_callable,
            )

            self.speed_callable = speed_callable
            logger.info("模型和流水线初始化完成")

        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            self.error_occurred = True
            raise e

    def _get_silence(self, duration_seconds: float) -> np.ndarray:
        """生成特定长度的静音"""
        return np.zeros(int(self.sample_rate * duration_seconds))

    async def process_text(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """
        处理文本并异步生成音频流

        Args:
            text: 要处理的文本

        Yields:
            音频数据块
        """
        self.should_stop = False
        self.error_occurred = False

        # 分段处理长文本
        from utils import split_text_into_sentences

        paragraphs = [p for p in text.split("\n") if p.strip()]
        text_segments = split_text_into_sentences(paragraphs, is_chinese=True)

        logger.info(f"开始处理文本，共{len(text_segments)}个段落")

        # 使用两阶段处理：先处理句子，生成完整的WAV，然后分块推送到队列
        # 这样可以实现每个句子的快速处理和流式播放
        for i, paragraph in enumerate(text_segments):
            # 检查是否要停止处理
            if self.should_stop or self.error_occurred:
                logger.info("文本处理被终止")
                break

            for j, sentence in enumerate(paragraph):
                # 检查是否要停止处理
                if self.should_stop or self.error_occurred:
                    break

                logger.debug(
                    f"处理段落 {i + 1}/{len(text_segments)}, 句子 {j + 1}/{len(paragraph)}"
                )

                # 启动一个专门的任务来处理当前句子，避免阻塞流式播放
                try:
                    # 异步处理当前句子
                    loop = asyncio.get_event_loop()

                    # 使用线程池进行TTS生成以避免阻塞事件循环
                    wav = await loop.run_in_executor(
                        self.executor, lambda: self._generate_single_audio(sentence)
                    )

                    if wav is None:
                        # 处理中出现错误，跳过当前句子
                        continue

                    # 分块发送音频数据
                    async for chunk in self._chunk_audio(wav):
                        yield chunk

                except asyncio.CancelledError:
                    logger.info("音频生成任务被取消")
                    break
                except Exception as e:
                    logger.error(f"处理句子时出错: {str(e)}\n{traceback.format_exc()}")
                    self.error_occurred = True
                    continue

            # 段落之间添加停顿
            if (
                i < len(text_segments) - 1
                and not self.should_stop
                and not self.error_occurred
            ):
                silence = self._get_silence(self.silence_duration)
                async for chunk in self._chunk_audio(silence):
                    yield chunk

        # 向队列发送结束信号
        if not self.should_stop and not self.error_occurred:
            await self._try_put_to_queue(self.AUDIO_END_SIGNAL)  # 结束标记

        logger.info("文本处理完成")

    def _generate_single_audio(self, sentence: str) -> Optional[np.ndarray]:
        """生成单个句子的完整音频（在线程池中执行）"""
        try:
            if self.should_stop or self.error_occurred:
                return None

            result = next(
                self.zh_pipeline(sentence, voice=self.voice, speed=self.speed_callable)
            )

            return result.audio.cpu().numpy()
        except Exception as e:
            logger.error(f"生成音频时出错: {e}")
            self.error_occurred = True
            return None

    async def _chunk_audio(self, wav: np.ndarray) -> AsyncGenerator[np.ndarray, None]:
        """将完整音频分块并推送到队列"""
        if wav is None or len(wav) == 0:
            return

        # 分块处理音频
        for start_idx in range(0, len(wav), self.chunk_size):
            if self.should_stop or self.error_occurred:
                break

            end_idx = min(start_idx + self.chunk_size, len(wav))
            chunk = wav[start_idx:end_idx]

            # 将音频块推送到队列
            await self._try_put_to_queue(chunk)
            yield chunk

    async def _try_put_to_queue(self, item, max_retries=10):
        """尝试将项目放入队列，有最大重试次数限制"""
        retries = 0
        while (
            not self.should_stop and not self.error_occurred and retries < max_retries
        ):
            try:
                # 使用非阻塞方式尝试放入队列
                self.audio_queue.put_nowait(item)
                return True
            except queue.Full:
                # 队列已满，等待一小段时间再重试
                retries += 1
                await asyncio.sleep(0.05)

        if retries >= max_retries:
            logger.warning("将项目放入队列失败，队列可能已满")
            return False
        return False

    def _try_get_from_queue(self):
        """尝试从队列获取项目，非阻塞"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    async def text_to_speech_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        将文本转换为语音流，以字节形式返回

        Args:
            text: 要转换的文本

        Yields:
            音频数据字节流
        """
        try:
            async for audio_chunk in self.process_text(text):
                # 转换为字节流
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                yield audio_bytes
        except Exception as e:
            logger.error(f"音频流生成错误: {e}\n{traceback.format_exc()}")
            self.error_occurred = True

    async def start_playback(self, text: str) -> None:
        """
        开始播放文本的音频

        Args:
            text: 要播放的文本
        """
        if self.is_playing:
            logger.warning("已有音频正在播放，请先停止")
            return

        self.should_stop = False
        self.error_occurred = False
        self.is_playing = True

        # 清空队列
        self._clear_queue()

        # 启动生成任务，将文本处理结果直接放入队列
        generator_task = asyncio.create_task(self._process_and_enqueue(text))

        # 启动播放任务
        self.player_task = asyncio.create_task(self._play_from_queue())

        # 等待播放完成
        try:
            # 设置超时监测，防止永久阻塞
            timeout_task = asyncio.create_task(self._monitor_playback())

            done, pending = await asyncio.wait(
                [self.player_task, timeout_task], return_when=asyncio.FIRST_COMPLETED
            )

            # 取消未完成的任务
            for task in pending:
                task.cancel()

        except asyncio.CancelledError:
            logger.info("播放被取消")
        except Exception as e:
            logger.error(f"播放过程中出错: {e}\n{traceback.format_exc()}")
            self.error_occurred = True
        finally:
            self.is_playing = False

            # 取消生成任务
            if not generator_task.done():
                generator_task.cancel()

            # 确保停止播放
            self.stop()

            logger.info("播放结束")

    async def _monitor_playback(self, max_idle_time=30):
        """监控播放进度，防止卡死"""
        idle_start = None

        while self.is_playing and not self.should_stop:
            # 检查是否有错误发生
            if self.error_occurred:
                logger.error("检测到错误，停止播放")
                self.stop()
                break

            # 检查队列状态，判断是否可能卡住
            if self.audio_queue.empty() and not self.should_stop:
                if idle_start is None:
                    idle_start = time.time()
                elif time.time() - idle_start > max_idle_time:
                    logger.warning(
                        f"播放已空闲超过{max_idle_time}秒，可能卡住，强制停止"
                    )
                    self.error_occurred = True
                    self.stop()
                    break
            else:
                # 重置空闲计时器
                idle_start = None

            await asyncio.sleep(1)

    async def _process_and_enqueue(self, text: str) -> None:
        """处理文本并将音频块放入队列"""
        try:
            async for _ in self.process_text(text):
                # 音频块已在process_text中添加到队列
                pass
        except asyncio.CancelledError:
            logger.info("音频生成被取消")
        except Exception as e:
            logger.error(f"处理文本并填充队列时发生错误: {e}\n{traceback.format_exc()}")
            self.error_occurred = True
            # 确保放入结束标记，以便播放线程能够退出
            await self._try_put_to_queue(self.AUDIO_END_SIGNAL)

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio回调函数，提供实时音频数据"""
        if status:
            logger.warning(f"PyAudio回调状态: {status}")

        if self.should_stop or self.error_occurred:
            return (None, pyaudio.paComplete)

        try:
            # 从队列获取音频块
            chunk = self._try_get_from_queue()

            if chunk is None:
                # 队列为空，但不一定是结束
                # 返回静音，继续播放
                return (
                    np.zeros(frame_count, dtype=np.float32).tobytes(),
                    pyaudio.paContinue,
                )

            # AUDIO_END_SIGNAL作为特殊标记，表示播放结束
            if chunk is self.AUDIO_END_SIGNAL:
                return (None, pyaudio.paComplete)

            if isinstance(chunk, torch.Tensor):
                chunk = chunk.cpu().numpy().astype(np.float32)

            # 确保我们有足够的数据，如果不够，补零
            if len(chunk) < frame_count:
                # 补零
                padding = np.zeros(frame_count - len(chunk), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, padding])
            elif len(chunk) > frame_count:
                # 只取需要的部分
                chunk = chunk[:frame_count]

            # 返回音频数据
            return (chunk.astype(np.float32).tobytes(), pyaudio.paContinue)

        except Exception as e:
            logger.error(f"PyAudio回调出错: {e}")
            self.error_occurred = True
            return (None, pyaudio.paAbort)

    async def _play_from_queue(self) -> None:
        """从队列播放音频的异步任务"""
        try:
            # 创建PyAudio实例
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()

            # 创建音频流
            self.play_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._pyaudio_callback,
            )

            logger.info("开始播放音频")

            # 启动流
            self.play_stream.start_stream()

            # 等待流播放完成或被停止
            max_retries = 300  # 最多等待30秒
            retry_count = 0

            while (
                not self.should_stop
                and not self.error_occurred
                and self.play_stream
                and self.play_stream.is_active()
            ):
                await asyncio.sleep(0.1)
                retry_count += 1

                # 如果队列为空且已经很长时间没有数据，检查是否应该结束
                if self.audio_queue.empty() and retry_count > max_retries:
                    logger.warning("播放队列长时间为空，可能已经播放完成或出现问题")
                    break

                # 如果收到新数据，重置计数器
                if not self.audio_queue.empty():
                    retry_count = 0

        except Exception as e:
            logger.error(f"播放音频时出错: {e}\n{traceback.format_exc()}")
            self.error_occurred = True
        finally:
            # 清理资源
            self._cleanup_audio_resources()
            logger.info("音频播放结束")

    def _cleanup_audio_resources(self):
        """清理音频相关资源"""
        if self.play_stream:
            try:
                if self.play_stream.is_active():
                    self.play_stream.stop_stream()
                self.play_stream.close()
                self.play_stream = None
            except Exception as e:
                logger.error(f"关闭音频流时出错: {e}")

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            except Exception as e:
                logger.error(f"终止PyAudio实例时出错: {e}")

        logger.info("TTS系统已关闭")

    def _clear_queue(self):
        """清空音频队列"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        """停止所有正在进行的TTS处理和播放"""
        logger.info("停止TTS处理和播放")
        self.should_stop = True

        # 停止音频流
        if self.play_stream and self.play_stream.is_active():
            try:
                self.play_stream.stop_stream()
            except Exception as e:
                logger.error(f"停止音频流时出错: {e}")

        # 取消播放任务
        if self.player_task and not self.player_task.done():
            try:
                self.player_task.cancel()
            except Exception as e:
                logger.error(f"取消播放任务时出错: {e}")

        # 清空队列
        self._clear_queue()

        self.is_playing = False

    async def save_to_file(self, text: str, output_file: Union[str, Path]) -> Path:
        """
        将文本转换为语音并保存到文件

        Args:
            text: 要转换的文本
            output_file: 输出文件路径

        Returns:
            保存的文件路径
        """
        output_path = Path(output_file)

        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 收集所有音频片段
        audio_chunks = []
        try:
            async for chunk in self.process_text(text):
                audio_chunks.append(chunk)

            # 如果处理被中断，检查是否有收集到的音频
            if not audio_chunks:
                logger.warning("没有生成任何音频，保存文件失败")
                return None

            # 合并音频片段并保存
            full_audio = np.concatenate(audio_chunks)
            sf.write(output_path, full_audio, self.sample_rate)

            logger.info(f"音频已保存到: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"保存音频到文件时出错: {e}\n{traceback.format_exc()}")
            self.error_occurred = True
            return None

    def close(self) -> None:
        """关闭TTS系统并释放资源"""
        logger.info("关闭TTS系统并释放资源")
        self.stop()

        # 确保PyAudio资源被正确释放
        if self.play_stream:
            try:
                if self.play_stream.is_active():
                    self.play_stream.stop_stream()
                self.play_stream.close()
                self.play_stream = None
            except Exception as e:
                logger.error(f"关闭音频流时出错: {e}")

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            except Exception as e:
                logger.error(f"终止PyAudio实例时出错: {e}")

        # 关闭线程池
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"关闭线程池时出错: {e}")

        logger.info("TTS系统已关闭")


# 示例用法
async def example_usage():
    # 初始化TTS系统
    tts = StreamingTTS(buffer_size=100)

    # 示例文本
    text = """
    Kokoro 是一系列体积虽小但功能强大的 TTS 模型。
    经过短期训练，该模型已融入来自专业数据集的100名中文使用者的语言习惯和表达方式，从而在特定领域的中文文本处理能力上得到了显著提升。
    衷心感谢专业数据集公司「龙猫数据」慷慨地免费提供中文数据，他们的无私奉献为该模型的成功训练奠定了坚实基础，并使得模型在中文处理能力上取得了显著进展，谨此致谢，是你们的帮助让模型成为可能。
    """

    try:
        # 异步播放文本
        await tts.start_playback(text)

        # 将文本保存为音频文件
        await tts.save_to_file("这是一个测试。", "output/test.wav")
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
    finally:
        # 确保关闭TTS系统
        tts.close()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
