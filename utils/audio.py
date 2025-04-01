import io
import numpy as np
import torch
from pydub import AudioSegment
from typing import Union

def convert_audio_format(audio_data: Union[np.ndarray, torch.Tensor], 
                         sample_rate: int = 16000, 
                         output_format: str = "mp3") -> bytes:
    """
    将音频数据转换为指定格式的二进制数据（完全内存操作版本）
    
    参数:
        audio_data: 音频数据（numpy数组或PyTorch张量）
        sample_rate: 采样率，默认16000Hz
        output_format: 输出格式，支持:
            - 有损压缩: mp3, ogg, aac, m4a, webm
            - 无损压缩: flac, wav
            默认为mp3
    
    返回:
        二进制音频数据
    
    依赖:
        pip install pydub numpy
        需要安装FFmpeg并添加到系统PATH，且需要支持以下编解码器：
        - libmp3lame (MP3)
        - libvorbis (OGG)
        - aac
        - libopus (WebM)
        - flac
    """
    # 输入验证
    if not isinstance(audio_data, (np.ndarray, torch.Tensor)):
        raise TypeError("输入必须是numpy数组或PyTorch张量")
    
    if sample_rate <= 0:
        raise ValueError("采样率必须为正数")
    
    # 转换为numpy数组并确保是float32类型
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 规范化音频数据到[-1, 1]范围
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0 or max_val == 0.0:
        audio_data = audio_data / (max_val + 1e-7)  # 避免除以零
    
    # 处理单声道/立体声数据
    if len(audio_data.shape) == 1:
        audio_data = np.expand_dims(audio_data, axis=0)  # 转为(1, samples)
    elif len(audio_data.shape) > 2:
        raise ValueError("音频数据维度过多，应为(samples,)或(channels, samples)")
    
    # 转换为pydub需要的格式
    channels = audio_data.shape[0]
    
    # 将float32转换为int16 (pydub需要)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    # 创建AudioSegment对象
    try:
        audio_segment = AudioSegment(
            audio_data_int16.tobytes(), 
            frame_rate=sample_rate,
            sample_width=2,  # int16是2字节
            channels=channels
        )
    except Exception as e:
        raise ValueError(f"创建AudioSegment失败: {str(e)}")
    
    # 根据输出格式导出
    output_format = output_format.lower()
    supported_formats = {"mp3", "wav", "aac", "m4a", "ogg", "flac", "webm"}
    if output_format not in supported_formats:
        raise ValueError(f"不支持的输出格式: {output_format}。支持: {supported_formats}")
    
    # 使用BytesIO在内存中处理
    buffer = io.BytesIO()
    
    try:
        # 无损格式
        if output_format == "wav":
            audio_segment.export(buffer, format="wav")
        elif output_format == "flac":
            audio_segment.export(
                buffer,
                format="flac",
                parameters=["-compression_level", "8"]  # 最高压缩率
            )
        # 有损格式
        elif output_format == "mp3":
            audio_segment.export(
                buffer,
                format="mp3",
                bitrate="192k",
                parameters=["-q:a", "0"]  # 最高质量
            )
        elif output_format == "ogg":
            audio_segment.export(
                buffer,
                format="ogg",
                codec="libvorbis",
                parameters=["-q:a", "10"]  # Vorbis质量等级0-10
            )
        elif output_format == "webm":
            audio_segment.export(
                buffer,
                format="webm",
                codec="libopus",
                bitrate="128k",
                parameters=[
                    "-application", "audio",
                    "-frame_duration", "20"
                ]
            )
        elif output_format == "aac":
            audio_segment.export(
                buffer, 
                format="adts",  # 使用ADTS容器格式，可直接作为AAC文件播放
                codec="aac",
                bitrate="192k",
                parameters=[
                    "-strict", "-2",
                    "-aac_coder", "twoloop"  # 使用双循环AAC编码器提高质量
                ]
            )
        elif output_format == "m4a":
            audio_segment.export(
                buffer, 
                format="ipod",  # 对于m4a使用ipod格式
                codec="aac",
                bitrate="192k",
                parameters=[
                    "-strict", "-2",
                    "-movflags", "faststart"  # 优化网络流式播放
                ]
            )
        
        buffer.seek(0)
        return buffer.read()
    
    except Exception as e:
        raise RuntimeError(f"音频导出失败: {str(e)}")
    finally:
        buffer.close()
