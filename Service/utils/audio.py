import io
import tempfile
import os
import torchaudio
import soundfile as sf
import numpy as np
import torch
from typing import Union
from utils.logger import default_logger as logger

def convert_audio_format(audio_data: Union[np.ndarray, torch.Tensor], 
                         sample_rate: int = 16000, 
                         output_format: str = "mp3") -> bytes:
    """
    将音频数据转换为指定格式的二进制数据
    
    参数:
        audio_data: 音频数据（numpy数组或PyTorch张量）
        sample_rate: 采样率，默认16000Hz
        output_format: 输出格式，支持mp3和wav，默认为mp3
    
    返回:
        二进制音频数据
    """
    try:
        # 确保采样率是有效的
        if not sample_rate or sample_rate <= 0:
            logger.warning(f"检测到无效采样率: {sample_rate}，使用默认值16000")
            sample_rate = 16000
            
        # 确保音频数据是numpy数组
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        
        logger.info(f"音频数据形状: {audio_data.shape}, 类型: {audio_data.dtype}, 采样率: {sample_rate}")
        
        # 确保音频数据是float32类型，并且值在-1到1之间
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # 规范化音频数据
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # 确保音频数据形状正确(samples, channels)
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, 1)  # 单声道
        
        audio_data = audio_data.T
        # 直接返回WAV格式数据
        if output_format.lower() == "wav":
            logger.info("正在生成WAV格式音频")
            # 使用BytesIO直接保存为wav
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            wav_buffer.seek(0)
            return wav_buffer.read()
        
        # 处理MP3格式
        elif output_format.lower() == "mp3":
            logger.info("正在生成MP3格式音频")
            
            # 创建临时WAV文件
            wav_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_temp.close()
            
            try:
                # 保存为WAV
                sf.write(wav_temp.name, audio_data, sample_rate, format="WAV", subtype="PCM_16")
                logger.info(f"临时WAV文件已创建: {wav_temp.name}")
                
                # 确认文件已正确创建
                if not os.path.exists(wav_temp.name) or os.path.getsize(wav_temp.name) == 0:
                    raise ValueError(f"临时WAV文件创建失败或为空: {wav_temp.name}")
                
                # 创建临时MP3文件
                mp3_temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                mp3_temp.close()
                
                # 使用torchaudio将WAV转换为MP3
                try:
                    # 加载WAV并保存为MP3
                    waveform, sr = torchaudio.load(wav_temp.name)
                    torchaudio.save(mp3_temp.name, waveform, sr, format="mp3")
                    logger.info(f"转换为MP3成功: {mp3_temp.name}")
                    
                    # 读取MP3数据
                    with open(mp3_temp.name, "rb") as f:
                        mp3_data = f.read()
                        
                    return mp3_data
                except Exception as e:
                    logger.error(f"转换为MP3失败，尝试返回WAV格式: {str(e)}")
                    # MP3转换失败，返回WAV格式
                    with open(wav_temp.name, "rb") as f:
                        wav_data = f.read()
                    return wav_data
                finally:
                    # 清理MP3临时文件
                    if os.path.exists(mp3_temp.name):
                        os.remove(mp3_temp.name)
            finally:
                # 确保WAV临时文件被清理
                if os.path.exists(wav_temp.name):
                    os.remove(wav_temp.name)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    except Exception as e:
        logger.error(f"音频转换失败: {str(e)}")
        raise ValueError(f"音频转换失败: {str(e)}")
