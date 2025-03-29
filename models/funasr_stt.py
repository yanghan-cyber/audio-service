from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
from typing import Dict, Any
import emoji
from models.thread_safe_base import ThreadSafeModelBase


class FunASRSTT(ThreadSafeModelBase):
    """
    语音转文字模型，基于FunASR实现
    
    继承自ThreadSafeModelBase，自动具备线程安全的特性
    """
    
    def __init__(
        self,
        model_params: Dict[str, Any] = None,
        device: str = None,
        thread_safe: bool = True
    ):
        """
        初始化语音转文字模型
        
        参数:
            model_params: 模型参数字典，支持以下参数:
                - model_dir: 模型名称或本地磁盘中的模型路径
                - vad_model: VAD模型名称，将长音频切割成短音频
                - vad_max_segment_time: VAD最大切割音频时长(毫秒)
                - use_itn: 是否包含标点与逆文本正则化
                - batch_size_s: 动态batch中总音频时长(秒)
                - merge_vad: 是否合并VAD切割的短音频碎片
                - merge_length_s: 合并后长度(秒)
                - ban_emo_unk: 是否禁用emo_unk标签
            device: 模型运行设备，默认自动选择
            thread_safe: 是否启用线程安全机制，默认为True
        """
        # 设置默认参数
        default_params = {
            "model_dir": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "vad_max_segment_time": 60000,
            "use_itn": True,
            "batch_size_s": 60,
            "merge_vad": True,
            "merge_length_s": 15,
            "ban_emo_unk": True,
        }
        
        # 合并默认参数和用户参数
        model_params = {**default_params, **(model_params or {})}
        
        # 如果设备未指定，自动选择
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # 调用父类初始化
        super().__init__(model_params, device, thread_safe)
    
    def _load_model(self) -> None:
        """实际加载模型的方法，实现父类抽象方法"""
        self.logger.info(f"加载语音识别模型 {self.model_params['model_dir']}")

        # 设置VAD参数
        vad_kwargs = (
            {"max_single_segment_time": self.model_params["vad_max_segment_time"]}
            if self.model_params["vad_model"]
            else {}
        )

        # 初始化模型
        self.model = AutoModel(
            model=self.model_params["model_dir"],
            vad_model=self.model_params["vad_model"],
            vad_kwargs=vad_kwargs,
            device=self.device
        )
            
        self.logger.info("语音识别模型加载完成")
    
    def _infer(self, infer_params: Dict[str, Any]) -> Any:
        """
        实际推理的方法，实现父类抽象方法
        
        参数:
            infer_params: 推理参数，包含:
                - audio_file_path: 音频文件路径或者音频数据
                - language: 语言选择，默认为auto
                - cache: 缓存字典
                
        返回:
            处理后的文本
        """
        audio_file_path = infer_params.get("audio_file_path")
        language = infer_params.get("language", "auto")
        cache = infer_params.get("cache", {})
            
        # 调用模型进行推理
        res = self.model.generate(
            input=audio_file_path,
            cache=cache,
            language=language,
            use_itn=self.model_params["use_itn"],
            batch_size_s=self.model_params["batch_size_s"],
            merge_vad=self.model_params["merge_vad"],
            merge_length_s=self.model_params["merge_length_s"],
            ban_emo_unk=self.model_params["ban_emo_unk"],
        )

        # 处理结果
        text = rich_transcription_postprocess(res[0]["text"])
        
        if self.model_params["ban_emo_unk"]:
            text = emoji.replace_emoji(text, "")
        return text
    
    def _unload_model(self) -> None:
        """实际卸载模型的方法，实现父类抽象方法"""
        if self.model is not None:
            self.logger.info("释放语音识别模型资源")
            del self.model
            self.model = None
            
            # 手动触发垃圾回收
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def transcribe(self, audio_file_path, language="auto", cache=None):
        """
        将音频文件转换为文字，便捷方法
        
        参数:
            audio_file_path: 音频文件路径或者音频数据
            language: 语言选择，默认为自动检测
            cache: 缓存字典，默认为空字典
            
        返回:
            处理后的文本
        """
        infer_params = {
            "audio_file_path": audio_file_path,
            "language": language,
            "cache": cache or {}
        }
        return self.infer(infer_params)


# 示例用法
if __name__ == "__main__":
    # 初始化转写器
    stt = FunASRSTT(
        model_params={"model_dir": "iic/SenseVoiceSmall"},
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        thread_safe=True  # 默认启用线程安全
    )

    import os
    import sys

    # 增加当前目录到python路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    # 增加上一级目录到python路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 检查模型是否已加载
    print(f"模型加载状态: {stt.is_loaded}")

    # 加载模型
    stt.load()
    print(f"模型加载状态: {stt.is_loaded}")
    print(f"模型路径: {stt.model.model_path}")

    # 准备示例音频文件路径
    try:
        # 尝试使用模型内置的示例
        example_file = f"{stt.model.model_path}/example/zh.mp3"

        # 转写音频
        text = stt.transcribe(example_file)
        print(f"转写结果: {text}")

        # 手动卸载模型
        print("手动卸载模型")
        stt.unload()
        print(f"模型加载状态: {stt.is_loaded}")

        # 禁用线程安全测试
        print("\n禁用线程安全测试")
        stt.set_thread_safe(False)
        print(f"线程安全状态: {stt.thread_safe}")
        
        # 再次加载模型
        stt.load()
        text = stt.transcribe(example_file)
        print(f"转写结果 (非线程安全模式): {text}")
        stt.unload()

    except Exception as e:
        print(f"示例运行出错: {e}") 