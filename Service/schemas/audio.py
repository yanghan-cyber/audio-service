from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Any



class SpeechRequest(BaseModel):
    """语音合成请求"""
    
    input: str = Field(..., description="要合成的文本")
    voice: str = Field(..., description="说话人ID")
    model: str = Field("cosyvoice", description="TTS模型类型，可选值: cosyvoice, kokora")
    speed: float = Field(1.0, description="语速调整因子", ge=0.5, le=2.0)
    response_format: str = Field("mp3", description="音频输出格式")
    instruct: Optional[str] = Field(None, description="指令描述（仅用于cosyvoice模型的instruct模式）")
    random_seed: Optional[int] = Field(None, description="随机种子，用于控制生成的一致性")
    
    # kokora模型特有参数
    language: Optional[str] = Field("zh", description="语言代码，用于kokora模型")
    silence_duration: Optional[float] = Field(0.1, description="段落间的静音持续时间，用于kokora模型")


class TranscriptionResponse(BaseModel):
    """语音转文本响应"""
    
    text: str = Field(..., description="转录的文本")
