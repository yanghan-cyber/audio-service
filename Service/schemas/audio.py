from pydantic import BaseModel, Field
from typing import Literal, Optional



class SpeechRequest(BaseModel):
    """文本转语音请求"""
    input: str = Field(..., description="The text to generate audio for")
    model: str = Field("iic/CosyVoice2-0.5B", description="The model to use for audio generation")
    voice: str = Field("wendy", description="The voice to use for audio generation")
    instruct: Optional[str] = Field(None, description="Control the voice of your generated audio with additional instruct.")
    response_format: Literal["mp3", "wav", "aac"] = Field("wav", description="The format of the audio generation")
    speed: float = Field(1.0, description="The speed of the audio generation")


class TranscriptionResponse(BaseModel):
    """语音转文本响应"""
    text: str = Field(..., description="The transcribed text")
