import tempfile
from models.funasr_stt import FunASRSTT
from models.cosyvoice_tts import CosyVoiceTTS

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional, Iterator
import torch
from Service.schemas.audio import SpeechRequest, TranscriptionResponse
from Service.utils.audio import convert_audio_format

STT_MODEL = FunASRSTT()
TTS_MODEL = CosyVoiceTTS()

STT_MODEL.load()
TTS_MODEL.load()

# 设置模板目录
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# 创建templates实例
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def get_stt_model():
    global STT_MODEL
    return STT_MODEL

def get_tts_model():
    global TTS_MODEL
    return TTS_MODEL

router = APIRouter(
    tags=["Audio"],
    prefix="/audio",
    responses={404: {"description": "Not Found"}}
)

@router.get("/ui", response_class=HTMLResponse)
async def get_ui(request: Request):
    """
    返回语音合成UI页面
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.get("/voices")
async def get_voices():
    tts_model = get_tts_model()
    return tts_model.speaker_list


@router.post("/speech", response_class=Response)
async def speech(request: SpeechRequest):
    try:
        tts_model = get_tts_model()
        
        infer_params = {
            "text": request.input,
            "voice": request.voice,
            "instruct": request.instruct,
            "speed": request.speed,
        }
        
        if request.instruct:
            infer_params["mode"] = "instruct"
        
        else:
            infer_params["mode"] = "zero_shot"

        audio_generator = tts_model.infer(infer_params)

        all_audio = torch.cat([chunk for chunk in audio_generator], dim=0)
        sample_rate = tts_model.sample_rate

        audio_binary = convert_audio_format(all_audio, sample_rate, request.response_format)
        content_type = "audio/mpeg" if request.response_format == "mp3" else "audio/wav"

        return Response(
            content=audio_binary,
            media_type=content_type
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"语音生成失败: {str(e)}"
        )

@router.post("/stream_speech")
async def stream_speech(request: SpeechRequest):
    """
    流式文本转语音接口，实时返回生成的语音片段
    """
    try:
        tts_model = get_tts_model()
        
        infer_params = {
            "text": request.input,
            "voice": request.voice,
            "instruct": request.instruct,
            "speed": request.speed,
            "stream": True,  # 启用流式输出
        }
        
        if request.instruct:
            infer_params["mode"] = "instruct"
        else:
            infer_params["mode"] = "zero_shot"

        # 创建音频流生成器函数
        def generate_audio_chunks() -> Iterator[bytes]:
            sample_rate = tts_model.sample_rate
            audio_generator = tts_model.infer(infer_params)
            
            for chunk in audio_generator:
                # 将每个音频片段转换为指定格式的二进制数据
                audio_binary = convert_audio_format(chunk, sample_rate, request.response_format)
                yield audio_binary
        
        content_type = "audio/mpeg" if request.response_format == "mp3" else "audio/aac" if request.response_format == "aac" else "audio/wav"
        
        return StreamingResponse(
            generate_audio_chunks(),
            media_type=content_type
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"流式语音生成失败: {str(e)}"
        )

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("iic/SenseVoiceSmall", description="要使用的模型ID"),
    language: str = Form("auto"),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("text"),
    background_tasks: BackgroundTasks = None
):
    """
    将上传的音频文件转录为文本
    """
    try:
        stt_model = get_stt_model()
        
        # 读取上传的文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

            # 使用STT模型进行转录
            transcript = stt_model.transcribe(temp_file_path, language=language, cache=None)
        
        return TranscriptionResponse(text=transcript)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"音频转录失败: {str(e)}"
        )













