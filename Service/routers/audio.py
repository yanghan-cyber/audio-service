from random import randint
import tempfile
from models.funasr_stt import FunASRSTT
from models.tts_manager import get_tts, get_tts_manager
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import Response, StreamingResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional, Iterator, Dict, Any
import torch
import json
import os
from Service.schemas.audio import SpeechRequest, TranscriptionResponse
from utils.audio import convert_audio_format
from Service.logger import get_logger
from third_party import set_all_random_seed

# 获取logger
logger = get_logger()
api_logger = get_logger("api")
error_logger = get_logger("error")

logger.info("初始化语音处理模型")
STT_MODEL = FunASRSTT()
# 初始化TTS管理器，设置较长的超时时间
get_tts_manager(timeout_seconds=1800)  # 30分钟超时

STT_MODEL.load()
# 预热加载常用模型
default_tts = get_tts("cosyvoice")
default_tts.load()

# 设置模板目录
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# 设置asset目录
ASSET_DIR = Path(__file__).parent.parent.parent / "asset"
ASSET_DIR.mkdir(exist_ok=True)

# 创建templates实例
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

def get_stt_model():
    global STT_MODEL
    return STT_MODEL

def get_tts_model(model_type: str = "cosyvoice", model_params: Dict[str, Any] = None):
    """
    根据指定类型获取TTS模型
    
    参数:
        model_type: TTS模型类型，可选值: "cosyvoice", "kokora"
        model_params: 可选的模型参数
        
    返回:
        TTS模型实例
    """
    if model_type not in ["cosyvoice", "kokora"]:
        logger.warning(f"不支持的TTS模型类型: {model_type}，使用默认的cosyvoice")
        model_type = "cosyvoice"
    
    logger.debug(f"获取TTS模型: {model_type}")
    return get_tts(model_type, model_params)

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
    logger.info("访问语音合成UI页面")
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@router.get("/voices")
async def get_voices(model_type: str = "cosyvoice"):
    """
    获取可用声音列表
    
    参数:
        model_type: TTS模型类型，可选值: "cosyvoice", "kokora"
    """
    logger.info(f"获取可用声音列表，模型类型: {model_type}")
    tts_model = get_tts_model(model_type)
    voices = tts_model.speaker_list
    logger.debug(f"返回 {len(voices) if isinstance(voices, list) else '未知数量的'} 个声音")
    return voices

@router.get("/voice_details")
async def get_voice_details(model_type: str = "cosyvoice"):
    """
    获取所有声音样本的详细信息，包括prompt_text
    
    参数:
        model_type: TTS模型类型，可选值: "cosyvoice", "kokora"
    """
    logger.info(f"获取声音样本详细信息，模型类型: {model_type}")
    
    # 如果是kokora模型，返回内置的声音列表
    if model_type == "kokora":
        tts_model = get_tts_model(model_type)
        speakers = tts_model.speaker_list
        voice_details = []
        for speaker in speakers:
            # 为kokora模型构建简化的声音详情
            voice_details.append({
                "id": speaker,
                "name": speaker,  # 使用ID作为名称
                "gender": "unknown",  # kokora模型没有提供性别信息
                "prompt_text": "",  # kokora模型不需要prompt_text
                "path": ""  # kokora模型不需要音频文件路径
            })
        logger.debug(f"返回kokora模型的 {len(voice_details)} 个声音样本详情")
        return voice_details
    
    # 对于CosyVoice模型，使用之前的逻辑
    try:
        # 读取speaker.json
        speaker_file = ASSET_DIR / "speaker.json"
        with open(speaker_file, "r", encoding="utf-8") as f:
            speakers = json.load(f)
        
        # 构建详细信息数组
        voice_details = []
        for speaker_id, info in speakers.items():
            voice_details.append({
                "id": speaker_id,
                "name": info["name"],
                "gender": info["gender"], 
                "prompt_text": info["prompt_text"],
                "path": info["path"]
            })
        
        logger.debug(f"返回cosyvoice模型的 {len(voice_details)} 个声音样本详情")
        return voice_details
    
    except Exception as e:
        error_message = f"获取声音样本详细信息失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@router.get("/sample_audio/{speaker_id}")
async def get_sample_audio(speaker_id: str):
    """
    获取指定声音样本的音频文件
    """
    logger.info(f"获取声音样本音频: {speaker_id}")
    try:
        # 读取speaker.json
        speaker_file = ASSET_DIR / "speaker.json"
        with open(speaker_file, "r", encoding="utf-8") as f:
            speakers = json.load(f)
        
        # 检查speaker_id是否存在
        if speaker_id not in speakers:
            error_message = f"找不到ID为 '{speaker_id}' 的声音样本"
            logger.warning(error_message)
            raise HTTPException(
                status_code=404,
                detail=error_message
            )
        
        # 获取音频文件路径
        audio_path = ASSET_DIR / speakers[speaker_id]["path"]
        if not os.path.exists(audio_path):
            error_message = f"找不到声音样本的音频文件: {audio_path}"
            error_logger.error(error_message)
            raise HTTPException(
                status_code=404,
                detail="找不到声音样本的音频文件"
            )
        
        logger.debug(f"返回声音样本音频文件: {audio_path}")
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=f"{speaker_id}.mp3"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_message = f"获取声音样本音频失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@router.post("/upload_voice")
async def upload_voice(
    name: str = Form(...),
    gender: str = Form(...),
    prompt_text: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    上传声音样本并添加到speaker列表 (仅适用于CosyVoice模型)
    """
    logger.info(f"上传声音样本: {name}")
    try:
        # 获取CosyVoice模型
        tts_model = get_tts_model("cosyvoice")
        
        # 读取现有的speaker.json
        speaker_file = ASSET_DIR / "speaker.json"
        with open(speaker_file, "r", encoding="utf-8") as f:
            speakers = json.load(f)
        
        # 生成唯一的speaker_id
        speaker_id = name.lower().replace(" ", "_")
        logger.debug(f"生成的speaker_id: {speaker_id}")
        
        # 检查是否已存在同名speaker
        if speaker_id in speakers:
            error_message = f"已存在名为 '{name}' 的声音样本"
            logger.warning(error_message)
            raise HTTPException(
                status_code=400,
                detail=error_message
            )
        
        # 保存音频文件
        audio_path = f"{speaker_id}.mp3"
        audio_full_path = ASSET_DIR / audio_path
        logger.debug(f"保存音频文件到: {audio_full_path}")

        with open(audio_full_path, "wb") as f:
            f.write(await audio_file.read())
            
        # 使用sf保存
        import librosa
        import soundfile as sf
        wav,sr = librosa.load(audio_full_path)
        sf.write(audio_full_path,wav,sr)
        
        
        # 添加到speaker.json
        speakers[speaker_id] = {
            "name": name,
            "prompt_text": prompt_text,
            "path": audio_path,
            "gender": gender
        }
        
        # 保存更新后的speaker.json
        with open(speaker_file, "w", encoding="utf-8") as f:
            json.dump(speakers, f, ensure_ascii=False, indent=4)
        
        # 重新加载speaker配置
        tts_model.reload_speaker_config()
        logger.info(f"声音样本 '{name}' 已成功上传")
        
        return {"status": "success", "message": f"声音样本 '{name}' 已成功上传"}
    
    except Exception as e:
        error_message = f"上传声音样本失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@router.delete("/delete_voice/{speaker_id}")
async def delete_voice(speaker_id: str):
    """
    删除声音样本 (仅适用于CosyVoice模型)
    """
    logger.info(f"删除声音样本: {speaker_id}")
    try:
        # 获取CosyVoice模型
        tts_model = get_tts_model("cosyvoice")
        
        # 读取现有的speaker.json
        speaker_file = ASSET_DIR / "speaker.json"
        with open(speaker_file, "r", encoding="utf-8") as f:
            speakers = json.load(f)
        
        # 检查speaker_id是否存在
        if speaker_id not in speakers:
            error_message = f"找不到ID为 '{speaker_id}' 的声音样本"
            logger.warning(error_message)
            raise HTTPException(
                status_code=404,
                detail=error_message
            )
        
        # 删除音频文件
        audio_path = ASSET_DIR / speakers[speaker_id]["path"]
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.debug(f"已删除音频文件: {audio_path}")
        
        # 从speaker.json中删除
        del speakers[speaker_id]
        
        # 保存更新后的speaker.json
        with open(speaker_file, "w", encoding="utf-8") as f:
            json.dump(speakers, f, ensure_ascii=False, indent=4)
        
        # 重新加载speaker配置
        tts_model.reload_speaker_config()
        logger.info(f"声音样本 '{speaker_id}' 已成功删除")
        
        return {"status": "success", "message": f"声音样本 '{speaker_id}' 已成功删除"}
    
    except Exception as e:
        error_message = f"删除声音样本失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@router.post("/speech", response_class=Response)
async def speech(request: SpeechRequest):
    logger.info(f"文本转语音请求: 文本长度={len(request.input)}, 声音={request.voice}, 模型类型={request.model}")
    try:
        # 根据请求中的model_type参数获取对应的TTS模型
        tts_model = get_tts_model(request.model)
        
        # 设置随机种子（如果提供）
        random_seed = request.random_seed
        if random_seed is not None:
            set_all_random_seed(random_seed)
        else:
            set_all_random_seed(randint(0, 2147483647))
        infer_params = {
            "text": request.input,
            "voice": request.voice,
            "instruct": request.instruct,
            "speed": request.speed,
        }
        
        # 判断模型类型和请求参数，适配不同模型的参数需求
        if request.model == "cosyvoice":
            if request.instruct:
                infer_params["mode"] = "instruct"
                logger.debug(f"使用指令模式: {request.instruct}")
            else:
                infer_params["mode"] = "zero_shot"
                logger.debug("使用零样本模式")
        elif request.model == "kokora":
            # kokora模型参数调整
            # 移除kokora不支持的参数
            if "instruct" in infer_params:
                del infer_params["instruct"]
            if "mode" in infer_params:
                del infer_params["mode"]
            # 可以添加kokora特有的参数
            infer_params["language"] = request.language if hasattr(request, "language") else "zh"
            infer_params["silence_duration"] = request.silence_duration if hasattr(request, "silence_duration") else 0.1

        logger.debug(f"开始使用{request.model}模型生成语音")
        audio_generator = tts_model.infer(infer_params)
        all_audio_chunks = [chunk for chunk in audio_generator]
        
        if all_audio_chunks:
            all_audio = torch.cat(all_audio_chunks, dim=1) if request.model == "cosyvoice" else all_audio_chunks[0] if len(all_audio_chunks) == 1 else torch.from_numpy(np.concatenate(all_audio_chunks))
            sample_rate = tts_model.sample_rate

            logger.debug(f"转换音频格式为: {request.response_format}")
            audio_binary = convert_audio_format(all_audio, sample_rate, request.response_format)
            
            # 根据格式设置正确的 Content-Type
            content_type = {
                "mp3": "audio/mpeg",
                "wav": "audio/wav",
                "aac": "audio/aac",
                "m4a": "audio/mp4",
                "ogg": "audio/ogg",
                "flac": "audio/flac",
                "webm": "audio/webm"
            }.get(request.response_format, "audio/mpeg")

            logger.info("语音合成完成")
            return Response(
                content=audio_binary,
                media_type=content_type
            )
        else:
            raise ValueError("生成的音频为空")

    except Exception as e:
        error_message = f"语音生成失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )

@router.post("/stream_speech")
async def stream_speech(request: SpeechRequest):
    """
    流式文本转语音接口，实时返回生成的语音片段
    """
    logger.info(f"流式文本转语音请求: 文本长度={len(request.input)}, 声音={request.voice}, 模型类型={request.model}")
    try:
        # 根据请求中的model_type参数获取对应的TTS模型
        tts_model = get_tts_model(request.model)
        
        # 设置随机种子（如果提供）
        random_seed = request.random_seed
        
        infer_params = {
            "text": request.input,
            "voice": request.voice,
            "instruct": request.instruct,
            "speed": request.speed,
            "stream": True,  # 启用流式输出
            "random_seed": random_seed,
        }
        
        # 判断模型类型和请求参数，适配不同模型的参数需求
        if request.model == "cosyvoice":
            if request.instruct:
                infer_params["mode"] = "instruct"
                logger.debug(f"使用指令模式: {request.instruct}")
            else:
                infer_params["mode"] = "zero_shot"
                logger.debug("使用零样本模式")
        elif request.model == "kokora":
            # kokora模型参数调整
            # 移除kokora不支持的参数
            if "instruct" in infer_params:
                del infer_params["instruct"]
            if "mode" in infer_params:
                del infer_params["mode"]
            # 可以添加kokora特有的参数
            infer_params["language"] = request.language if hasattr(request, "language") else "zh"
            infer_params["silence_duration"] = request.silence_duration if hasattr(request, "silence_duration") else 0.1

        logger.debug(f"开始使用{request.model}模型流式生成语音")
        
        # 创建音频流生成器函数
        def generate_audio_chunks() -> Iterator[bytes]:
            sample_rate = tts_model.sample_rate
            audio_generator = tts_model.infer(infer_params)
            
            chunk_count = 0
            import numpy as np  # 确保numpy已导入
            
            for chunk in audio_generator:
                # 处理不同模型可能返回不同类型的音频数据
                if request.model == "kokora":
                    # kokora返回的是numpy数组
                    if not isinstance(chunk, torch.Tensor):
                        chunk = torch.from_numpy(chunk)
                
                # 将每个音频片段转换为指定格式的二进制数据
                audio_binary = convert_audio_format(chunk, sample_rate, request.response_format)
                chunk_count += 1
                logger.debug(f"生成音频片段 #{chunk_count}: {len(audio_binary)} 字节")
                yield audio_binary
            
            logger.info(f"流式语音合成完成，共 {chunk_count} 个片段")
        
        # 根据格式设置正确的 Content-Type
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "aac": "audio/aac",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "webm": "audio/webm"
        }.get(request.response_format, "audio/mpeg")
        
        return StreamingResponse(
            generate_audio_chunks(),
            media_type=content_type
        )

    except Exception as e:
        error_message = f"流式语音生成失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
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
    logger.info(f"音频转文字请求: 文件名={file.filename}, 语言={language}, 模型={model}")
    try:
        stt_model = get_stt_model()
        
        # 读取上传的文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await file.read()
            file_size = len(content)
            logger.debug(f"上传的音频文件大小: {file_size} 字节")
            
            temp_file.write(content)
            temp_file_path = temp_file.name
            logger.debug(f"保存到临时文件: {temp_file_path}")

            # 使用STT模型进行转录
            logger.debug("开始转录音频")
            transcript = stt_model.transcribe(temp_file_path, language=language, cache=None)
            logger.debug(f"转录结果字数: {len(transcript)}")
        
        logger.info("音频转录完成")
        return TranscriptionResponse(text=transcript)
    
    except Exception as e:
        error_message = f"音频转录失败: {str(e)}"
        error_logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message
        )













