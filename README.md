# 语音服务 (Audio Service)

一个基于FastAPI的语音服务系统，集成了语音合成(TTS)和语音识别(STT)功能。本项目使用CosyVoice2作为TTS引擎，FunASR作为STT引擎，提供高质量的语音服务API。

## 功能特点

### 语音合成 (TTS)
- 支持零样本语音合成，可使用任意声音样本进行克隆
- 支持指令模式，可通过指令控制语音风格（如方言、情感等）
- 支持流式输出，实时返回生成的语音片段
- 支持自定义声音样本管理，可上传、删除声音样本
- 支持WAV、MP3等多种音频格式输出
- 可调节语音速度和随机种子

### 语音识别 (STT)
- 支持多种语言的语音转文字
- 支持标点与逆文本正则化
- 支持长音频自动切分处理

### 系统架构
- 基于FastAPI构建RESTful API
- 线程安全的模型封装，支持多线程并发处理
- Web UI界面，方便测试和使用
- 支持流式响应和文件下载

## 安装步骤

### 环境要求
- Python 3.8+
- CUDA支持 (GPU加速，推荐但非必须)

### 依赖安装

```bash
# 克隆项目
git clone https://github.com/yourusername/audio_service.git
cd audio_service

# 安装依赖
pip install -r requirements.txt

# 安装FunASR和CosyVoice2
# 请参考这些项目的官方安装指南
```

### 初始化

首次运行前，需确保`asset`目录下有`speaker.json`文件，用于管理声音样本：

```json
{
  "default": {
    "name": "默认声音",
    "prompt_text": "这是一个示例音频",
    "path": "default.mp3",
    "gender": "female"
  }
}
```

## 使用方法

### 启动服务

```bash
python run.py
```

服务默认运行在 http://127.0.0.1:8080

### Web界面

访问 http://127.0.0.1:8080/v1/audio/ui 可以使用Web界面测试语音合成和识别功能。

### API文档

访问 http://127.0.0.1:8080/docs 可以查看完整的API文档。

## API接口

### 语音合成

#### 获取可用声音列表
```
GET /v1/audio/voices
```

#### 获取声音样本详情
```
GET /v1/audio/voice_details
```

#### 获取声音样本音频
```
GET /v1/audio/sample_audio/{speaker_id}
```

#### 上传新声音样本
```
POST /v1/audio/upload_voice
```
参数:
- name: 声音名称
- gender: 性别
- prompt_text: 示例文本
- audio_file: 音频文件

#### 删除声音样本
```
DELETE /v1/audio/delete_voice/{speaker_id}
```

#### 文本转语音
```
POST /v1/audio/speech
```
请求体:
```json
{
  "input": "要转换的文本",
  "voice": "声音ID",
  "instruct": "指令或说话风格", 
  "speed": 1.0,
  "response_format": "mp3",
  "random_seed": 123
}
```

#### 流式文本转语音
```
POST /v1/audio/stream_speech
```
请求参数同上，但以流的形式返回音频数据。

### 语音识别

#### 音频转文字
```
POST /v1/audio/transcribe
```
表单参数:
- file: 音频文件
- model: 模型ID (默认 "iic/SenseVoiceSmall")
- language: 语言 (默认 "auto")
- prompt: 提示文本 (可选)
- response_format: 响应格式 (默认 "text")

## 项目结构

```
audio_service/
├── Service/                # 服务模块
│   ├── app.py              # FastAPI应用入口
│   ├── routers/            # 路由定义
│   │   ├── __init__.py     # 路由入口
│   │   └── audio.py        # 音频相关路由
│   ├── schemas/            # 数据模型定义
│   ├── templates/          # Web界面模板
│   └── utils/              # 实用工具
├── models/                 # 模型封装
│   ├── base.py             # 模型基类
│   ├── thread_safe_base.py # 线程安全模型基类  
│   ├── cosyvoice_tts.py    # 语音合成模型
│   └── funasr_stt.py       # 语音识别模型
├── third_party/            # 第三方模型集成
├── utils/                  # 通用工具
├── asset/                  # 资源文件
│   └── speaker.json        # 声音样本配置
├── requirements.txt        # 依赖清单
├── run.py                  # 运行入口
└── .gitignore              # Git忽略配置
```

## 扩展开发

### 添加新的TTS模型
1. 继承 `ThreadSafeModelBase` 类
2. 实现 `_load_model`、`_infer` 和 `_unload_model` 方法
3. 在 `Service/app.py` 中注册新模型

### 添加新的STT模型
同上