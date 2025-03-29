from fastapi import FastAPI
from .routers import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TTS API",
)
# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)
app.include_router(router, prefix="/v1")

@app.get("/")
async def root():
    return {"message": "语音服务API正在运行"} 