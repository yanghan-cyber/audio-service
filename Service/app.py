from fastapi import FastAPI, Request
from .routers import router
from fastapi.middleware.cors import CORSMiddleware
import time
from .logger import get_logger, app_logger

# 初始化应用日志
app_logger.info("初始化语音服务API")

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

# 添加日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    api_logger = get_logger("api")
    
    # 获取客户端IP
    client_ip = request.client.host if request.client else "未知IP"
    
    # 记录请求开始
    start_time = time.time()
    
    # 绑定客户端IP到logger
    api_logger = api_logger.bind(client_ip=client_ip)
    
    # 记录请求信息
    api_logger.info(f"开始请求 {request.method} {request.url.path}")
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 记录响应信息
        api_logger.info(f"完成请求 {request.method} {request.url.path} - 状态码: {response.status_code} - 耗时: {process_time:.4f}秒")
        
        return response
    except Exception as e:
        # 记录异常信息
        error_logger = get_logger("error")
        error_logger.error(f"处理请求 {request.method} {request.url.path} 时发生错误: {str(e)}")
        raise

app.include_router(router, prefix="/v1")

@app.get("/")
async def root():
    app_logger.info("访问根路径")
    return {"message": "语音服务API正在运行"} 