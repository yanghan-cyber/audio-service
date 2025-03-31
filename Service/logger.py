import os
from pathlib import Path
from utils.logger import create_logger

# 获取日志目录路径
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 创建应用级logger
app_logger = create_logger(
    name="audio_service",
    level="INFO",
    file=str(LOG_DIR / "app.log"),
    rotation="100 MB",
    retention="30 days"
)

# 创建API访问日志logger
api_logger = create_logger(
    name="api_access",
    level="INFO",
    file=str(LOG_DIR / "api_access.log"),
    rotation="100 MB",
    retention="30 days",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[client_ip]}</cyan> | <level>{message}</level>"
)

# 创建错误日志logger
error_logger = create_logger(
    name="error_log",
    level="ERROR",
    file=str(LOG_DIR / "error.log"),
    rotation="50 MB",
    retention="60 days"
)

# 为了便于导入，定义一个get_logger函数
def get_logger(name=None):
    """
    获取应用logger
    
    参数:
        name: 可选的logger名称，默认为app_logger
        
    返回:
        配置好的logger对象
    """
    if name == "api":
        return api_logger
    elif name == "error":
        return error_logger
    else:
        return app_logger 