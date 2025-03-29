import os
import sys
from loguru import logger


def create_logger(name, level="INFO", console=True, file=None, rotation="10 MB", retention="1 week", 
                 format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"):
    """
    创建一个配置好的loguru logger对象
    
    参数:
        name (str): 日志记录器的名称
        level (str): 日志级别 ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
        console (bool): 是否输出到控制台
        file (str, optional): 日志文件路径，如果为None则不写入文件
        rotation (str): 日志文件轮转条件，可以是大小("10 MB")或时间("1 day")
        retention (str): 日志文件保留时间
        format (str): 日志格式
    
    返回:
        logger: 配置好的logger对象
    """
    # 清除默认的处理器
    logger.remove()
    
    # 配置格式
    log_format = format
    
    # 添加控制台处理器
    if console:
        logger.add(sys.stderr, format=log_format, level=level)
    
    # 添加文件处理器
    if file:
        # 确保日志目录存在
        log_dir = os.path.dirname(file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 添加文件处理器，支持轮转、保留和压缩
        logger.add(
            file,
            format=log_format,
            level=level,
            rotation=rotation,  # 每当文件达到这个大小时轮转
            retention=retention,  # 保留时间
            enqueue=True,        # 异步写入
            backtrace=True,      # 异常时显示完整回溯
            diagnose=True,       # 显示变量值
            compression="zip"  
        )
    
    # 创建一个带名字的logger
    named_logger = logger.bind(name=name)
    
    return named_logger

default_logger = create_logger("default_logger")

# 示例用法
if __name__ == "__main__":
    # 创建一个基本的logger
    log = create_logger("test_logger")
    log.info("这是一条信息日志")
    
    # 创建一个写入文件的logger
    file_logger = create_logger(
        name="file_logger",
        level="DEBUG",
        file="logs/app.log",
        rotation="1 day"
    )
    file_logger.debug("这是一条调试日志")
    file_logger.info("这是一条信息日志")
    file_logger.warning("这是一条警告日志")
    file_logger.error("这是一条错误日志")
    
    # 测试自动文件轮转（会创建 logs/app.log-1, logs/app.log-2 等）
    for i in range(10):
        file_logger.info(f"这是第{i}条测试日志，用于测试轮转功能") 