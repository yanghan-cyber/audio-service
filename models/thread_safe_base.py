import threading
from typing import Dict, Any

from models.base import ModelBase


class ThreadSafeModelBase(ModelBase):
    """
    线程安全的模型基类，在ModelBase基础上增加了线程同步机制
    
    属性:
        _lock: 可重入锁，保护模型的所有关键操作
        thread_safe: 是否启用线程安全机制
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, device: str = "cpu", thread_safe: bool = True):
        """
        初始化线程安全的模型基类
        
        参数:
            model_params: 模型参数字典，默认为None
            device: 模型运行设备，默认为"cpu"
            thread_safe: 是否启用线程安全机制，默认为True
        """
        # 调用父类初始化
        super().__init__(model_params, device)
        
        # 设置线程安全相关属性
        self._lock = threading.RLock()
        self.thread_safe = thread_safe
    
    def _with_lock(self, func, *args, **kwargs):
        """
        使用锁保护执行函数
        
        参数:
            func: 要执行的函数
            *args, **kwargs: 传递给函数的参数
            
        返回:
            函数的返回值
        """
        if self.thread_safe:
            with self._lock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def load(self, model_params: Dict[str, Any] = None) -> None:
        """
        线程安全的加载模型方法
        
        参数:
            model_params: 新的模型参数，如果为None则使用当前参数
        """
        return self._with_lock(super().load, model_params)
    
    def reload(self, model_params: Dict[str, Any] = None) -> bool:
        """
        线程安全的重新加载模型方法
        
        参数:
            model_params: 新的模型参数
            
        返回:
            bool: 是否进行了重新加载
        """
        return self._with_lock(super().reload, model_params)
    
    def infer(self, infer_params: Dict[str, Any], model_params: Dict[str, Any] = None) -> Any:
        """
        线程安全的模型推理方法
        
        参数:
            infer_params: 推理参数
            model_params: 模型参数，如果与当前参数不同则会重载模型
            
        返回:
            推理结果
        """
        return self._with_lock(super().infer, infer_params, model_params)
    
    def unload(self) -> None:
        """
        线程安全的卸载模型方法
        """
        return self._with_lock(super().unload)
    
    def set_thread_safe(self, enabled: bool) -> None:
        """
        设置是否启用线程安全机制
        
        参数:
            enabled: 是否启用
        """
        self.thread_safe = enabled 