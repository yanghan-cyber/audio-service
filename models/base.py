from typing import Dict, Any
import uuid
import json
from abc import ABC, abstractmethod

from utils.logger import create_logger

class ModelBase(ABC):
    """
    模型基类，提供模型重载、推理和卸载的基本功能。
    
    属性:
        model_params: 模型参数字典
        model: 加载的模型实例
        device: 模型运行设备
        is_loaded: 模型是否已加载
        model_id: 模型唯一标识
    """
    
    def __init__(self, model_params: Dict[str, Any] = None, device: str = "cpu"):
        """
        初始化模型基类
        
        参数:
            model_params: 模型参数字典，默认为None
            device: 模型运行设备，默认为"cpu"
        """
        self.model_params = model_params or {}
        self.model = None
        self.device = device
        self.is_loaded = False
        
        # 仅根据模型参数生成UUID
        params_str = json.dumps(self.model_params, sort_keys=True)
        params_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, params_str).hex[:8]
        
        # 模型ID格式为：类名_UUID
        self.model_id = f"{self.__class__.__name__}_{params_uuid}"
        
        # 创建包含model_id的logger
        self.logger = create_logger(name=f"{self.__class__.__name__}[{self.model_id}]")
    
    def load(self, model_params: Dict[str, Any] = None) -> None:
        """
        加载模型
        
        参数:
            model_params: 新的模型参数，如果为None则使用当前参数
        """
        # 如果提供了新参数，则更新模型参数
        if model_params is not None:
            self.model_params.update(model_params)
        
        # 实际加载模型的逻辑，由子类实现
        try:
            self._load_model()
            self.is_loaded = True
            self.logger.info(f"模型已成功加载，参数: {self.model_params}")
        except Exception as e:
            self.is_loaded = False
            self.logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def reload(self, model_params: Dict[str, Any] = None) -> bool:
        """
        重新加载模型，如果参数发生变化
        
        参数:
            model_params: 新的模型参数
            
        返回:
            bool: 是否进行了重新加载
        """
        # 如果模型未加载，直接加载
        if not self.is_loaded:
            self.load(model_params)
            return True
        
        # 如果没有新参数，不需要重载
        if model_params is None:
            return False
        
        # 检查参数是否有变化
        params_changed = False
        for key, value in model_params.items():
            if key not in self.model_params or self.model_params[key] != value:
                params_changed = True
                break
        
        # 如果参数有变化，卸载后重新加载
        if params_changed:
            self.logger.info("模型参数已变化，重新加载模型")
            self.unload()
            self.load(model_params)
            return True
        
        return False
    
    def infer(self, infer_params: Dict[str, Any], model_params: Dict[str, Any] = None) -> Any:
        """
        模型推理方法
        
        参数:
            infer_params: 推理参数
            model_params: 模型参数，如果与当前参数不同则会重载模型
            
        返回:
            推理结果
        """
        # 检查是否需要重载模型
        self.reload(model_params)
        
        # 确保模型已加载
        if not self.is_loaded:
            self.load()
        
        # 调用子类实现的推理方法
        try:
            result = self._infer(infer_params)
            return result
        except Exception as e:
            self.logger.error(f"推理失败: {str(e)}")
            raise
    
    def unload(self) -> None:
        """
        卸载模型，释放资源
        """
        if not self.is_loaded:
            return
        
        try:
            self._unload_model()
            self.model = None
            self.is_loaded = False
            self.logger.info("模型已卸载")
        except Exception as e:
            self.logger.error(f"模型卸载失败: {str(e)}")
            raise
    
    @abstractmethod
    def _load_model(self) -> None:
        """
        实际加载模型的方法，需要子类实现
        """
        pass
    
    @abstractmethod
    def _infer(self, infer_params: Dict[str, Any]) -> Any:
        """
        实际推理的方法，需要子类实现
        
        参数:
            infer_params: 推理参数
            
        返回:
            推理结果
        """
        pass
    
    @abstractmethod
    def _unload_model(self) -> None:
        """
        实际卸载模型的方法，需要子类实现
        """
        pass 