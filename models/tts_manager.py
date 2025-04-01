import threading
import time
from typing import Dict, Any, Optional, List, Type, Union
import uuid

from models.thread_safe_base import ThreadSafeModelBase
from models.kokora_tts import KokoraTTS
from models.cosyvoice_tts import CosyVoiceTTS
from utils.logger import create_logger


class TTSFactory:
    """
    TTS模型工厂，负责创建和管理TTS模型实例
    """
    
    # 支持的TTS模型类字典
    TTS_MODELS = {
        "kokora": KokoraTTS,
        "cosyvoice": CosyVoiceTTS
    }
    
    @classmethod
    def create(cls, tts_type: str, model_params: Dict[str, Any] = None, device: str = None) -> ThreadSafeModelBase:
        """
        创建TTS模型实例
        
        参数:
            tts_type: TTS模型类型，可选值: "kokora", "cosyvoice"
            model_params: 模型参数
            device: 运行设备
            
        返回:
            TTS模型实例
        """
        if tts_type not in cls.TTS_MODELS:
            raise ValueError(f"不支持的TTS类型: {tts_type}，可选值为: {list(cls.TTS_MODELS.keys())}")
        
        # 获取模型类并实例化
        model_class = cls.TTS_MODELS[tts_type]
        model = model_class(model_params=model_params, device=device)
        
        return model
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        获取所有可用的TTS模型类型
        
        返回:
            模型类型列表
        """
        return list(cls.TTS_MODELS.keys())


class TTSManager:
    """
    TTS管理器，负责管理多个TTS模型实例和自动释放资源
    
    特性:
    - 支持多个TTS模型实例同时存在
    - 支持超时自动释放未使用的模型
    - 线程安全操作
    """
    
    def __init__(self, timeout_seconds: int = 300):
        """
        初始化TTS管理器
        
        参数:
            timeout_seconds: 模型自动卸载的超时时间（秒）
        """
        self.timeout_seconds = timeout_seconds
        self.instances = {}  # 存储模型实例
        self.last_used = {}  # 记录每个实例最后使用时间
        self.lock = threading.RLock()  # 线程锁
        self.logger = create_logger(name="TTSManager")
        
        # 启动自动释放线程
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info(f"TTS管理器已初始化，超时时间: {timeout_seconds}秒")
    
    def get_instance(self, tts_type: str, model_params: Dict[str, Any] = None, device: str = None) -> ThreadSafeModelBase:
        """
        获取TTS模型实例，如果不存在则创建新实例
        
        参数:
            tts_type: TTS模型类型，可选值: "kokora", "cosyvoice"
            model_params: 模型参数
            device: 运行设备
            
        返回:
            TTS模型实例
        """
        with self.lock:
            # 生成实例标识符
            instance_id = self._generate_instance_id(tts_type, model_params)
            
            # 检查实例是否存在
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                # 更新最后使用时间
                self.last_used[instance_id] = time.time()
                self.logger.debug(f"使用已有实例: {instance_id}")
                return instance
            
            # 创建新实例
            instance = TTSFactory.create(tts_type, model_params, device)
            self.instances[instance_id] = instance
            self.last_used[instance_id] = time.time()
            
            self.logger.info(f"创建新实例: {tts_type}, ID: {instance_id}")
            return instance
    
    def release_instance(self, instance_id: str) -> bool:
        """
        手动释放指定的模型实例
        
        参数:
            instance_id: 实例ID
            
        返回:
            bool: 是否成功释放
        """
        with self.lock:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                # 卸载模型
                if instance.is_loaded:
                    instance.unload()
                
                # 删除记录
                del self.instances[instance_id]
                if instance_id in self.last_used:
                    del self.last_used[instance_id]
                
                self.logger.info(f"实例已手动释放: {instance_id}")
                return True
            
            return False
    
    def update_last_used(self, instance_id: str) -> None:
        """
        更新实例最后使用时间
        
        参数:
            instance_id: 实例ID
        """
        with self.lock:
            if instance_id in self.instances:
                self.last_used[instance_id] = time.time()
    
    def _cleanup_task(self) -> None:
        """
        超时清理任务，定期检查并释放超时未使用的模型
        """
        while self.running:
            time.sleep(30)  # 每30秒检查一次
            
            with self.lock:
                current_time = time.time()
                instances_to_remove = []
                
                # 找出超时的实例
                for instance_id, last_used_time in self.last_used.items():
                    if current_time - last_used_time > self.timeout_seconds:
                        instances_to_remove.append(instance_id)
                
                # 释放超时实例
                for instance_id in instances_to_remove:
                    try:
                        if instance_id in self.instances:
                            instance = self.instances[instance_id]
                            if instance.is_loaded:
                                instance.unload()
                            
                            del self.instances[instance_id]
                            del self.last_used[instance_id]
                            
                            self.logger.info(f"实例已超时释放: {instance_id}")
                    except Exception as e:
                        self.logger.error(f"释放实例 {instance_id} 时出错: {str(e)}")
    
    def _generate_instance_id(self, tts_type: str, model_params: Dict[str, Any] = None) -> str:
        """
        根据TTS类型和参数生成实例ID
        
        参数:
            tts_type: TTS模型类型
            model_params: 模型参数
            
        返回:
            实例ID
        """
        # 为保证唯一性，同类型但是参数不同的模型会使用不同的实例ID
        if model_params:
            params_str = str(sorted(model_params.items()))
            instance_id = f"{tts_type}_{uuid.uuid5(uuid.NAMESPACE_DNS, params_str).hex[:8]}"
        else:
            instance_id = f"{tts_type}_default"
        
        return instance_id
    
    def get_all_instances(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有实例的状态信息
        
        返回:
            实例状态字典
        """
        with self.lock:
            result = {}
            current_time = time.time()
            
            for instance_id, instance in self.instances.items():
                last_used_time = self.last_used.get(instance_id, 0)
                time_since_last_use = current_time - last_used_time
                
                # 获取实例基本信息
                instance_info = {
                    "tts_type": instance_id.split("_")[0],
                    "is_loaded": instance.is_loaded,
                    "model_id": instance.model_id,
                    "last_used": int(time_since_last_use),
                    "timeout_in": max(0, self.timeout_seconds - int(time_since_last_use))
                }
                
                result[instance_id] = instance_info
            
            return result
    
    def shutdown(self) -> None:
        """
        关闭管理器，释放所有资源
        """
        self.running = False
        
        with self.lock:
            # 卸载所有模型
            for instance_id, instance in self.instances.items():
                try:
                    if instance.is_loaded:
                        instance.unload()
                    self.logger.info(f"关闭时释放实例: {instance_id}")
                except Exception as e:
                    self.logger.error(f"关闭时释放实例 {instance_id} 出错: {str(e)}")
            
            # 清空记录
            self.instances.clear()
            self.last_used.clear()
        
        # 等待清理线程结束
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1)
        
        self.logger.info("TTS管理器已关闭")


# 全局TTS管理器实例
_global_tts_manager = None

def get_tts_manager(timeout_seconds: int = 300) -> TTSManager:
    """
    获取全局TTS管理器实例
    
    参数:
        timeout_seconds: 模型自动卸载的超时时间（秒）
        
    返回:
        TTS管理器实例
    """
    global _global_tts_manager
    
    if _global_tts_manager is None:
        _global_tts_manager = TTSManager(timeout_seconds=timeout_seconds)
    
    return _global_tts_manager


def get_tts(tts_type: str, model_params: Dict[str, Any] = None, device: str = None) -> ThreadSafeModelBase:
    """
    便捷函数：获取指定类型的TTS模型实例
    
    参数:
        tts_type: TTS模型类型，可选值: "kokora", "cosyvoice"
        model_params: 模型参数
        device: 运行设备
        
    返回:
        TTS模型实例
    """
    manager = get_tts_manager()
    return manager.get_instance(tts_type, model_params, device)


# 示例用法
if __name__ == "__main__":
    # 获取TTS管理器实例
    tts_manager = get_tts_manager(timeout_seconds=60)
    
    try:
        # 获取Kokora TTS实例
        kokora_tts = tts_manager.get_instance("kokora")
        print(f"获取Kokora TTS实例: {kokora_tts.model_id}")
        
        # 获取CosyVoice TTS实例
        cosyvoice_tts = tts_manager.get_instance("cosyvoice")
        print(f"获取CosyVoice TTS实例: {cosyvoice_tts.model_id}")
        
        # 查看所有实例
        instances_info = tts_manager.get_all_instances()
        print("当前所有实例:")
        for instance_id, info in instances_info.items():
            print(f"- {instance_id}: {info}")
        
        # 实验加载和使用
        kokora_tts.load()
        print(f"Kokora TTS加载状态: {kokora_tts.is_loaded}")
        
        # 等待超时自动释放（在实际应用中不需要这样）
        print("等待自动释放...")
        time.sleep(65)
        
        # 查看实例是否已释放
        instances_info = tts_manager.get_all_instances()
        print("超时后的实例:")
        for instance_id, info in instances_info.items():
            print(f"- {instance_id}: {info}")
    
    finally:
        # 关闭TTS管理器
        tts_manager.shutdown()
        print("TTS管理器已关闭")
