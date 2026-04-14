"""
模型工厂类

使用工厂模式创建不同类型的油藏数值模拟模型
"""

from typing import Dict, Any, Type
import logging
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    模型工厂类
    
    负责注册和创建不同类型的数学模型实例
    """
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        """
        注册模型类
        
        Args:
            model_type: 模型类型名称
            model_class: 模型类
        """
        cls._registry[model_type] = model_class
        logger.debug(f"Registered model: {model_type} -> {model_class.__name__}")
        
    @classmethod
    def create_model(cls, model_type: str, mesh, physics, config: Dict[str, Any]) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型名称
            mesh: 网格对象
            physics: 物理属性对象
            config: 模型配置
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型类型未注册
        """
        if model_type not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {available_types}")
            
        model_class = cls._registry[model_type]
        return model_class(mesh, physics, config)
        
    @classmethod
    def get_registered_models(cls) -> Dict[str, Type[BaseModel]]:
        """
        获取已注册的模型
        
        Returns:
            注册的模型字典
        """
        return cls._registry.copy()
        
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """
        检查模型类型是否已注册
        
        Args:
            model_type: 模型类型名称
            
        Returns:
            是否已注册
        """
        return model_type in cls._registry


def register_builtin_models():
    """注册内置模型"""
    try:
        from .single_phase.single_phase_model import SinglePhaseModel
        ModelFactory.register('single_phase', SinglePhaseModel)
    except ImportError:
        logger.warning("SinglePhaseModel not available")

    try:
        from .two_phase_impes import TwoPhaseIMPES
        ModelFactory.register('two_phase_impes', TwoPhaseIMPES)
    except ImportError:
        logger.warning("TwoPhaseIMPES not available")

    try:
        from .two_phase_fim import TwoPhaseFIM
        ModelFactory.register('two_phase_fim', TwoPhaseFIM)
    except ImportError:
        logger.warning("TwoPhaseFIM not available")


# 自动注册内置模型
register_builtin_models()