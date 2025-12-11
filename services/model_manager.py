"""模型管理器"""

import os
import threading
from typing import Dict, Optional, List
from pathlib import Path

from core.config import settings
from core.logging_config import get_logger
from services.detectors import TorchDetector
from core.interfaces import BaseDetector

logger = get_logger(__name__)


class ModelManager:
    """管理多个YOLO模型实例"""
    
    def __init__(self):
        self._models: Dict[str, BaseDetector] = {}
        self._lock = threading.Lock()
        self._load_all_models()
    
    def _load_all_models(self):
        model_paths = self._get_model_paths()
        
        if not model_paths:
            logger.warning("未找到任何模型文件，使用默认配置")
            if settings.MODEL_PATH and os.path.exists(settings.MODEL_PATH):
                model_name = self._get_model_name(settings.MODEL_PATH)
                try:
                    detector = TorchDetector(model_path=settings.MODEL_PATH, suppress_device_log=True)
                    self._models[model_name] = detector
                    class_names = sorted(detector.class_names.values())
                    classes_str = ", ".join(class_names)
                    logger.info(f"✓ 模型加载成功: {model_name} | 类别: {classes_str}")
                except Exception as e:
                    logger.error(f"✗ 模型加载失败 {settings.MODEL_PATH}: {str(e)}")
            return
        
        logger.info(f"开始加载 {len(model_paths)} 个模型...")
        device_log_shown = False
        
        for idx, model_path in enumerate(model_paths, 1):
            model_name = self._get_model_name(model_path)
            
            if model_name in self._models:
                logger.warning(f"[{idx}/{len(model_paths)}] 模型名称冲突: {model_name}，跳过")
                continue
            
            try:
                logger.info(f"[{idx}/{len(model_paths)}] 加载模型: {model_name}")
                detector = TorchDetector(model_path=model_path, suppress_device_log=device_log_shown)
                self._models[model_name] = detector
                class_names = sorted(detector.class_names.values())
                classes_str = ", ".join(class_names)
                logger.info(f"    ✓ 成功 | 类别: {classes_str}")
                device_log_shown = True
            except Exception as e:
                logger.error(f"[{idx}/{len(model_paths)}] ✗ 加载失败: {model_name} - {str(e)}")
        
        logger.info(f"模型加载完成，共 {len(self._models)} 个模型: {', '.join(self._models.keys())}")
    
    def _get_model_paths(self) -> List[str]:
        model_paths = []
        base_dir = Path(__file__).parent.parent
        
        if settings.MODEL_DIR:
            model_dir = Path(settings.MODEL_DIR)
            if not model_dir.is_absolute():
                model_dir = base_dir / model_dir
            
            if model_dir.exists() and model_dir.is_dir():
                pt_files = list(model_dir.glob("*.pt"))
                model_paths.extend([str(f) for f in pt_files])
                logger.info(f"从目录 {model_dir} 找到 {len(pt_files)} 个模型文件")
            else:
                logger.warning(f"模型目录不存在: {model_dir}")
        
        if settings.MODEL_PATHS:
            paths_str = settings.MODEL_PATHS.strip()
            for path_str in paths_str.split(","):
                path_str = path_str.strip()
                if not path_str:
                    continue
                
                model_path = Path(path_str)
                if not model_path.is_absolute():
                    model_path = base_dir / model_path
                
                if model_path.exists():
                    model_paths.append(str(model_path))
                else:
                    logger.warning(f"模型文件不存在: {model_path}")
        
        if not model_paths and settings.MODEL_PATH:
            if os.path.exists(settings.MODEL_PATH):
                model_paths.append(settings.MODEL_PATH)
        
        return list(set(model_paths))
    
    def _get_model_name(self, model_path: str) -> str:
        path_obj = Path(model_path)
        return path_obj.stem
    
    def get_model(self, model_name: str) -> Optional[BaseDetector]:
        with self._lock:
            return self._models.get(model_name)
    
    def list_models(self) -> List[str]:
        with self._lock:
            return list(self._models.keys())
    
    def has_model(self, model_name: str) -> bool:
        with self._lock:
            return model_name in self._models
    
    def get_default_model(self) -> Optional[BaseDetector]:
        with self._lock:
            if self._models:
                return next(iter(self._models.values()))
            return None


_model_manager: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """获取模型管理器单例"""
    global _model_manager
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager

