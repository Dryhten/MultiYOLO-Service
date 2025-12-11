"""YOLO检测器核心接口定义"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio


@dataclass
class DetectionResult:
    """单目标检测结果"""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


class BaseDetector(ABC):
    """检测器抽象基类"""
    
    @abstractmethod
    def detect(self, image, conf_threshold: Optional[float] = None) -> List[DetectionResult]:
        """检测图像中的目标对象（同步方法）"""
        pass
    
    async def detect_async(self, image, conf_threshold: Optional[float] = None) -> List[DetectionResult]:
        """异步检测图像中的目标对象"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.detect, image, conf_threshold)
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """获取硬件设备信息"""
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        pass

