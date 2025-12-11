"""YOLO 服务配置管理模块"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    """YOLO 服务配置类"""
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")
    MODEL_PATH: str = None
    MODEL_DIR: Optional[str] = None
    MODEL_PATHS: Optional[str] = None
    CLASSES_FILE: Optional[str] = None
    CONF_THRESHOLD: float = 0.5
    IMGSZ: int = 640
    MAX_DET: int = 1000
    SAVE_TMP_DIR: Path = Path("./tmp")
    SAVE_TMP_ENABLED: bool = False

    def __init__(self, **kwargs):
        load_dotenv()
        if "HOST" not in kwargs:
            kwargs["HOST"] = os.getenv("YOLO_SERVICE_HOST", "0.0.0.0")
        if "PORT" not in kwargs:
            kwargs["PORT"] = int(os.getenv("YOLO_SERVICE_PORT", "8003"))
        if "RELOAD" not in kwargs:
            kwargs["RELOAD"] = os.getenv("YOLO_SERVICE_RELOAD", "false").lower() == "true"
        if "LOG_LEVEL" not in kwargs:
            kwargs["LOG_LEVEL"] = os.getenv("YOLO_SERVICE_LOG_LEVEL", "INFO")
        if "LOG_DIR" not in kwargs:
            log_dir = os.getenv("YOLO_SERVICE_LOG_DIR", "./logs")
            kwargs["LOG_DIR"] = Path(log_dir)
        if "MODEL_DIR" not in kwargs:
            model_dir = os.getenv("YOLO_SERVICE_MODEL_DIR")
            if model_dir:
                model_dir_obj = Path(model_dir)
                if not model_dir_obj.is_absolute():
                    base_dir = Path(__file__).parent.parent
                    model_dir = str(base_dir / model_dir_obj)
                else:
                    model_dir = str(model_dir_obj)
                kwargs["MODEL_DIR"] = model_dir
        
        if "MODEL_PATHS" not in kwargs:
            model_paths = os.getenv("YOLO_SERVICE_MODEL_PATHS")
            if model_paths:
                kwargs["MODEL_PATHS"] = model_paths
        
        if "MODEL_PATH" not in kwargs:
            model_path = os.getenv("YOLO_SERVICE_MODEL_PATH")
            if model_path:
                model_path_obj = Path(model_path)
                if not model_path_obj.is_absolute():
                    base_dir = Path(__file__).parent.parent
                    model_path = str(base_dir / model_path_obj)
                else:
                    model_path = str(model_path_obj)
                kwargs["MODEL_PATH"] = model_path
            else:
                base_dir = Path(__file__).parent.parent
                default_path = base_dir / "module" / "model.pt"
                kwargs["MODEL_PATH"] = str(default_path)
        if "SAVE_TMP_DIR" not in kwargs:
            save_tmp_dir = os.getenv("save_tmp_dir", "./tmp")
            save_tmp_dir_obj = Path(save_tmp_dir)
            if not save_tmp_dir_obj.is_absolute():
                base_dir = Path(__file__).parent.parent
                save_tmp_dir = str(base_dir / save_tmp_dir_obj)
            else:
                save_tmp_dir = str(save_tmp_dir_obj)
            kwargs["SAVE_TMP_DIR"] = Path(save_tmp_dir)
        if "SAVE_TMP_ENABLED" not in kwargs:
            save_tmp_enabled = os.getenv("save_tmp_enabled", "true").lower()
            kwargs["SAVE_TMP_ENABLED"] = save_tmp_enabled in ("true", "1", "yes", "on")

        super().__init__(**kwargs)


settings = Settings()

