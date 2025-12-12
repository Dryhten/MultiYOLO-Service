# MultiYOLO Service

一个基于 FastAPI 的通用 YOLO 多模型检测服务，支持多种硬件加速（MUSA GPU、CUDA GPU、MPS、CPU），提供 RESTful API 接口进行图像目标检测。

## ✨ 特性

- 🚀 **高性能推理**：基于原生 PyTorch，支持多种硬件加速
  - 优先使用 MUSA GPU（摩尔线程）
  - 支持 CUDA GPU（NVIDIA）
  - 支持 Apple MPS（暂未测试，欢迎贡献代码）
  - 自动降级到 CPU
- 🔧 **灵活配置**：支持通过环境变量动态配置端口和模型文件
- 📦 **多模型管理**：支持同时加载和管理多个 YOLO 模型
  - 支持模型目录自动加载（推荐）
  - 支持多个模型路径配置
  - 向后兼容单个模型配置
- 🐳 **Docker 支持**：提供完整的 Docker 镜像和部署方案
- 🎯 **通用模型支持**：自动解析模型元数据，支持任意 YOLO 模型
- 📊 **详细日志**：基于 Loguru 的日志系统，支持日志文件输出
- 🔌 **RESTful API**：标准的 FastAPI 接口，支持 CORS
- 📝 **健康检查**：内置健康检查端点
- 📚 **完整文档**：提供详细的 API 文档和使用示例

## 📋 目录结构

```
general_yolo/
├── core/                    # 核心模块
│   ├── config.py           # 配置管理
│   ├── interfaces.py       # 检测器接口定义
│   └── logging_config.py   # 日志配置
├── services/                # 服务层
│   ├── detection.py        # API 路由和业务逻辑
│   └── detectors/          # 检测器实现
│       └── torch_detector.py  # PyTorch 检测器
├── module/                  # 模型文件目录
├── logs/                    # 日志文件目录
├── main.py                  # 应用入口
├── classes.txt              # 类别标签文件（可选）
├── requirements.txt         # Python 依赖
├── pyproject.toml           # 项目配置
├── Dockerfile               # Docker 镜像构建文件
├── .env.example             # 环境变量示例
└── README.md                # 项目说明文档
```

## 🚀 快速开始

### MUSA 部署

MUSA GPU 推荐使用 Docker 容器方式部署，容器内已配置好 conda 环境。

#### 一、环境准备

```bash
# 1. 更新系统包列表
sudo apt update

# 2. 安装 Python pip（如果已安装可跳过）
sudo apt install python3-pip -y

# 3. 安装官方 musa-deploy 工具
sudo pip install musa-deploy -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### 二、一键拉起 Torch_MUSA 开发 & 推理的 docker 容器

```bash
sudo musa-deploy --demo torch_musa \
                 --name yolo \
                 -v /home/server:/home/server \
                 --network host \
                 --pid host
```

**参数说明：**
- `--name yolo`：容器名称，可自定义
- `-v /home/server:/home/server`：将主机项目目录挂载进容器（请根据实际路径修改）
- `--network host --pid host`：推荐加上，便于后续调试和性能最优
- `--force`：如果已安装驱动但版本不兼容，加上此参数可强制更换驱动到目标容器适配的版本

**注意事项：**
- 如果已经安装了驱动，运行以上命令报驱动版本不兼容的问题，需要加上 `--force` 参数，系统会自动回退到兼容的驱动版本
- 安装到一半会提示重启服务器，启动完服务器再运行一次这个命令
- 没提示重启服务器就安装结束的，可以直接进入下一步（因为你装过这个容器所需的目标驱动了）

#### 三、进入容器并完成环境补全

```bash
# 进入docker容器，会自动激活conda环境
sudo docker exec -it yolo bash
```

在容器内执行以下命令安装视觉模型相关依赖：

```bash
# 如果要跑视觉相关的模型(yolo)，容器内执行以下两行命令安装相关依赖
apt update && apt install -y libgl1-mesa-glx libglib2.0-0
conda install -c conda-forge libstdcxx-ng=13 -y
```

#### 四、安装项目依赖

在容器内执行：

```bash
# 进入项目目录（根据实际挂载路径调整）
cd /home/server

# 安装项目依赖
pip install -r requirements.txt
```

#### 五、配置环境变量

在容器内复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 服务配置
YOLO_SERVICE_HOST=0.0.0.0
YOLO_SERVICE_PORT=8003
YOLO_SERVICE_RELOAD=false
YOLO_SERVICE_LOG_LEVEL=INFO
YOLO_SERVICE_LOG_DIR=./logs

# 模型配置（支持三种方式，优先级从高到低）：
# 方式1: 模型目录（推荐）- 自动加载目录下所有.pt文件
YOLO_SERVICE_MODEL_DIR=./module
# 方式2: 多个模型路径，用逗号分隔
# YOLO_SERVICE_MODEL_PATHS=./module/model1.pt,./module/model2.pt
# 方式3: 单个模型路径（向后兼容，如果上面两个都未配置则使用此配置）
# YOLO_SERVICE_MODEL_PATH=./module/model.pt

# 临时文件保存配置
save_tmp_dir=./tmp
save_tmp_enabled=false
```

#### 六、准备模型文件

将 YOLO 模型文件（`.pt` 格式）放置在 `module/` 目录下，例如：

```
module/
  ├── model1.pt
  ├── model2.pt
  └── model3.pt
```

**模型配置方式说明：**

1. **模型目录方式（推荐）**：设置 `YOLO_SERVICE_MODEL_DIR`，会自动加载该目录下所有 `.pt` 文件
2. **多模型路径方式**：设置 `YOLO_SERVICE_MODEL_PATHS`，用逗号分隔多个模型路径
3. **单模型路径方式**：设置 `YOLO_SERVICE_MODEL_PATH`，仅加载单个模型（向后兼容）

#### 七、运行服务

在容器内执行：

```bash
python main.py
```

服务启动后，访问：
- API 文档（Swagger UI）：http://localhost:8003/docs
- API 文档（ReDoc）：http://localhost:8003/redoc
- 健康检查：http://localhost:8003/health
- 完整 API 文档：查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

---

### CUDA 部署

#### 一、环境要求

- Python >= 3.12
- PyTorch >= 2.0.0（带 CUDA 支持）
- CUDA 驱动和工具包（根据 PyTorch 版本要求）

#### 二、安装依赖

```bash
# 使用 uv 安装依赖
uv sync
```

#### 三、配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 服务配置
YOLO_SERVICE_HOST=0.0.0.0
YOLO_SERVICE_PORT=8003
YOLO_SERVICE_RELOAD=false
YOLO_SERVICE_LOG_LEVEL=INFO
YOLO_SERVICE_LOG_DIR=./logs

# 模型配置（支持三种方式，优先级从高到低）：
# 方式1: 模型目录（推荐）- 自动加载目录下所有.pt文件
YOLO_SERVICE_MODEL_DIR=./module
# 方式2: 多个模型路径，用逗号分隔
# YOLO_SERVICE_MODEL_PATHS=./module/model1.pt,./module/model2.pt
# 方式3: 单个模型路径（向后兼容，如果上面两个都未配置则使用此配置）
# YOLO_SERVICE_MODEL_PATH=./module/model.pt

# 临时文件保存配置
save_tmp_dir=./tmp
save_tmp_enabled=false
```

#### 四、准备模型文件

将 YOLO 模型文件（`.pt` 格式）放置在 `module/` 目录下，例如：

```
module/
  ├── model1.pt
  ├── model2.pt
  └── model3.pt
```

**模型配置方式说明：**

1. **模型目录方式（推荐）**：设置 `YOLO_SERVICE_MODEL_DIR`，会自动加载该目录下所有 `.pt` 文件
2. **多模型路径方式**：设置 `YOLO_SERVICE_MODEL_PATHS`，用逗号分隔多个模型路径
3. **单模型路径方式**：设置 `YOLO_SERVICE_MODEL_PATH`，仅加载单个模型（向后兼容）

#### 五、运行服务

```bash
uv run main.py
```

服务启动后，访问：
- API 文档（Swagger UI）：http://localhost:8003/docs
- API 文档（ReDoc）：http://localhost:8003/redoc
- 健康检查：http://localhost:8003/health
- 完整 API 文档：查看 [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)



## 📡 API 接口

### 快速参考

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 获取服务信息 |
| `/health` | GET | 健康检查 |
| `/api/v1/models` | GET | 列出所有已加载的模型 |
| `/api/v1/device-info` | GET | 获取设备信息 |
| `/api/v1/detect` | POST | 执行目标检测 |

### 根路径

```bash
GET /
```

返回服务基本信息，包括已加载的模型列表。

### 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "service": "yolo-detection-service"
}
```

### 列出所有模型

```bash
GET /api/v1/models
```

响应：
```json
{
  "status": "success",
  "models": ["model1", "model2"],
  "count": 2
}
```

### 获取设备信息

```bash
GET /api/v1/device-info?model_name=model1
```

可选参数：
- `model_name`: 模型名称（可选），如果不指定则返回默认模型的信息

响应：
```json
{
  "status": "success",
  "model_name": "model1",
  "device_info": {
    "device": "cuda:0",
    "device_type": "CUDA GPU",
    "device_name": "NVIDIA GeForce RTX 3090",
    "device_count": 1,
    "memory_total": "24.00 GB"
  }
}
```

### 目标检测

```bash
POST /api/v1/detect
Content-Type: multipart/form-data
```

请求参数：
- `file`: 图片文件（multipart/form-data，必填）
- `model_name`: 模型名称（必填），使用 `/api/v1/models` 获取可用模型列表
- `conf_threshold`: 检测阈值（可选），范围 0-1，默认值 0.5

响应：
```json
{
  "detections": {
    "car": 2,
    "person": 1
  },
  "total_objects": 3,
  "details": [
    {
      "class_name": "car",
      "class_id": 1,
      "confidence": 0.95,
      "bbox": [100.5, 200.3, 300.7, 400.9]
    }
  ]
}
```

### 使用示例

**使用 curl：**

```bash
# 获取模型列表
curl http://localhost:8003/api/v1/models

# 执行检测（指定阈值）
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_name=model1" \
  -F "conf_threshold=0.6"

# 执行检测（使用默认阈值0.5）
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_name=model1"
```

**使用 Python：**

```python
import requests

# 获取模型列表
response = requests.get("http://localhost:8003/api/v1/models")
models = response.json()["models"]
print(f"可用模型: {models}")

# 执行检测
url = "http://localhost:8003/api/v1/detect"
files = {"file": open("image.jpg", "rb")}
data = {
    "model_name": models[0],  # 使用第一个可用模型
    "conf_threshold": 0.5
}
response = requests.post(url, files=files, data=data)
result = response.json()
print(f"检测到 {result['total_objects']} 个对象")
```

**详细 API 文档请查看：[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)**

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `YOLO_SERVICE_HOST` | 服务监听地址 | `0.0.0.0` |
| `YOLO_SERVICE_PORT` | 服务端口 | `8003` |
| `YOLO_SERVICE_RELOAD` | 是否启用热重载 | `false` |
| `YOLO_SERVICE_LOG_LEVEL` | 日志级别 | `INFO` |
| `YOLO_SERVICE_LOG_DIR` | 日志目录 | `./logs` |
| `YOLO_SERVICE_MODEL_DIR` | 模型目录路径（优先使用，会加载目录下所有.pt文件） | - |
| `YOLO_SERVICE_MODEL_PATHS` | 多个模型路径，用逗号分隔 | - |
| `YOLO_SERVICE_MODEL_PATH` | 单个模型文件路径（向后兼容） | `./module/model.pt` |
| `save_tmp_dir` | 临时文件保存目录 | `./tmp` |
| `save_tmp_enabled` | 是否启用临时文件保存 | `false` |

### 模型文件

模型文件支持以下格式：
- `.pt` (PyTorch 模型)

模型文件应包含以下元数据（可选，会自动解析）：
- `names`: 类别名称字典或列表

如果模型文件中没有类别信息，系统会按以下优先级查找：
1. 模型 checkpoint 中的 `names` 字段
2. 模型对象的 `names` 属性
3. `classes.txt` 文件（模型目录或项目根目录）
4. 默认的 18 类映射（向后兼容）

## 🔧 开发

### 项目结构说明

- `core/`: 核心模块，包含配置、接口定义、日志等
- `services/`: 服务层，包含 API 路由和检测器实现
- `main.py`: FastAPI 应用入口

### 添加新的检测器实现

1. 在 `services/detectors/` 目录下创建新的检测器类
2. 继承 `BaseDetector` 并实现必要的方法
3. 在 `services/detection.py` 中注册新的检测器

### 日志

日志文件保存在 `logs/` 目录下，按日期和级别分类。日志配置在 `core/logging_config.py` 中。

## 📝 注意事项

1. **模型文件**：
   - 确保模型文件路径正确，且文件存在
   - 支持 `.pt` 格式的 PyTorch 模型文件
   - 推荐使用 `MODEL_DIR` 配置方式，自动加载目录下所有模型

2. **模型名称**：
   - 模型名称自动从文件名提取（不含扩展名）
   - 使用 `/api/v1/models` 接口获取可用的模型名称列表
   - 在调用检测接口时，必须使用正确的模型名称

3. **硬件支持**：
   - 使用 MUSA GPU 需要安装 `torch-musa`
   - 使用 CUDA GPU 需要安装带 CUDA 支持的 PyTorch
   - CPU 模式性能较慢，建议使用 GPU

4. **内存占用**：
   - 每个模型加载会占用一定内存
   - 多模型同时加载时，确保有足够的可用内存
   - GPU 模式下，确保显存充足

5. **端口冲突**：确保配置的端口未被其他服务占用

6. **API 使用**：
   - 所有 API 接口都支持 CORS，可在浏览器中直接调用
   - 检测接口的 `conf_threshold` 参数控制检测敏感度，值越高越严格
   - 边界框坐标格式为 `[x1, y1, x2, y2]`（左上角和右下角坐标）

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📚 文档

- [完整 API 文档](./API_DOCUMENTATION.md) - 详细的 API 接口说明和使用示例
- [Docker 使用指南](./DOCKER_USAGE.md) - Docker 部署和配置说明

## 🔗 相关链接

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [PyTorch 文档](https://pytorch.org/docs/)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [Docker 文档](https://docs.docker.com/)

