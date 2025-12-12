# YOLO Detection Service - cURL 命令文档

**服务地址**: `http://localhost:8003`  
**API 前缀**: `/api/v1`

---

## 1. 根路径

```bash
curl -X GET "http://localhost:8003/"
```

---

## 2. 健康检查

```bash
curl -X GET "http://localhost:8003/health"
```

---

## 3. 列出所有模型

```bash
curl -X GET "http://localhost:8003/api/v1/models"
```

---

## 4. 获取设备信息

```bash
# 指定模型
curl -X GET "http://localhost:8003/api/v1/device-info?model_name=model1"
```

---

## 5. 目标检测

### 5.1 单模型检测

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\"]"
```

### 5.2 单模型检测（指定阈值）

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\"]" \
  -F "conf_thresholds={\"model1\": 0.6}"
```

### 5.3 多模型检测（使用默认阈值）

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\", \"model2\", \"model3\"]"
```

### 5.4 多模型检测（为每个模型指定阈值）

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\", \"model2\", \"model3\"]" \
  -F "conf_thresholds={\"model1\": 0.5, \"model2\": 0.6, \"model3\": 0.7}"
```

### 5.5 多模型检测（部分模型指定阈值）

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\", \"model2\", \"model3\"]" \
  -F "conf_thresholds={\"model1\": 0.6}"
```

**参数说明**:
- `file`: 图片文件（必填）
- `model_names`: 模型名称数组，JSON格式（必填），支持单个或多个模型，如 `["model1"]` 或 `["model1", "model2"]`
- `conf_thresholds`: 每个模型的阈值映射，JSON格式（可选），如 `{"model1": 0.5, "model2": 0.6}`，未指定的模型使用默认值 0.5

**响应格式**:
```json
{
  "results": [
    {
      "model_name": "model1",
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
      ],
      "error": null
    },
    {
      "model_name": "model2",
      "detections": {},
      "total_objects": 0,
      "details": [],
      "error": null
    }
  ]
}
```

---

## 完整工作流程

```bash
# 1. 健康检查
curl -X GET "http://localhost:8003/health"

# 2. 获取模型列表
curl -X GET "http://localhost:8003/api/v1/models"

# 3. 单模型检测
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\"]"

# 4. 多模型检测（推荐）
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=[\"model1\", \"model2\"]" \
  -F "conf_thresholds={\"model1\": 0.5, \"model2\": 0.6}"
```

## 使用技巧

### 在 shell 中处理 JSON 参数

如果 JSON 字符串中包含特殊字符，可以使用单引号包裹：

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F 'model_names=["model1", "model2"]' \
  -F 'conf_thresholds={"model1": 0.5, "model2": 0.6}'
```

### 使用变量存储 JSON

```bash
# 定义模型列表
MODELS='["model1", "model2", "model3"]'

# 定义阈值配置
THRESHOLDS='{"model1": 0.5, "model2": 0.6, "model3": 0.7}'

# 执行检测
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_names=$MODELS" \
  -F "conf_thresholds=$THRESHOLDS"
```
