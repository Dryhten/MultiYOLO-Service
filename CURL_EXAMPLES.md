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

```bash
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_name=model1" \
  -F "conf_threshold=0.5"
```

**参数说明**:
- `file`: 图片文件（必填）
- `model_name`: 模型名称（必填）
- `conf_threshold`: 置信度阈值，默认 0.5（可选）

---

## 完整工作流程

```bash
# 1. 健康检查
curl -X GET "http://localhost:8003/health"

# 2. 获取模型列表
curl -X GET "http://localhost:8003/api/v1/models"

# 3. 目标检测
curl -X POST "http://localhost:8003/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_name=model1" \
  -F "conf_threshold=0.5"
```
