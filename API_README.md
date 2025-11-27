# MMAudio FastAPI 服务 - 完整使用文档

## 📦 文件清单

完整的 FastAPI 后端服务，包含以下文件：

### 核心服务文件
1. **api_server.py** - 音频生成 API（文生音频、视频生音频、文件管理）
2. **api_training.py** - 训练和评估 API（训练管理、状态监控、检查点管理）
3. **main_api.py** - 主服务入口（整合所有功能、系统监控）

### 辅助文件
4. **requirements_api.txt** - API 额外依赖
5. **test_api.py** - 自动化测试脚本
6. **start_api.sh** - 服务启动脚本
7. **web_demo.html** - Web 可视化界面
8. **API_使用说明.md** - 本文档

## 🚀 三步启动服务

### 第一步：安装依赖

```bash
# 1. 安装 MMAudio 基础依赖（如果还没安装）
pip install -e .

# 2. 安装 API 服务依赖
pip install -r requirements_api.txt
```

### 第二步：启动服务

选择以下任一方式：

```bash
# 方式 1：使用启动脚本（最简单，推荐）
./start_api.sh

# 方式 2：直接运行 Python
python main_api.py

# 方式 3：使用 uvicorn（生产环境）
uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 第三步：访问和测试

服务启动后，你可以：

1. **查看 API 文档**（自动生成的交互式文档）
   - 打开浏览器访问：http://localhost:8000/docs
   - 可以直接在网页上测试所有接口

2. **使用 Web 界面**（图形化操作）
   - 用浏览器打开 `web_demo.html` 文件
   - 可视化操作，无需编程

3. **运行自动测试**
   ```bash
   python test_api.py
   ```

## 🎯 核心功能说明

### 1️⃣ 文生音频 (Text-to-Audio)

**功能**：输入文本描述，AI 生成对应的音频

**API 端点**：`POST /api/v1/text-to-audio`

**使用示例**：

```bash
# 命令行测试
curl -X POST "http://localhost:8000/api/v1/text-to-audio" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "狗在大声叫",
    "duration": 8.0,
    "model_variant": "large_44k_v2"
  }'
```

```python
# Python 代码
import requests

response = requests.post(
    "http://localhost:8000/api/v1/text-to-audio",
    json={
        "prompt": "海浪拍打沙滩的声音",
        "duration": 8.0,
        "cfg_strength": 4.5,
        "num_steps": 25,
        "model_variant": "large_44k_v2"
    }
)

result = response.json()
print(f"任务ID: {result['task_id']}")
print(f"音频下载地址: http://localhost:8000{result['audio_url']}")
```

### 2️⃣ 视频生音频 (Video-to-Audio)

**功能**：为视频生成同步的音频，可选文本提示

**API 端点**：`POST /api/v1/video-to-audio`

**使用示例**：

```bash
# 命令行测试
curl -X POST "http://localhost:8000/api/v1/video-to-audio" \
  -F "video=@your_video.mp4" \
  -F "prompt=海浪声" \
  -F "duration=8.0" \
  -F "model_variant=large_44k_v2"
```

```python
# Python 代码
import requests

with open("my_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/video-to-audio",
        files={"video": f},
        data={
            "prompt": "鸟叫声",
            "duration": 8.0,
            "model_variant": "large_44k_v2"
        }
    )

result = response.json()
print(f"音频: http://localhost:8000{result['audio_url']}")
print(f"合成视频: http://localhost:8000{result['video_url']}")
```

### 3️⃣ 模型训练

**功能**：启动和管理模型训练任务

**API 端点**：`POST /api/v1/training/start`

**使用示例**：

```python
import requests

# 启动训练
response = requests.post(
    "http://localhost:8000/api/v1/training/start",
    json={
        "exp_id": "my_experiment",
        "model": "small_16k",
        "batch_size": 512,
        "num_iterations": 300000,
        "learning_rate": 0.0001
    }
)

task_id = response.json()["task_id"]
print(f"训练任务已启动，ID: {task_id}")

# 查看训练状态
status_response = requests.get(
    f"http://localhost:8000/api/v1/training/status/{task_id}"
)
print(status_response.json())

# 查看训练日志
logs_response = requests.get(
    f"http://localhost:8000/api/v1/training/logs/{task_id}?lines=50"
)
print(logs_response.json())
```

### 4️⃣ 模型评估

**功能**：批量评估模型性能

**API 端点**：`POST /api/v1/training/evaluate`

**使用示例**：

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/training/evaluate",
    json={
        "exp_id": "my_experiment",
        "model": "large_44k_v2",
        "dataset": "audiocaps",
        "duration_s": 8.0,
        "batch_size": 16
    }
)

print(response.json())
```

## 🎨 Web 界面使用

1. **打开 Web 界面**
   - 用浏览器打开 `web_demo.html` 文件
   - 或者访问 http://localhost:8000（如果配置了静态文件服务）

2. **文生音频标签页**
   - 输入文本描述（如"狗在叫"）
   - 选择模型版本（推荐 large_44k_v2）
   - 调整参数（时长、CFG强度等）
   - 点击"生成音频"
   - 等待生成完成，可以在线播放或下载

3. **视频生音频标签页**
   - 点击"选择文件"上传视频
   - 可选：输入文本提示词
   - 调整参数
   - 点击"生成音频"
   - 生成完成后可以播放音频和下载合成视频

4. **系统信息标签页**
   - 查看 GPU 状态
   - 监控系统资源使用
   - 点击"刷新"更新信息

## 📋 完整的 API 端点列表

### 音频生成
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/text-to-audio` | POST | 文生音频 |
| `/api/v1/video-to-audio` | POST | 视频生音频 |
| `/api/v1/download/{task_id}/{filename}` | GET | 下载生成的文件 |
| `/api/v1/tasks` | GET | 列出所有任务 |
| `/api/v1/tasks/{task_id}` | DELETE | 删除任务及文件 |

### 训练管理
| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/training/start` | POST | 启动训练任务 |
| `/api/v1/training/status/{task_id}` | GET | 查看训练状态 |
| `/api/v1/training/stop/{task_id}` | POST | 停止训练任务 |
| `/api/v1/training/logs/{task_id}` | GET | 查看训练日志 |
| `/api/v1/training/list` | GET | 列出所有训练任务 |
| `/api/v1/training/evaluate` | POST | 启动评估任务 |
| `/api/v1/training/checkpoints/{exp_id}` | GET | 列出检查点文件 |
| `/api/v1/training/checkpoints/{exp_id}/{filename}` | DELETE | 删除检查点 |

### 系统管理
| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/models` | GET | 列出可用模型 |
| `/api/v1/system/info` | GET | 系统信息（设备、版本） |
| `/api/v1/system/stats` | GET | 系统资源统计（CPU、内存、GPU） |

## ⚙️ 配置参数说明

### 模型版本选择
- `small_16k` - 小型 16kHz 模型（速度快，质量一般）
- `small_44k` - 小型 44kHz 模型
- `medium_44k` - 中型 44kHz 模型（平衡）
- `large_44k` - 大型 44kHz 模型（质量好）
- `large_44k_v2` - 大型 44kHz 模型 v2（**推荐**，质量最好）

### 生成参数
- **duration**（时长）：1-30 秒，推荐 8 秒
- **cfg_strength**（CFG强度）：1-10，推荐 4.5
  - 值越大，生成结果越符合提示词，但可能过拟合
  - 值越小，生成结果越多样，但可能偏离提示词
- **num_steps**（采样步数）：10-100，推荐 25
  - 步数越多，质量越好，但速度越慢
- **seed**（随机种子）：任意整数
  - 相同的种子和参数会生成相同的结果

### 训练参数
- **batch_size**：批次大小，根据 GPU 显存调整
- **num_iterations**：训练迭代次数
- **learning_rate**：学习率，推荐 1e-4

## 🔧 常见问题解决

### 问题 1：服务启动失败

**错误信息**：`ModuleNotFoundError: No module named 'fastapi'`

**解决方法**：
```bash
pip install -r requirements_api.txt
```

### 问题 2：CUDA 内存不足

**错误信息**：`RuntimeError: CUDA out of memory`

**解决方法**：
1. 使用较小的模型：`small_16k` 或 `small_44k`
2. 关闭其他占用 GPU 的程序
3. 减少 batch_size（训练时）

### 问题 3：模型下载慢或失败

**解决方法**：
1. 检查网络连接
2. 手动下载模型文件到 `./ext_weights/` 目录
3. 参考 `docs/MODELS.md` 获取下载链接

### 问题 4：生成的音频质量不好

**解决方法**：
1. 使用 `large_44k_v2` 模型
2. 增加 `num_steps` 到 50
3. 调整 `cfg_strength` 在 3-6 之间
4. 使用更详细的文本描述

### 问题 5：视频处理失败

**解决方法**：
1. 确保视频格式为 MP4, AVI 等常见格式
2. 使用 ffmpeg 转换视频：
   ```bash
   ffmpeg -i input.mov -c:v libx264 output.mp4
   ```
3. 检查视频文件是否损坏

## 💻 系统要求

### 最低配置
- Python 3.9+
- 8GB 内存
- CPU 模式可运行（速度较慢）

### 推荐配置
- Python 3.10+
- 16GB+ 内存
- NVIDIA GPU（8GB+ 显存）
- CUDA 11.8+

### GPU 显存占用
- `small_16k`：约 4GB
- `large_44k_v2`：约 6GB（推理模式）
- 训练模式：需要更多显存

## 📊 性能优化建议

### 提高生成速度
1. 使用 `small_16k` 模型
2. 减少 `num_steps` 到 15-20
3. 使用 GPU 而非 CPU

### 提高生成质量
1. 使用 `large_44k_v2` 模型
2. 增加 `num_steps` 到 50
3. 使用更详细的文本描述
4. 调整 `cfg_strength` 参数

### 节省显存
1. 使用较小的模型
2. 减少 batch_size
3. 关闭不必要的程序

## 🚀 生产部署建议

### 1. 使用多进程
```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. 使用 Nginx 反向代理
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

### 3. 使用 Docker 部署
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -e . && pip install -r requirements_api.txt
EXPOSE 8000
CMD ["python", "main_api.py"]
```

### 4. 添加 API 认证
建议在生产环境添加 API key 验证，防止滥用。

## 📚 相关文档

- **README.md** - MMAudio 项目原始文档
- **docs/TRAINING.md** - 训练详细文档
- **docs/EVAL.md** - 评估详细文档
- **docs/MODELS.md** - 模型下载和说明

## 🎯 下一步

1. **测试服务** - 运行 `python test_api.py`
2. **查看 API 文档** - 访问 http://localhost:8000/docs
3. **尝试 Web 界面** - 打开 `web_demo.html`
4. **集成到应用** - 参考上面的代码示例

## 💡 使用技巧

1. **文本提示词建议**
   - 使用英文描述效果更好
   - 描述要具体（如"a dog barking loudly"而不是"dog"）
   - 可以描述音色、节奏、环境等

2. **视频生音频建议**
   - 视频时长建议 8 秒左右
   - 可以添加文本提示词增强效果
   - 视频质量不影响音频生成

3. **批量处理**
   - 可以编写脚本批量调用 API
   - 使用异步请求提高效率

## 📞 获取帮助

如有问题：
1. 查看本文档
2. 运行 `python test_api.py` 进行诊断
3. 检查服务日志
4. 访问原项目 GitHub 仓库

---

**祝使用愉快！** 🎉
