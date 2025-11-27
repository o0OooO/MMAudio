"""
MMAudio 完整 API 服务
整合所有功能：文生音频、视频生音频、训练、评估等
"""
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_training import router as training_router

# 导入主服务的端点
from api_server import (
    app as base_app,
    root, health_check, list_models,
    text_to_audio, video_to_audio,
    download_file, delete_task, list_tasks
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# 创建主应用
app = FastAPI(
    title="MMAudio 完整 API 服务",
    description="""
    MMAudio 音频生成与训练服务
    
    ## 功能模块
    
    ### 1. 音频生成
    - 文生音频 (Text-to-Audio)
    - 视频生音频 (Video-to-Audio)
    
    ### 2. 模型训练
    - 启动训练任务
    - 监控训练状态
    - 管理检查点
    
    ### 3. 模型评估
    - 批量评估
    - 性能指标
    
    ### 4. 文件管理
    - 下载生成的音频/视频
    - 管理任务文件
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册训练路由
app.include_router(training_router)

# 挂载静态文件（如果需要）
output_dir = Path("./api_output")
output_dir.mkdir(parents=True, exist_ok=True)


# ==================== 系统信息端点 ====================

@app.get("/api/v1/system/info")
async def system_info():
    """获取系统信息"""
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
    elif torch.backends.mps.is_available():
        device = 'mps'
        gpu_info = {"type": "Apple Silicon"}
    else:
        gpu_info = None
    
    return {
        "device": device,
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available()
    }


@app.get("/api/v1/system/stats")
async def system_stats():
    """获取系统统计信息"""
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "memory": {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        },
        "disk": {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent
        }
    }
    
    if torch.cuda.is_available():
        stats["gpu"] = {
            "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
        }
    
    return stats


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """服务启动时执行"""
    log.info("=" * 60)
    log.info("MMAudio API 服务启动")
    log.info("=" * 60)
    
    # 检查设备
    if torch.cuda.is_available():
        log.info(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        log.info(f"  GPU 数量: {torch.cuda.device_count()}")
        log.info(f"  显存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    elif torch.backends.mps.is_available():
        log.info("✓ Apple Silicon MPS 可用")
    else:
        log.warning("⚠ 仅 CPU 可用，性能可能较慢")
    
    # 检查必要的文件
    required_dirs = ["./ext_weights", "./output"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        log.info(f"✓ 目录已准备: {dir_path}")
    
    log.info("=" * 60)
    log.info("服务已就绪，访问 http://localhost:8000/docs 查看 API 文档")
    log.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时执行"""
    log.info("MMAudio API 服务正在关闭...")
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("✓ GPU 缓存已清理")


if __name__ == "__main__":
    import uvicorn
    
    # 启动服务
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 生产环境设为 False
        log_level="info",
        access_log=True
    )
