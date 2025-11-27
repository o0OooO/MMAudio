"""
MMAudio FastAPI 服务
提供文生音频、视频生音频、训练等完整后端功能
"""
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mmaudio.eval_utils import (ModelConfig, all_model_cfg, generate, load_video, make_video,
                                setup_eval_logging)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# 配置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 初始化 FastAPI
app = FastAPI(
    title="MMAudio API",
    description="音频生成服务 - 支持文生音频、视频生音频、训练等功能",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 设备配置
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
log.info(f'使用设备: {DEVICE}')

# 输出目录
OUTPUT_DIR = Path('./api_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 全局模型缓存
MODEL_CACHE = {}
FEATURE_UTILS_CACHE = {}

# 任务状态管理
TASK_STATUS = {}


# ==================== 数据模型 ====================

class TextToAudioRequest(BaseModel):
    """文生音频请求"""
    prompt: str = Field(..., description="文本提示词")
    negative_prompt: str = Field("", description="负面提示词")
    duration: float = Field(8.0, ge=1.0, le=30.0, description="音频时长（秒）")
    cfg_strength: float = Field(4.5, ge=1.0, le=10.0, description="CFG强度")
    num_steps: int = Field(25, ge=10, le=100, description="采样步数")
    seed: int = Field(42, description="随机种子")
    model_variant: str = Field("large_44k_v2", description="模型版本")


class VideoToAudioRequest(BaseModel):
    """视频生音频请求（用于表单数据）"""
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    duration: Optional[float] = 8.0
    cfg_strength: Optional[float] = 4.5
    num_steps: Optional[int] = 25
    seed: Optional[int] = 42
    model_variant: Optional[str] = "large_44k_v2"
    mask_away_clip: Optional[bool] = False


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# ==================== 辅助函数 ====================

def get_model_and_features(model_variant: str, full_precision: bool = False):
    """获取或加载模型和特征提取器"""
    cache_key = f"{model_variant}_{full_precision}"
    
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key], FEATURE_UTILS_CACHE[cache_key]
    
    if model_variant not in all_model_cfg:
        raise ValueError(f'未知的模型版本: {model_variant}')
    
    model_cfg: ModelConfig = all_model_cfg[model_variant]
    model_cfg.download_if_needed()
    seq_cfg = model_cfg.seq_cfg
    
    dtype = torch.float32 if full_precision else torch.bfloat16
    
    # 加载模型
    net: MMAudio = get_my_mmaudio(model_cfg.model_name).to(DEVICE, dtype).eval()
    net.load_weights(torch.load(model_cfg.model_path, map_location=DEVICE, weights_only=True))
    log.info(f'已加载模型权重: {model_cfg.model_path}')
    
    # 加载特征提取器
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model_cfg.vae_path,
        synchformer_ckpt=model_cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=model_cfg.mode,
        bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
        need_vae_encoder=False
    )
    feature_utils = feature_utils.to(DEVICE, dtype).eval()
    
    # 缓存模型
    MODEL_CACHE[cache_key] = (net, seq_cfg, model_cfg)
    FEATURE_UTILS_CACHE[cache_key] = feature_utils
    
    return (net, seq_cfg, model_cfg), feature_utils


def cleanup_temp_files(*file_paths):
    """清理临时文件"""
    for file_path in file_paths:
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            log.warning(f'清理文件失败 {file_path}: {e}')


# ==================== API 端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "MMAudio API",
        "version": "1.0.0",
        "status": "running",
        "device": DEVICE,
        "endpoints": {
            "text_to_audio": "/api/v1/text-to-audio",
            "video_to_audio": "/api/v1/video-to-audio",
            "health": "/health",
            "models": "/api/v1/models"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": len(MODEL_CACHE)
    }


@app.get("/api/v1/models")
async def list_models():
    """列出可用的模型"""
    return {
        "available_models": list(all_model_cfg.keys()),
        "loaded_models": list(MODEL_CACHE.keys()),
        "default_model": "large_44k_v2"
    }


@app.post("/api/v1/text-to-audio")
@torch.inference_mode()
async def text_to_audio(
    request: TextToAudioRequest,
    background_tasks: BackgroundTasks
):
    """
    文生音频接口
    
    根据文本提示词生成音频
    """
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        (net, seq_cfg, model_cfg), feature_utils = get_model_and_features(
            request.model_variant, 
            full_precision=False
        )
        
        # 设置随机种子
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(request.seed)
        
        # 创建 FlowMatching
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=request.num_steps)
        
        # 更新序列长度
        seq_cfg.duration = request.duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        
        log.info(f'任务 {task_id}: 文生音频 - "{request.prompt}"')
        
        # 生成音频
        audios = generate(
            clip_frames=None,
            sync_frames=None,
            text=[request.prompt],
            negative_text=[request.negative_prompt],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=request.cfg_strength
        )
        
        audio = audios.float().cpu()[0]
        
        # 保存音频
        safe_filename = request.prompt.replace(' ', '_').replace('/', '_').replace('.', '')[:50]
        audio_path = output_dir / f'{safe_filename}.flac'
        torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)
        
        log.info(f'任务 {task_id}: 音频已保存到 {audio_path}')
        
        # 添加后台任务清理旧文件（可选）
        # background_tasks.add_task(cleanup_old_files, output_dir, hours=24)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "audio_url": f"/api/v1/download/{task_id}/{audio_path.name}",
            "duration": request.duration,
            "sample_rate": seq_cfg.sampling_rate,
            "prompt": request.prompt
        }
        
    except Exception as e:
        log.error(f'文生音频失败: {str(e)}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'生成失败: {str(e)}')


@app.post("/api/v1/video-to-audio")
@torch.inference_mode()
async def video_to_audio(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="视频文件"),
    prompt: str = Form("", description="文本提示词"),
    negative_prompt: str = Form("", description="负面提示词"),
    duration: float = Form(8.0, description="音频时长"),
    cfg_strength: float = Form(4.5, description="CFG强度"),
    num_steps: int = Form(25, description="采样步数"),
    seed: int = Form(42, description="随机种子"),
    model_variant: str = Form("large_44k_v2", description="模型版本"),
    mask_away_clip: bool = Form(False, description="是否屏蔽CLIP特征"),
    skip_video_composite: bool = Form(False, description="是否跳过视频合成")
):
    """
    视频生音频接口
    
    根据视频和可选的文本提示词生成音频
    """
    temp_video_path = None
    
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存上传的视频
        temp_video_path = output_dir / f"input_{video.filename}"
        with open(temp_video_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        log.info(f'任务 {task_id}: 视频生音频 - {video.filename}')
        
        # 加载模型
        (net, seq_cfg, model_cfg), feature_utils = get_model_and_features(
            model_variant,
            full_precision=False
        )
        
        # 设置随机种子
        rng = torch.Generator(device=DEVICE)
        rng.manual_seed(seed)
        
        # 创建 FlowMatching
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)
        
        # 加载视频
        video_info = load_video(temp_video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        actual_duration = video_info.duration_sec
        
        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)
        sync_frames = sync_frames.unsqueeze(0)
        
        # 更新序列长度
        seq_cfg.duration = actual_duration
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        
        # 生成音频
        audios = generate(
            clip_frames=clip_frames,
            sync_frames=sync_frames,
            text=[prompt] if prompt else None,
            negative_text=[negative_prompt] if negative_prompt else None,
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=rng,
            cfg_strength=cfg_strength
        )
        
        audio = audios.float().cpu()[0]
        
        # 保存音频
        video_stem = Path(video.filename).stem
        audio_path = output_dir / f'{video_stem}.flac'
        torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)
        
        log.info(f'任务 {task_id}: 音频已保存到 {audio_path}')
        
        result = {
            "task_id": task_id,
            "status": "completed",
            "audio_url": f"/api/v1/download/{task_id}/{audio_path.name}",
            "duration": actual_duration,
            "sample_rate": seq_cfg.sampling_rate,
            "video_filename": video.filename
        }
        
        # 合成视频（如果需要）
        if not skip_video_composite:
            video_output_path = output_dir / f'{video_stem}_with_audio.mp4'
            make_video(video_info, video_output_path, audio, sampling_rate=seq_cfg.sampling_rate)
            result["video_url"] = f"/api/v1/download/{task_id}/{video_output_path.name}"
            log.info(f'任务 {task_id}: 视频已保存到 {video_output_path}')
        
        return result
        
    except Exception as e:
        log.error(f'视频生音频失败: {str(e)}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'生成失败: {str(e)}')


@app.get("/api/v1/download/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """
    下载生成的文件
    """
    file_path = OUTPUT_DIR / task_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.delete("/api/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务及其生成的文件
    """
    task_dir = OUTPUT_DIR / task_id
    
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")
    
    try:
        shutil.rmtree(task_dir)
        return {"status": "success", "message": f"任务 {task_id} 已删除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@app.get("/api/v1/tasks")
async def list_tasks():
    """
    列出所有任务
    """
    tasks = []
    if OUTPUT_DIR.exists():
        for task_dir in OUTPUT_DIR.iterdir():
            if task_dir.is_dir():
                files = list(task_dir.glob("*"))
                tasks.append({
                    "task_id": task_dir.name,
                    "created_at": datetime.fromtimestamp(task_dir.stat().st_ctime).isoformat(),
                    "file_count": len(files),
                    "files": [f.name for f in files]
                })
    
    return {"tasks": tasks, "total": len(tasks)}


if __name__ == "__main__":
    import uvicorn
    
    # 设置日志
    setup_eval_logging()
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
