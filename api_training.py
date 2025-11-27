"""
MMAudio 训练 API
提供模型训练、评估等功能
"""
import logging
import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# 创建训练路由
router = APIRouter(prefix="/api/v1/training", tags=["training"])

# 训练任务状态
TRAINING_TASKS = {}


# ==================== 数据模型 ====================

class TrainingConfig(BaseModel):
    """训练配置"""
    exp_id: str = Field(..., description="实验ID")
    model: str = Field("small_16k", description="模型类型")
    batch_size: int = Field(512, description="批次大小")
    num_iterations: int = Field(300000, description="训练迭代次数")
    learning_rate: float = Field(1e-4, description="学习率")
    num_workers: int = Field(10, description="数据加载器工作进程数")
    seed: int = Field(14159265, description="随机种子")
    checkpoint: Optional[str] = Field(None, description="检查点路径")
    weights: Optional[str] = Field(None, description="预训练权重路径")
    debug: bool = Field(False, description="调试模式")


class EvaluationConfig(BaseModel):
    """评估配置"""
    exp_id: str = Field(..., description="实验ID")
    model: str = Field("large_44k_v2", description="模型版本")
    dataset: str = Field("audiocaps", description="数据集名称")
    duration_s: float = Field(8.0, description="音频时长")
    batch_size: int = Field(16, description="批次大小")
    weights: Optional[str] = Field(None, description="模型权重路径")
    output_name: Optional[str] = Field(None, description="输出名称")


class TrainingStatus(BaseModel):
    """训练状态"""
    task_id: str
    exp_id: str
    status: str  # running, completed, failed, stopped
    start_time: str
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    error: Optional[str] = None


# ==================== 辅助函数 ====================

def run_training_process(task_id: str, config: TrainingConfig):
    """在后台运行训练进程"""
    try:
        TRAINING_TASKS[task_id]["status"] = "running"
        
        # 构建训练命令
        cmd = [
            "python", "train.py",
            f"exp_id={config.exp_id}",
            f"model={config.model}",
            f"batch_size={config.batch_size}",
            f"num_iterations={config.num_iterations}",
            f"learning_rate={config.learning_rate}",
            f"num_workers={config.num_workers}",
            f"seed={config.seed}",
            f"debug={config.debug}"
        ]
        
        if config.checkpoint:
            cmd.append(f"checkpoint={config.checkpoint}")
        if config.weights:
            cmd.append(f"weights={config.weights}")
        
        # 设置日志文件
        log_dir = Path(f"./output/{config.exp_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"training_{task_id}.log"
        
        TRAINING_TASKS[task_id]["log_file"] = str(log_file)
        TRAINING_TASKS[task_id]["checkpoint_dir"] = str(log_dir)
        
        # 运行训练
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            TRAINING_TASKS[task_id]["process"] = process
            process.wait()
            
            if process.returncode == 0:
                TRAINING_TASKS[task_id]["status"] = "completed"
                log.info(f"训练任务 {task_id} 完成")
            else:
                TRAINING_TASKS[task_id]["status"] = "failed"
                TRAINING_TASKS[task_id]["error"] = f"进程退出码: {process.returncode}"
                log.error(f"训练任务 {task_id} 失败")
                
    except Exception as e:
        TRAINING_TASKS[task_id]["status"] = "failed"
        TRAINING_TASKS[task_id]["error"] = str(e)
        log.error(f"训练任务 {task_id} 异常: {e}", exc_info=True)


def run_evaluation_process(task_id: str, config: EvaluationConfig):
    """在后台运行评估进程"""
    try:
        TRAINING_TASKS[task_id]["status"] = "running"
        
        # 构建评估命令
        cmd = [
            "python", "batch_eval.py",
            f"exp_id={config.exp_id}",
            f"model={config.model}",
            f"dataset={config.dataset}",
            f"duration_s={config.duration_s}",
            f"batch_size={config.batch_size}"
        ]
        
        if config.weights:
            cmd.append(f"weights={config.weights}")
        if config.output_name:
            cmd.append(f"output_name={config.output_name}")
        
        # 设置日志文件
        log_dir = Path(f"./output/{config.exp_id}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"evaluation_{task_id}.log"
        
        TRAINING_TASKS[task_id]["log_file"] = str(log_file)
        
        # 运行评估
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            TRAINING_TASKS[task_id]["process"] = process
            process.wait()
            
            if process.returncode == 0:
                TRAINING_TASKS[task_id]["status"] = "completed"
                log.info(f"评估任务 {task_id} 完成")
            else:
                TRAINING_TASKS[task_id]["status"] = "failed"
                TRAINING_TASKS[task_id]["error"] = f"进程退出码: {process.returncode}"
                log.error(f"评估任务 {task_id} 失败")
                
    except Exception as e:
        TRAINING_TASKS[task_id]["status"] = "failed"
        TRAINING_TASKS[task_id]["error"] = str(e)
        log.error(f"评估任务 {task_id} 异常: {e}", exc_info=True)


# ==================== API 端点 ====================

@router.post("/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """
    启动训练任务
    """
    import uuid
    
    task_id = str(uuid.uuid4())
    
    TRAINING_TASKS[task_id] = {
        "task_id": task_id,
        "exp_id": config.exp_id,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "config": config.dict(),
        "total_iterations": config.num_iterations
    }
    
    # 在后台启动训练
    background_tasks.add_task(run_training_process, task_id, config)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "训练任务已创建，正在启动...",
        "exp_id": config.exp_id
    }


@router.get("/status/{task_id}")
async def get_training_status(task_id: str):
    """
    获取训练任务状态
    """
    if task_id not in TRAINING_TASKS:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = TRAINING_TASKS[task_id]
    
    # 尝试从日志文件中提取当前迭代次数
    if task.get("log_file") and Path(task["log_file"]).exists():
        try:
            with open(task["log_file"], "r") as f:
                lines = f.readlines()
                # 简单解析最后几行查找迭代信息
                for line in reversed(lines[-50:]):
                    if "iteration" in line.lower() or "iter" in line.lower():
                        # 这里可以添加更复杂的解析逻辑
                        pass
        except Exception:
            pass
    
    return TrainingStatus(**task)


@router.post("/stop/{task_id}")
async def stop_training(task_id: str):
    """
    停止训练任务
    """
    if task_id not in TRAINING_TASKS:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = TRAINING_TASKS[task_id]
    
    if task["status"] not in ["running", "pending"]:
        raise HTTPException(status_code=400, detail="任务未在运行中")
    
    # 终止进程
    if "process" in task:
        try:
            task["process"].terminate()
            task["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            task["process"].kill()
    
    task["status"] = "stopped"
    
    return {
        "task_id": task_id,
        "status": "stopped",
        "message": "训练任务已停止"
    }


@router.get("/list")
async def list_training_tasks():
    """
    列出所有训练任务
    """
    return {
        "tasks": list(TRAINING_TASKS.values()),
        "total": len(TRAINING_TASKS)
    }


@router.get("/logs/{task_id}")
async def get_training_logs(
    task_id: str,
    lines: int = 100
):
    """
    获取训练日志
    """
    if task_id not in TRAINING_TASKS:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = TRAINING_TASKS[task_id]
    log_file = task.get("log_file")
    
    if not log_file or not Path(log_file).exists():
        raise HTTPException(status_code=404, detail="日志文件不存在")
    
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
        return {
            "task_id": task_id,
            "log_lines": recent_lines,
            "total_lines": len(all_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取日志失败: {str(e)}")


@router.post("/evaluate")
async def start_evaluation(
    config: EvaluationConfig,
    background_tasks: BackgroundTasks
):
    """
    启动评估任务
    """
    import uuid
    
    task_id = str(uuid.uuid4())
    
    TRAINING_TASKS[task_id] = {
        "task_id": task_id,
        "exp_id": config.exp_id,
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "config": config.dict(),
        "type": "evaluation"
    }
    
    # 在后台启动评估
    background_tasks.add_task(run_evaluation_process, task_id, config)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "评估任务已创建，正在启动...",
        "exp_id": config.exp_id
    }


@router.get("/checkpoints/{exp_id}")
async def list_checkpoints(exp_id: str):
    """
    列出实验的所有检查点
    """
    checkpoint_dir = Path(f"./output/{exp_id}")
    
    if not checkpoint_dir.exists():
        return {"checkpoints": [], "total": 0}
    
    checkpoints = []
    for ckpt_file in checkpoint_dir.glob("*.pth"):
        stat = ckpt_file.stat()
        checkpoints.append({
            "filename": ckpt_file.name,
            "path": str(ckpt_file),
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {
        "exp_id": exp_id,
        "checkpoints": checkpoints,
        "total": len(checkpoints)
    }


@router.delete("/checkpoints/{exp_id}/{filename}")
async def delete_checkpoint(exp_id: str, filename: str):
    """
    删除指定的检查点文件
    """
    checkpoint_path = Path(f"./output/{exp_id}/{filename}")
    
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="检查点文件不存在")
    
    if not checkpoint_path.suffix == ".pth":
        raise HTTPException(status_code=400, detail="只能删除 .pth 文件")
    
    try:
        checkpoint_path.unlink()
        return {
            "status": "success",
            "message": f"检查点 {filename} 已删除"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
