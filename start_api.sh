#!/bin/bash

# MMAudio API 服务启动脚本

echo "=========================================="
echo "MMAudio API 服务启动脚本"
echo "=========================================="

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

echo "✓ Python 版本: $(python --version)"

# 检查依赖
echo ""
echo "检查依赖..."

if ! python -c "import fastapi" &> /dev/null; then
    echo "⚠ FastAPI 未安装，正在安装..."
    pip install -r requirements_api.txt
fi

if ! python -c "import torch" &> /dev/null; then
    echo "⚠ PyTorch 未安装，请先安装 MMAudio 依赖"
    echo "运行: pip install -e ."
    exit 1
fi

echo "✓ 依赖检查完成"

# 检查必要目录
echo ""
echo "检查目录..."
mkdir -p ./api_output
mkdir -p ./output
mkdir -p ./ext_weights

echo "✓ 目录准备完成"

# 检查 GPU
echo ""
echo "检查 GPU..."
python -c "import torch; print('✓ CUDA 可用:', torch.cuda.is_available()); print('  设备数量:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# 启动选项
echo ""
echo "=========================================="
echo "启动选项:"
echo "1. 开发模式 (单进程, 自动重载)"
echo "2. 生产模式 (多进程)"
echo "3. 调试模式 (详细日志)"
echo "=========================================="

read -p "请选择 (1-3, 默认 1): " choice
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "启动开发模式..."
        python main_api.py
        ;;
    2)
        read -p "进程数 (默认 4): " workers
        workers=${workers:-4}
        echo ""
        echo "启动生产模式 (${workers} 个进程)..."
        uvicorn main_api:app --host 0.0.0.0 --port 8000 --workers $workers
        ;;
    3)
        echo ""
        echo "启动调试模式..."
        uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload --log-level debug
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
