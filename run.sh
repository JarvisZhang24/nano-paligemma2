#!/bin/bash

# PaliGemma 极简启动脚本
# 用法: ./run.sh [image] [prompt]

# 默认值
IMAGE="${1:-examples/car.png}"
PROMPT="${2:-describe this image}"
MODEL="paligemma2-3b-mix-224"

# 如果只有一个参数且不是图像文件，则作为prompt使用默认图像
if [ $# -eq 1 ] && [ ! -f "$1" ]; then
    PROMPT="$1"
    IMAGE="examples/car.png"
fi

# 运行推理
if [ $# -eq 0 ]; then
    # 无参数时进入交互模式
    echo "🚀 Starting PaliGemma Interactive Mode..."
    conda run -n paligemma2 python paligemma.py chat "$IMAGE"
else
    # 有参数时执行单次推理
    echo "🖼️  Image: $IMAGE"
    echo "💬 Prompt: $PROMPT"
    echo ""
    conda run -n paligemma2 python simple_inference.py -i "$IMAGE" -p "$PROMPT"
fi
