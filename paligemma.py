#!/usr/bin/env python3
"""
PaliGemma 精简版CLI - 专注于推理
"""

import sys
import argparse
from pathlib import Path

# Add project root to path  
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from inference import SimpleInference


def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(
        description="PaliGemma Vision Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s describe car.png              # Describe an image
  %(prog)s detect car.png "car"          # Detect objects in image  
  %(prog)s -i photo.jpg -p "what is this?"  # Custom inference
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # describe命令 - 描述图像
    describe_parser = subparsers.add_parser('describe', help='Describe an image')
    describe_parser.add_argument('image', help='Image file path')
    describe_parser.add_argument('--detail', action='store_true', help='Detailed description')
    
    # detect命令 - 物体检测
    detect_parser = subparsers.add_parser('detect', help='Detect objects in image')
    detect_parser.add_argument('image', help='Image file path')
    detect_parser.add_argument('object', help='Object to detect')
    
    # 通用参数（用于直接调用）
    parser.add_argument('-i', '--image', type=str, help='Image file path')
    parser.add_argument('-p', '--prompt', type=str, help='Text prompt')
    parser.add_argument('--model', type=str, default='paligemma2-3b-mix-224', help='Model path')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature (0.1-2.0)')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda/mps)')
    
    return parser


def handle_describe(args, engine):
    """处理描述命令"""
    if args.detail:
        prompt = "Describe this image in detail, including colors, objects, composition and atmosphere"
    else:
        prompt = "describe"
    
    return engine.generate(args.image, prompt, max_tokens=1024)


def handle_detect(args, engine):
    """处理检测命令"""
    print(f"Running object detection...")
    print(f"Looking for: {args.object}")
    
    prompt = f"detect {args.object}"
    
    # 直接使用简化的推理引擎，启用检测可视化
    engine.generate(args.image, prompt, max_tokens=1024, detection=True)
    
    return 0



def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 初始化模型（延迟加载）
    engine = None
    
    def get_engine():
        nonlocal engine
        if engine is None:
            print("Loading model... ⏳")
            engine = SimpleInference(args.model, args.device)
            print("Model loaded! ✅\n")
        return engine
    
    # 处理子命令
    if args.command == 'describe':
        handle_describe(args, get_engine())
    
    elif args.command == 'detect':
        handle_detect(args, get_engine())
    
    # 处理直接推理（使用-i和-p参数）
    elif args.image and args.prompt:
        get_engine().generate(
            args.image,
            args.prompt,
            args.max_tokens,
            args.temperature
        )
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
