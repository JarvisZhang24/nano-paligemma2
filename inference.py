#!/usr/bin/env python3
"""
Simplified PaliGemma Inference Script
专注于核心推理功能，移除了冗余的验证和下载逻辑
"""

import torch
from PIL import Image
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.processor import PaliGemmaProcessor
from src.kv_cache import KVCache
from src.generation import sample_top_p
from src.detection import display_detection

def load_model(model_path="paligemma2-3b-mix-224", device="auto"):
    """加载模型的简化版本"""
    from scripts.download_weights import load_weights
    
    # 自动选择设备
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # 加载模型 - 默认使用paligemma2
    model, tokenizer = load_weights(model_path, "paligemma2", device)
    model = model.to(device).eval()
    
    # 初始化处理器
    vision_config = model.config.vision_config
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=vision_config.num_image_tokens,
        image_size=vision_config.image_size,
    )
    
    return model, processor, device


class SimpleInference:
    """简化的推理引擎"""
    
    def __init__(self, model_path="paligemma2-3b-mix-224", device="auto"):
        self.model, self.processor, self.device = load_model(model_path, device)
    
    def generate(self, image_path, prompt, max_tokens=1024, temperature=0.8, top_p=0.9, detection=False):
        """生成响应"""
        # 加载图像
        image = Image.open(image_path)
        
        # 准备输入
        model_inputs = self.processor(text=[prompt], images=[image])
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        
        # 生成文本
        kv_cache = KVCache()
        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []
        
        print(f"\n[Prompt] {prompt}")
        print("[Output] ", end="", flush=True)
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
                
                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]
                
                # 采样
                next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = sample_top_p(next_token_logits, top_p)
                next_token = next_token.squeeze(0)
                generated_tokens.append(next_token)
                
                # 解码并打印
                decoded_token = self.processor.tokenizer.decode(
                    next_token.item(), skip_special_tokens=False
                )
                print(decoded_token, end="", flush=True)
                
                # 检查停止标记
                if next_token.item() == stop_token:
                    break
                
                # 更新输入
                input_ids = next_token.unsqueeze(-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)], dim=-1
                )
        
        # 统计信息
        elapsed = time.time() - start_time
        token_count = len(generated_tokens)
        print(f"\n[Stats] {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.1f} tokens/s)\n")
        
        # 检测可视化处理
        if detection and "detect" in prompt.lower() and generated_tokens:
            # 将token转换为文本进行检测可视化
            token_ids = [token.item() for token in generated_tokens]
            decoded = self.processor.tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"[Detection] Showing detection results...")
            print(f"[Detection] Decoded output: {decoded}")
            display_detection(decoded, image_path)
        
        # 返回完整生成的文本（可选）
        return None  # 已经在生成过程中打印了输出


def main():
    """主函数 - 交互式或单次推理"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple PaliGemma Inference")
    parser.add_argument("--image", "-i", type=str, default="examples/car.png", help="Image path")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt (if not provided, enters interactive mode)")
    parser.add_argument("--model", "-m", type=str, default="paligemma2-3b-mix-224", help="Model path")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--detection", action="store_true", help="Enable detection visualization for detect prompts")
    
    args = parser.parse_args()
    
    # 初始化推理引擎
    print("Initializing PaliGemma 2 Vision Language Model...")
    engine = SimpleInference(args.model, args.device)
    
    # 单次推理模式
    if args.prompt:
        engine.generate(
            args.image, 
            args.prompt, 
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.detection
        )
    # 交互模式
    else:
        print("\n=== Interactive Mode ===")
        print("Commands:")
        print("  exit                    - Quit the program")
        print("  /image <path>          - Change image")
        print("  /temperature <value>   - Set temperature (0.1-2.0)")
        print("  /top_p <value>         - Set top_p (0.1-1.0)")
        print("  /help                  - Show this help")
        print("  describe               - Describe the current image")
        print("  detect <object>        - Detect objects in image")
        print(f"\nCurrent settings:")
        print(f"  Image: {args.image}")
        print(f"  Temperature: {args.temperature}")
        print(f"  Top_p: {args.top_p}\n")
        
        current_image = args.image
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() == "exit":
                    break
                
                if user_input.startswith("/image "):
                    current_image = user_input[7:].strip()
                    print(f"Image changed to: {current_image}\n")
                    continue
                # 更新参数
                if user_input.startswith("/temperature "):
                    args.temperature = float(user_input[13:].strip())
                    print(f"Temperature changed to: {args.temperature}\n")
                    continue
                if user_input.startswith("/top_p "):
                    args.top_p = float(user_input[7:].strip())
                    print(f"Top_p changed to: {args.top_p}\n")
                    continue
                
                if user_input.lower() == "/help":
                    print("\nCommands:")
                    print("  exit                    - Quit the program")
                    print("  /image <path>          - Change image")
                    print("  /temperature <value>   - Set temperature (0.1-2.0)")
                    print("  /top_p <value>         - Set top_p (0.1-1.0)")
                    print("  /help                  - Show this help")
                    print("  describe               - Describe the current image")
                    print("  detect <object>        - Detect objects in image")
                    print(f"\nCurrent settings:")
                    print(f"  Image: {current_image}")
                    print(f"  Temperature: {args.temperature}")
                    print(f"  Top_p: {args.top_p}\n")
                    continue
                
                # Skip empty input
                if not user_input:
                    continue

                # 自动检测是否为检测prompt
                is_detection = "detect" in user_input.lower()

                engine.generate(
                    current_image,
                    user_input,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    is_detection
                )
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
