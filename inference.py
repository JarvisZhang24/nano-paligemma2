#!/usr/bin/env python3
"""
Simplified PaliGemma Inference Script
Focus on core inference functionality, removing redundant validation and download logic
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
    """Load model"""
    from scripts.load_weights import load_weights
    
    # Auto select device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load model - default to paligemma2
    model, tokenizer = load_weights(model_path, "paligemma2", device)
    model = model.to(device).eval()
    
    # Initialize processor
    vision_config = model.config.vision_config
    processor = PaliGemmaProcessor(
        tokenizer,
        num_image_tokens=vision_config.num_image_tokens,
        image_size=vision_config.image_size,
    )
    
    return model, processor, device

def print_current_settings(image_path, temperature, top_p):
    print("\n=== Current Settings ===")
    print(f"Image: {image_path}")
    print(f"Temperature: {temperature}")
    print(f"Top_p: {top_p}")
    print("=== Current Settings ===\n")
    
class SimpleInference:
    """Simple Inference Engine"""
    
    def __init__(self, model_path="paligemma2-3b-mix-224", device="auto"):
        self.model, self.processor, self.device = load_model(model_path, device)
    
    def generate(self, image_path, prompt, max_tokens=1024, temperature=0.8, top_p=0.9, detection=False):
        """Generate response"""
        # Load image
        image = Image.open(image_path)
        
        # Prepare input
        model_inputs = self.processor(text=[prompt], image=image)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        
        # Generate text
        kv_cache = KVCache()
        stop_token = self.processor.tokenizer.eos_token_id
        generated_tokens = []
        
        print(f"\n[Prompt] {prompt}")
        print("[Output] ", end="", flush=True)
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                )
                
                kv_cache = outputs["kv_cache"]
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Sampling
                next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = sample_top_p(next_token_logits, top_p)
                next_token = next_token.squeeze(0)
                generated_tokens.append(next_token)
                
                # Decoding and printing
                decoded_token = self.processor.tokenizer.decode(
                    next_token.item(), skip_special_tokens=False
                )
                print(decoded_token, end="", flush=True)
                
                # Check stop token
                if next_token.item() == stop_token:
                    break
                
                # Update input
                input_ids = next_token.unsqueeze(-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=self.device)], dim=-1
                )
        
        # Statistics
        elapsed = time.time() - start_time
        token_count = len(generated_tokens)
        print(f"\n[Stats] {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.1f} tokens/s)\n")
        
        # Detection visualization
        if detection and "detect" in prompt.lower() and generated_tokens:
            # Convert token to text for detection visualization
            token_ids = [token.item() for token in generated_tokens]
            decoded = self.processor.tokenizer.decode(token_ids, skip_special_tokens=False)
            print(f"[Detection] Showing detection results...")
            print(f"[Detection] Decoded output: {decoded}")
            display_detection(decoded, image_path)
        
        # Return full generated text (optional)
        return None  # Already printing output during generation


def main():
    """Main function - Interactive or single inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple PaliGemma Inference")
    parser.add_argument("--image", "-i", type=str, default="examples/parrots.png", help="Image path")
    parser.add_argument("--prompt", "-p", type=str, help="Prompt (if not provided, enters interactive mode)")
    parser.add_argument("--model", "-m", type=str, default="paligemma2-3b-mix-224", help="Model path")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p value")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--detection", action="store_true", help="Enable detection visualization for detect prompts")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("Initializing PaliGemma 2 Vision Language Model...")
    engine = SimpleInference(args.model, args.device)
    
    # Single inference mode
    if args.prompt:
        engine.generate(
            args.image, 
            args.prompt, 
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.detection
        )
    # Interactive mode
    else:
        print("\n=== Interactive Mode ===")
        print("Commands:")
        print("  exit                    - Quit the program")
        print("  /image <path>           - Change image")
        print("  /temperature <value>   - Set temperature (0.1-2.0)")
        print("  /top_p <value>         - Set top_p (0.1-1.0)")
        print("  /help                  - Show this help")
        print("  describe               - Describe the current image")
        print("  detect <object>        - Detect objects in image")       
        print_current_settings(args.image, args.temperature, args.top_p)
        
        current_image = args.image
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if user_input.lower() == "exit":
                    break
                
                if user_input.startswith("/image "):
                    current_image = user_input[7:].strip()
                    print(f"Image changed to: {current_image}\n")
                    print_current_settings(current_image, args.temperature, args.top_p)
                    continue
                # Update parameters
                if user_input.startswith("/temperature "):
                    args.temperature = float(user_input[13:].strip())
                    print(f"Temperature changed to: {args.temperature}\n")
                    print_current_settings(current_image, args.temperature, args.top_p)
                    continue
                if user_input.startswith("/top_p "):
                    args.top_p = float(user_input[7:].strip())
                    print(f"Top_p changed to: {args.top_p}\n")
                    print_current_settings(current_image, args.temperature, args.top_p)
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
                    print_current_settings(current_image, args.temperature, args.top_p)
                    continue
                
                # Skip empty input
                if not user_input:
                    continue

                # Automatically detect if it's a detection prompt
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
