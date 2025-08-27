#!/usr/bin/env python3
"""
Performance benchmark script for PaliGemma Vision Language Model
"""

import time
import torch
import psutil
import platform
from pathlib import Path
from inference import SimpleInference
import json

def get_system_info():
    """Get system information"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    
    return info

def benchmark_inference(model_path="paligemma2-3b-mix-224", device="auto"):
    """Benchmark inference performance"""
    print("🚀 Starting PaliGemma Performance Benchmark")
    print("=" * 50)
    
    # System info
    system_info = get_system_info()
    print("📊 System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize model
    print("🔧 Initializing model...")
    start_time = time.time()
    engine = SimpleInference(model_path, device)
    init_time = time.time() - start_time
    print(f"✅ Model initialized in {init_time:.2f} seconds\n")
    
    # Test images and prompts
    test_cases = [
        {"image": "examples/car.png", "prompt": "describe car", "type": "description"},
        {"image": "examples/car.png", "prompt": "detect car", "type": "detection"},
        {"image": "examples/tiger.jpg", "prompt": "describe tiger", "type": "description"},
        {"image": "examples/home.jpg", "prompt": "describe home", "type": "description"},
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}/{len(test_cases)}: {test_case['type']} - {Path(test_case['image']).name}")
        
        # Memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Run inference
        start_time = time.time()
        try:
            # Redirect output to capture tokens
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            engine.generate(
                test_case["image"],
                test_case["prompt"],
                max_tokens=100,
                temperature=0.8,
                top_p=0.9,
                detection=test_case["type"] == "detection"
            )
            
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        inference_time = time.time() - start_time
        
        # Memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        
        # Extract tokens count from output
        tokens_generated = 0
        if "[Stats]" in output:
            stats_line = [line for line in output.split('\n') if "[Stats]" in line][0]
            tokens_generated = int(stats_line.split()[1])
        
        tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
        
        result = {
            "test_case": test_case,
            "inference_time": round(inference_time, 3),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_per_second, 2),
            "memory_used_mb": round(memory_after - memory_before, 2),
            "peak_memory_mb": round(memory_after, 2)
        }
        
        results.append(result)
        
        print(f"  ⏱️  Inference time: {inference_time:.2f}s")
        print(f"  🎯 Tokens generated: {tokens_generated}")
        print(f"  🚀 Tokens/second: {tokens_per_second:.2f}")
        print(f"  💾 Memory used: {memory_after - memory_before:.1f}MB")
        print()
    
    # Summary
    print("📈 Benchmark Summary:")
    print("=" * 50)
    
    avg_tokens_per_sec = sum(r['tokens_per_second'] for r in results) / len(results)
    avg_inference_time = sum(r['inference_time'] for r in results) / len(results)
    total_tokens = sum(r['tokens_generated'] for r in results)
    
    print(f"Average tokens/second: {avg_tokens_per_sec:.2f}")
    print(f"Average inference time: {avg_inference_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Device: {engine.device}")
    
    # Save results
    benchmark_data = {
        "system_info": system_info,
        "model_path": model_path,
        "device": str(engine.device),
        "initialization_time": init_time,
        "test_results": results,
        "summary": {
            "avg_tokens_per_second": avg_tokens_per_sec,
            "avg_inference_time": avg_inference_time,
            "total_tokens": total_tokens
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    
    print("\n💾 Results saved to benchmark_results.json")
    return benchmark_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaliGemma Performance Benchmark")
    parser.add_argument("--model", type=str, default="paligemma2-3b-mix-224", help="Model path")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda/mps)")
    
    args = parser.parse_args()
    
    benchmark_inference(args.model, args.device)
