"""
Utility functions for PaliGemma project
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
from PIL import Image
import numpy as np

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger('PaliGemma')

def get_device(force_cpu: bool = False) -> str:
    """智能选择计算设备"""
    if force_cpu:
        return 'cpu'
    
    if torch.cuda.is_available():
        # 检查CUDA内存
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"CUDA device found with {gpu_memory:.1f}GB memory")
        return 'cuda'
    elif torch.backends.mps.is_available():
        logging.info("MPS device found (Apple Silicon)")
        return 'mps'
    else:
        logging.info("Using CPU for inference")
        return 'cpu'

def validate_image(image_path: str) -> Tuple[bool, str]:
    """验证图像文件"""
    path = Path(image_path)
    
    if not path.exists():
        return False, f"File not found: {image_path}"
    
    if not path.is_file():
        return False, f"Not a file: {image_path}"
    
    # 检查文件扩展名
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    if path.suffix.lower() not in valid_extensions:
        return False, f"Unsupported image format: {path.suffix}"
    
    # 尝试打开图像
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Cannot open image: {str(e)}"

def load_json_config(config_path: str) -> Dict[str, Any]:
    """加载JSON配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config file: {e}")
        return {}

def save_json_config(config: Dict[str, Any], config_path: str):
    """保存配置到JSON文件"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"Configuration saved to {config_path}")

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def calculate_tokens_per_second(tokens: int, elapsed_time: float) -> float:
    """计算生成速度"""
    if elapsed_time > 0:
        return tokens / elapsed_time
    return 0.0

def memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    info = {
        'rss_gb': memory_info.rss / 1e9,  # Resident Set Size
        'vms_gb': memory_info.vms / 1e9,  # Virtual Memory Size
    }
    
    if torch.cuda.is_available():
        info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    
    return info

def print_memory_usage():
    """打印内存使用情况"""
    usage = memory_usage()
    print("\nMemory Usage:")
    print(f"  System RAM: {usage['rss_gb']:.2f} GB")
    if 'gpu_allocated_gb' in usage:
        print(f"  GPU Memory: {usage['gpu_allocated_gb']:.2f} GB allocated, "
              f"{usage['gpu_reserved_gb']:.2f} GB reserved")

class Timer:
    """计时器上下文管理器"""
    
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"[{self.name}] Starting...")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"[{self.name}] Completed in {format_time(self.elapsed)}")

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        self._display()
    
    def _display(self):
        """显示进度条"""
        percent = self.current / self.total * 100
        elapsed = time.time() - self.start_time
        
        # 估算剩余时间
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "N/A"
        
        # 进度条
        bar_length = 30
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{self.description}: [{bar}] {percent:.1f}% | ETA: {eta_str}", 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # 换行

def resize_image_aspect_ratio(
    image: Image.Image, 
    target_size: Tuple[int, int],
    method: str = 'pad'
) -> Image.Image:
    """
    调整图像大小，保持纵横比
    
    Args:
        image: 输入图像
        target_size: 目标大小 (width, height)
        method: 'pad' (填充) 或 'crop' (裁剪)
    """
    target_width, target_height = target_size
    
    if method == 'pad':
        # 计算缩放比例
        scale = min(target_width / image.width, target_height / image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # 调整大小
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建新图像并居中粘贴
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        new_image.paste(resized, (paste_x, paste_y))
        
        return new_image
    
    elif method == 'crop':
        # 计算缩放比例
        scale = max(target_width / image.width, target_height / image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # 调整大小
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 中心裁剪
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return resized.crop((left, top, right, bottom))
    
    else:
        raise ValueError(f"Unknown method: {method}")

def batch_process_images(
    image_paths: List[str],
    process_fn,
    batch_size: int = 4,
    show_progress: bool = True
) -> List[Any]:
    """
    批量处理图像
    
    Args:
        image_paths: 图像路径列表
        process_fn: 处理函数
        batch_size: 批次大小
        show_progress: 是否显示进度
    """
    results = []
    total = len(image_paths)
    
    if show_progress:
        tracker = ProgressTracker(total, "Processing images")
    
    for i in range(0, total, batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = []
        
        for path in batch:
            try:
                result = process_fn(path)
                batch_results.append(result)
            except Exception as e:
                logging.error(f"Error processing {path}: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        
        if show_progress:
            tracker.update(len(batch))
    
    return results

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    后面的配置会覆盖前面的
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result

def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证配置是否符合schema
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    for key, spec in schema.items():
        if 'required' in spec and spec['required']:
            if key not in config:
                errors.append(f"Missing required field: {key}")
                continue
        
        if key in config:
            value = config[key]
            
            # 检查类型
            if 'type' in spec:
                expected_type = spec['type']
                if not isinstance(value, expected_type):
                    errors.append(f"{key}: expected {expected_type.__name__}, "
                                f"got {type(value).__name__}")
            
            # 检查取值范围
            if 'min' in spec and value < spec['min']:
                errors.append(f"{key}: value {value} is below minimum {spec['min']}")
            
            if 'max' in spec and value > spec['max']:
                errors.append(f"{key}: value {value} is above maximum {spec['max']}")
            
            # 检查选项
            if 'choices' in spec and value not in spec['choices']:
                errors.append(f"{key}: value {value} not in allowed choices {spec['choices']}")
    
    return len(errors) == 0, errors

# 配置schema示例
CONFIG_SCHEMA = {
    'model_type': {
        'type': str,
        'required': True,
        'choices': ['paligemma', 'paligemma2']
    },
    'temperature': {
        'type': float,
        'required': False,
        'min': 0.0,
        'max': 2.0
    },
    'max_tokens_to_generate': {
        'type': int,
        'required': False,
        'min': 1,
        'max': 8192
    },
    'top_p': {
        'type': float,
        'required': False,
        'min': 0.0,
        'max': 1.0
    }
}
