#!/usr/bin/env python3
"""
PaliGemma CLI
"""

import sys
import argparse
from pathlib import Path

# Add project root to path  
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from inference import SimpleInference


def create_parser():
    """Create argument parser"""
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
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # describe command - Describe an image
    describe_parser = subparsers.add_parser('describe', help='Describe an image')
    describe_parser.add_argument('image', help='Image file path')
    describe_parser.add_argument('--detail', action='store_true', help='Detailed description')
    
    # detect command - Object detection
    detect_parser = subparsers.add_parser('detect', help='Detect objects in image')
    detect_parser.add_argument('image', help='Image file path')
    detect_parser.add_argument('object', help='Object to detect')
    
    # Common parameters (for direct inference)
    parser.add_argument('-i', '--image', type=str, help='Image file path')
    parser.add_argument('-p', '--prompt', type=str, help='Text prompt')
    parser.add_argument('--model', type=str, default='paligemma2-3b-mix-224', help='Model path')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature (0.1-2.0)')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda/mps)')
    
    return parser


def handle_describe(args, engine):
    """Handle describe command"""
    if args.detail:
        prompt = "Describe this image in detail, including colors, objects, composition and atmosphere"
    else:
        prompt = "describe"
    
    return engine.generate(args.image, prompt, max_tokens=1024)


def handle_detect(args, engine):
    """Handle detect command"""
    print(f"Running object detection...")
    print(f"Looking for: {args.object}")
    
    prompt = f"detect {args.object}"
    
    # Use simplified inference engine with detection visualization
    engine.generate(args.image, prompt, max_tokens=1024, detection=True)
    
    return 0



def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize model (lazy loading)
    engine = None
    
    def get_engine():
        nonlocal engine
        if engine is None:
            print("Loading model... ⏳")
            engine = SimpleInference(args.model, args.device)
            print("Model loaded! ✅\n")
        return engine
    
    # Handle subcommands
    if args.command == 'describe':
        handle_describe(args, get_engine())
    
    elif args.command == 'detect':
        handle_detect(args, get_engine())
    
    # Handle direct inference (using -i and -p parameters)
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
