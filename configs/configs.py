import argparse

################################### Configurations ###################################
def get_args():
    parser = argparse.ArgumentParser(
        description="Run inference with PaliGemma/PaliGemma2 model."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="paligemma2",
        choices=["paligemma", "paligemma2"],
        help="Model type (choices: paligemma or paligemma2).",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/jarviszhang/CV_Project/PaliGemma Vision Language Model/paligemma2-3b-mix-224",
        # required=True,
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default = None,
        # required=True,
        help="Prompt text given to the model.",
    )
    parser.add_argument(
        "--detection",
        type=bool,
        default=True,
        help="Display object detection (True/False). Must be used with a 'Detect <entity>' prompt, with the fine-tuned model.",
    )

    parser.add_argument(
        "--image_file_path",
        type=str,
        default="/Users/jarviszhang/CV_Project/PaliGemma Vision Language Model/examples/Atlanta_Zoo_Panda.jpg",
        # required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--max_tokens_to_generate",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.6, 
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p sampling value."
    )
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="Enable sampling (True/False).",
    )
    parser.add_argument(
        "--only_cpu",
        type=bool,
        default=False,
        help="Run on CPU only (True/False).",
    )

    return parser.parse_args()
