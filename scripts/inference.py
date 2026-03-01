"""CLI script: Run inference on images with trained VQA models."""

import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.models.vqa_model import VQAModel
from src.utils.helpers import get_device, decode_sequence

class VQAInferencePipeline:
    def __init__(self, checkpoint_path, vocab_path, device_str="auto"):
        self.device = get_device() if device_str == "auto" else torch.device(device_str)
        # Load vocab & model
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        # Khởi tạo model dựa trên state_dict hoặc config mặc định
        # ... (Logic khởi tạo và load_state_dict)

    def predict(self, image: Image.Image, question: str, beam_width: int = 5):
        # ... (Logic tiền xử lý và gọi model.generate)
        return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    pipe = VQAInferencePipeline(args.checkpoint, "data/processed/vocab_aokvqa.pth")
    if args.image and args.question:
        ans = pipe.predict(Image.open(args.image), args.question)
        print(f"Prediction: {ans}")

if __name__ == "__main__":
    main()