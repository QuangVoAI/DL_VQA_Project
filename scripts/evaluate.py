"""CLI script: Evaluate VQA models on test set."""

import argparse
import os
import sys
import torch
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.models.vqa_model import VQAModel
from src.engine.evaluator import evaluate_model, evaluate_by_question_type, get_failure_cases
from src.utils.helpers import get_device, setup_logging
from src.utils.visualization import plot_radar_chart, plot_confusion_matrix

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()
    
    # ... (Logic load dataset/vocab tương tự train.py)

    for name in (args.models or cfg.model_variants.keys()):
        model = VQAModel(len(question_vocab), len(answer_vocab), **cfg.model_variants[name]).to(device)
        
        # Đánh giá bằng Beam Search
        res = evaluate_model(model, test_loader, answer_vocab, question_vocab, device, name=name)
        
        # Phân tích sâu
        q_results = evaluate_by_question_type(res["preds"], res["refs"], res["questions"])
        plot_confusion_matrix(res["preds"], res["refs"], res["questions"], f"logs/cm_{name}.png")

if __name__ == "__main__":
    main()