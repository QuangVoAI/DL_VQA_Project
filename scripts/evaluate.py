"""CLI script: Evaluate VQA models on test set.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --config configs/default.yaml --models M4_Pretrained_Attn
    python scripts/evaluate.py --config configs/default.yaml --question-type-analysis
"""

from __future__ import annotations

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset

from src.config import Config
from src.data.preprocessing import extract_answer, expand_data_with_rationales
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.data.glove import download_glove, load_glove_embeddings
from src.models.vqa_model import VQAModel
from src.engine.evaluator import evaluate_model, evaluate_by_question_type, get_failure_cases
from src.utils.helpers import get_device, set_seed, setup_logging
from src.utils.visualization import (
    plot_radar_chart, plot_bar_chart,
    plot_confusion_matrix, plot_question_type_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VQA models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--question-type-analysis", action="store_true",
                        help="Run question-type breakdown analysis")
    parser.add_argument("--failure-analysis", action="store_true",
                        help="Show worst prediction failures")
    parser.add_argument("--num-failures", type=int, default=20,
                        help="Number of failure cases to show")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)
    logger = setup_logging(cfg.log_dir)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)

    # ── Data ──
    transform_eval = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Load saved vocab
    vocab_path = "data/processed/vocab_aokvqa.pth"
    vocabs = torch.load(vocab_path, weights_only=False)
    question_vocab = vocabs["question_vocab"]
    answer_vocab = vocabs["answer_vocab"]

    test_dataset = AOKVQA_Dataset(list(hf_val), question_vocab, answer_vocab, transform_eval)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size,
                             shuffle=False, collate_fn=collate_fn,
                             num_workers=0, pin_memory=False)

    # ── Evaluate ──
    variants = args.models or list(cfg.model_variants.keys())
    test_results = {}
    all_eval_data = {}

    for name in variants:
        if name not in cfg.model_variants:
            continue

        variant_cfg = cfg.model_variants[name]
        model = VQAModel(
            len(question_vocab), len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=0.0,  # no dropout during evaluation
            **variant_cfg,
        ).to(device)

        eval_data = evaluate_model(
            model, test_loader, answer_vocab, question_vocab,
            device, cfg.ckpt_dir, name, cfg.train.beam_width,
        )
        test_results[name] = eval_data["metrics"]
        all_eval_data[name] = eval_data

    # ── Print comparison table ──
    header = f"{'Model':<30s} {'Acc':>8s} {'EM':>8s} {'F1':>8s} {'METEOR':>8s} " \
             f"{'B-1':>8s} {'B-2':>8s} {'B-3':>8s} {'B-4':>8s}"
    print("\n" + "=" * 110)
    print(header)
    print("-" * 110)
    for name, m in test_results.items():
        print(f"{name:<30s} {m['accuracy']:>8.4f} {m['em']:>8.4f} {m['f1']:>8.4f} "
              f"{m['meteor']:>8.4f} {m['bleu1']:>8.4f} {m['bleu2']:>8.4f} "
              f"{m['bleu3']:>8.4f} {m['bleu4']:>8.4f}")
    print("=" * 110)

    best_name = max(test_results, key=lambda k: test_results[k]["f1"])
    print(f"\n★ Best model: {best_name}  (F1 = {test_results[best_name]['f1']:.4f})")

    # ── Visualize ──
    if len(test_results) > 1:
        plot_radar_chart(test_results)
        plot_bar_chart(test_results)

    # ── Question-type analysis ──
    if args.question_type_analysis:
        best_data = all_eval_data[best_name]
        qtype_results = evaluate_by_question_type(
            best_data["preds"], best_data["refs"], best_data["questions"]
        )
        print("\n" + "=" * 70)
        print(f"QUESTION TYPE ANALYSIS ({best_name})")
        print("-" * 70)
        for qtype, stats in qtype_results.items():
            print(f"  {qtype:>10s}: EM={stats['em']:.3f}  F1={stats['f1']:.3f}  "
                  f"METEOR={stats['meteor']:.3f}  (n={stats['total']:>4d})")

        plot_question_type_analysis(qtype_results)
        plot_confusion_matrix(
            best_data["preds"], best_data["refs"], best_data["questions"]
        )

    # ── Failure analysis ──
    if args.failure_analysis:
        best_data = all_eval_data[best_name]
        failures = get_failure_cases(
            best_data["preds"], best_data["refs"],
            best_data["questions"], n=args.num_failures,
        )
        print(f"\n{'=' * 80}")
        print(f"TOP {len(failures)} WORST PREDICTIONS ({best_name})")
        print(f"{'-' * 80}")
        for i, f in enumerate(failures):
            print(f"  [{i+1}] Type: {f['type']}")
            print(f"       Q: {f['question']}")
            print(f"       Pred: {f['prediction']}")
            print(f"       Ref:  {f['reference']}")
            print(f"       F1:   {f['f1']:.3f}")
            print()


if __name__ == "__main__":
    main()
