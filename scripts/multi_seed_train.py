"""Multi-seed training for statistical significance.

Run the same experiment across multiple seeds and report mean ± std.
Optionally runs paired t-test between model variants.

Usage:
    python scripts/multi_seed_train.py --config configs/default.yaml --seeds 42 123 456
    python scripts/multi_seed_train.py --config configs/default.yaml --n-seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
import random

from src.config import Config
from src.data.preprocessing import extract_answer, expand_data_with_rationales
from src.data.dataset import Vocabulary, AOKVQA_Dataset, collate_fn
from src.data.glove import download_glove, load_glove_embeddings
from src.models.vqa_model import VQAModel
from src.engine.trainer import train_model
from src.engine.evaluator import evaluate_model
from src.utils.helpers import get_device, set_seed, setup_logging


def run_single_seed(
    cfg: Config,
    seed: int,
    device: torch.device,
    variants: list[str],
) -> dict[str, dict[str, float]]:
    """Run full training + evaluation for one seed.

    Returns:
        Dict mapping model_name → metrics dict.
    """
    set_seed(seed)
    logger = setup_logging(cfg.log_dir)
    logger.info(f"\n{'#' * 70}")
    logger.info(f"  SEED = {seed}")
    logger.info(f"{'#' * 70}")

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load data
    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Build vocab
    all_questions, all_answers = [], []
    for item in hf_train:
        all_questions.append(item["question"])
        rationales = item.get("rationales", [])
        if rationales:
            all_answers.extend(rationales)
        else:
            all_answers.append(extract_answer(item))
    for item in hf_val:
        all_questions.append(item["question"])
        rationales = item.get("rationales", [])
        if rationales:
            all_answers.extend(rationales)
        else:
            all_answers.append(extract_answer(item))

    question_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    question_vocab.build_vocabulary(all_questions)
    answer_vocab = Vocabulary(freq_threshold=cfg.data.freq_threshold)
    answer_vocab.build_vocabulary(all_answers)

    # Split with this seed
    hf_train_list = list(hf_train)
    random.shuffle(hf_train_list)
    split_idx = int(len(hf_train_list) * cfg.data.train_ratio)
    train_data_raw = hf_train_list[:split_idx]
    val_data = hf_train_list[split_idx:]
    test_data = list(hf_val)

    train_data = expand_data_with_rationales(train_data_raw) if cfg.data.expand_rationales else train_data_raw

    train_dataset = AOKVQA_Dataset(train_data, question_vocab, answer_vocab, transform_train)
    val_dataset = AOKVQA_Dataset(val_data, question_vocab, answer_vocab, transform_eval)
    test_dataset = AOKVQA_Dataset(test_data, question_vocab, answer_vocab, transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0, pin_memory=False)

    # GloVe
    download_glove()
    q_glove = load_glove_embeddings(question_vocab)
    a_glove = load_glove_embeddings(answer_vocab)

    # Save vocab
    os.makedirs("data/processed", exist_ok=True)
    torch.save({"question_vocab": question_vocab, "answer_vocab": answer_vocab},
               "data/processed/vocab_aokvqa.pth")

    seed_results = {}
    for name in variants:
        variant_cfg = cfg.model_variants[name]
        model = VQAModel(
            len(question_vocab), len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            q_pretrained_emb=q_glove, a_pretrained_emb=a_glove,
            **variant_cfg,
        ).to(device)

        train_model(
            model=model, name=f"{name}_seed{seed}",
            train_loader=train_loader, val_loader=val_loader,
            answer_vocab=answer_vocab, device=device,
            epochs=cfg.train.epochs, lr=cfg.train.learning_rate,
            ckpt_dir=cfg.ckpt_dir,
        )

        eval_data = evaluate_model(
            model, test_loader, answer_vocab, question_vocab,
            device, cfg.ckpt_dir, f"{name}_seed{seed}", cfg.train.beam_width,
        )
        seed_results[name] = eval_data["metrics"]

    return seed_results


def aggregate_results(
    all_seed_results: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate metrics across seeds: compute mean ± std.

    Returns:
        Dict mapping model_name → metric_name → {'mean': float, 'std': float}.
    """
    model_names = all_seed_results[0].keys()
    aggregated = {}

    for name in model_names:
        metric_values: dict[str, list[float]] = defaultdict(list)
        for seed_result in all_seed_results:
            for metric, value in seed_result[name].items():
                metric_values[metric].append(value)

        aggregated[name] = {}
        for metric, values in metric_values.items():
            aggregated[name][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    return aggregated


def paired_ttest(
    results_a: list[float],
    results_b: list[float],
) -> dict[str, float]:
    """Paired t-test between two model variants.

    Returns:
        {'t_stat': float, 'p_value': float, 'significant': bool}
    """
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed VQA training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds to run (generates seeds automatically)")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--ttest", action="store_true",
                        help="Run paired t-test between best and second-best models")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    variants = args.models or list(cfg.model_variants.keys())

    seeds = args.seeds or [42 + i * 100 for i in range(args.n_seeds)]
    print(f"Running {len(seeds)} seeds: {seeds}")
    print(f"Variants: {variants}")

    all_seed_results = []
    for seed in seeds:
        result = run_single_seed(cfg, seed, device, variants)
        all_seed_results.append(result)

    # ── Aggregate ──
    agg = aggregate_results(all_seed_results)

    print("\n" + "=" * 100)
    print("MULTI-SEED RESULTS (mean ± std)")
    print("-" * 100)
    header = f"{'Model':<30s} {'Acc':>14s} {'EM':>14s} {'F1':>14s} {'METEOR':>14s}"
    print(header)
    print("-" * 100)

    for name in variants:
        a = agg[name]
        print(f"{name:<30s} "
              f"{a['accuracy']['mean']:.4f}±{a['accuracy']['std']:.4f} "
              f"{a['em']['mean']:.4f}±{a['em']['std']:.4f} "
              f"{a['f1']['mean']:.4f}±{a['f1']['std']:.4f} "
              f"{a['meteor']['mean']:.4f}±{a['meteor']['std']:.4f}")
    print("=" * 100)

    # ── Optional t-test ──
    if args.ttest and len(variants) >= 2:
        try:
            from scipy import stats

            # Sort by mean F1
            sorted_models = sorted(
                variants, key=lambda n: agg[n]["f1"]["mean"], reverse=True
            )
            best, second = sorted_models[0], sorted_models[1]

            f1_best = agg[best]["f1"]["values"]
            f1_second = agg[second]["f1"]["values"]

            result = paired_ttest(f1_best, f1_second)
            print(f"\nPaired t-test: {best} vs {second}")
            print(f"  t-statistic: {result['t_stat']:.4f}")
            print(f"  p-value:     {result['p_value']:.4f}")
            print(f"  Significant: {'YES (p < 0.05)' if result['significant'] else 'NO (p >= 0.05)'}")
        except ImportError:
            print("\nInstall scipy for t-test: pip install scipy")

    # ── Save results ──
    output = {
        "seeds": seeds,
        "aggregated": {name: {m: {"mean": v["mean"], "std": v["std"]}
                       for m, v in metrics.items()}
                       for name, metrics in agg.items()},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/multi_seed_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved → results/multi_seed_results.json")


if __name__ == "__main__":
    main()
