"""CLI script: Train all VQA model variants.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 10 --lr 1e-3
    python scripts/train.py --config configs/default.yaml --models M1_Scratch_NoAttn M4_Pretrained_Attn
"""

from __future__ import annotations

import argparse
import os
import sys
import random

# Add project root to path
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
from src.engine.trainer import train_model
from src.utils.helpers import get_device, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQA models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model variants to train (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    args = parser.parse_args()

    # ── Load config ──
    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.seed is not None:
        cfg.seed = args.seed

    # ── Setup ──
    set_seed(cfg.seed)
    logger = setup_logging(cfg.log_dir)
    device = get_device() if cfg.device == "auto" else torch.device(cfg.device)
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # ── Transforms ──
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

    # ── Load data ──
    logger.info("Loading A-OKVQA dataset ...")
    hf_train = load_dataset(cfg.data.hf_id, split="train")
    hf_val = load_dataset(cfg.data.hf_id, split="validation")

    # Build vocabularies
    all_questions, all_answers = [], []
    for item in tqdm(hf_train, desc="Collecting text"):
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
    logger.info(f"Vocab: Q={len(question_vocab)}, A={len(answer_vocab)}")

    # Split
    hf_train_list = list(hf_train)
    random.shuffle(hf_train_list)
    split_idx = int(len(hf_train_list) * cfg.data.train_ratio)
    train_data_raw = hf_train_list[:split_idx]
    val_data = hf_train_list[split_idx:]

    if cfg.data.expand_rationales:
        train_data = expand_data_with_rationales(train_data_raw)
        logger.info(f"Data expansion: {len(train_data_raw)} → {len(train_data)}")
    else:
        train_data = train_data_raw

    train_dataset = AOKVQA_Dataset(train_data, question_vocab, answer_vocab, transform_train)
    val_dataset = AOKVQA_Dataset(val_data, question_vocab, answer_vocab, transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=cfg.train.num_workers,
                              pin_memory=cfg.train.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=cfg.train.num_workers,
                            pin_memory=cfg.train.pin_memory)

    # GloVe
    download_glove()
    q_glove = load_glove_embeddings(question_vocab)
    a_glove = load_glove_embeddings(answer_vocab)

    # Save vocab
    os.makedirs("data/processed", exist_ok=True)
    torch.save({"question_vocab": question_vocab, "answer_vocab": answer_vocab},
               "data/processed/vocab_aokvqa.pth")

    # ── Train models ──
    variants = args.models or list(cfg.model_variants.keys())
    logger.info(f"Training variants: {variants}")

    for name in variants:
        if name not in cfg.model_variants:
            logger.warning(f"Unknown variant: {name}, skipping.")
            continue

        variant_cfg = cfg.model_variants[name]
        model = VQAModel(
            len(question_vocab), len(answer_vocab),
            embed_size=cfg.model.embed_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            q_pretrained_emb=q_glove,
            a_pretrained_emb=a_glove,
            **variant_cfg,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{name}: {n_params:,} trainable params")

        train_model(
            model=model, name=name,
            train_loader=train_loader, val_loader=val_loader,
            answer_vocab=answer_vocab, device=device,
            epochs=cfg.train.epochs, lr=cfg.train.learning_rate,
            ckpt_dir=cfg.ckpt_dir,
            label_smoothing=cfg.train.label_smoothing,
            patience=cfg.train.patience,
            grad_clip=cfg.train.grad_clip,
        )

    logger.info("All training complete!")


if __name__ == "__main__":
    main()
