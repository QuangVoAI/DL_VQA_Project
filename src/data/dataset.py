"""Dataset and Vocabulary classes for VQA."""

from __future__ import annotations

import io
from collections import Counter
from typing import Any, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.preprocessing import normalize_answer, extract_answer

# Special token indices
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class Vocabulary:
    """Word ↔ index mapping with frequency thresholding.

    Special tokens: <PAD>=0, <SOS>=1, <EOS>=2, <UNK>=3.
    """

    def __init__(self, freq_threshold: int = 2) -> None:
        self.itos: dict[int, str] = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi: dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenizer(text: str) -> list[str]:
        """Simple whitespace tokenizer with normalization."""
        return normalize_answer(text).split()

    def build_vocabulary(self, sentence_list: list[str]) -> None:
        """Build vocab from a list of sentences, keeping words with freq ≥ threshold."""
        frequencies: Counter = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                if word not in self.stoi:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> list[int]:
        """Convert text → list of token indices."""
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in self.tokenizer(text)]


class AOKVQA_Dataset(Dataset):
    """A-OKVQA dataset: each sample → (image_tensor, question_ids, answer_ids, answer_text).

    Args:
        data_list: Raw dataset items (dicts with 'image', 'question', etc.).
        question_vocab: Vocabulary for question tokenization.
        answer_vocab: Vocabulary for answer tokenization.
        transform: torchvision transform for images.
    """

    def __init__(
        self,
        data_list: list[dict[str, Any]],
        question_vocab: Vocabulary,
        answer_vocab: Vocabulary,
        transform: Optional[Any] = None,
    ) -> None:
        self.data = data_list
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        item = self.data[index]

        # ── Image ──
        try:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = image.convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        # ── Question ──
        question = item.get("question", "")
        q_vec = (
            [self.question_vocab.stoi["<SOS>"]]
            + self.question_vocab.numericalize(question)
            + [self.question_vocab.stoi["<EOS>"]]
        )

        # ── Answer ──
        answer = extract_answer(item)
        ans_vec = (
            [self.answer_vocab.stoi["<SOS>"]]
            + self.answer_vocab.numericalize(answer)
            + [self.answer_vocab.stoi["<EOS>"]]
        )

        return image, torch.tensor(q_vec), torch.tensor(ans_vec), answer


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Collate function: pad questions & answers, stack images.

    Returns:
        images, questions, q_lengths, answers, a_lengths, answer_texts
    """
    images, questions, answers, answer_texts = zip(*batch)
    images = torch.stack(images, 0)
    q_lengths = torch.tensor([len(q) for q in questions])
    a_lengths = torch.tensor([len(a) for a in answers])
    questions = pad_sequence(list(questions), batch_first=True, padding_value=PAD_IDX)
    answers = pad_sequence(list(answers), batch_first=True, padding_value=PAD_IDX)
    return images, questions, q_lengths, answers, a_lengths, list(answer_texts)
