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

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

class Vocabulary:
    def __init__(self, freq_threshold: int = 3) -> None:
        self.itos: dict[int, str] = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi: dict[str, int] = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self) -> int: return len(self.itos)

    @staticmethod
    def tokenizer(text: str) -> list[str]:
        return normalize_answer(text).split()

    def build_vocabulary(self, sentence_list: list[str]) -> None:
        frequencies: Counter = Counter()
        idx = len(self.itos)
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> list[int]:
        return [self.stoi.get(tok, UNK_IDX) for tok in self.tokenizer(text)]

class AOKVQA_Dataset(Dataset):
    def __init__(self, data_list: list[dict[str, Any]], question_vocab: Vocabulary, answer_vocab: Vocabulary, transform: Optional[Any] = None) -> None:
        self.data = data_list
        self.question_vocab = question_vocab
        self.answer_vocab = answer_vocab
        self.transform = transform

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
        item = self.data[index]
        try:
            image = item["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = image.convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        
        if self.transform: image = self.transform(image)

        q_str = item.get("question", "")
        q_vec = [SOS_IDX] + self.question_vocab.numericalize(q_str) + [EOS_IDX]
        answer_text = extract_answer(item)
        ans_vec = [SOS_IDX] + self.answer_vocab.numericalize(answer_text) + [EOS_IDX]

        refs = item.get("rationales", [])
        if not refs:
            refs = item.get("direct_answers", [answer_text])

        return image, torch.tensor(q_vec), torch.tensor(ans_vec), refs, q_str

def collate_fn(batch):
    images, questions, answers, answer_texts, raw_qs = zip(*batch)
    images = torch.stack(images, 0)
    q_lengths = torch.tensor([len(q) for q in questions])
    a_lengths = torch.tensor([len(a) for a in answers])
    questions = pad_sequence(list(questions), batch_first=True, padding_value=PAD_IDX)
    answers = pad_sequence(list(answers), batch_first=True, padding_value=PAD_IDX)
    return images, questions, q_lengths, answers, a_lengths, list(answer_texts), list(raw_qs)

# Giải mã chuỗi chỉ số thành câu văn bản
def decode_sequence(seq, vocab):
    """Giải mã một chuỗi chỉ số thành câu văn bản."""
    words = []
    for idx in seq:
        word = vocab.itos.get(idx, "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<SOS>"):
            words.append(word)
    return " ".join(words)