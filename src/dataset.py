"""
VQA Dataset and Vocabulary classes.

This module provides data loading and preprocessing utilities for the VQA task.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional


class Vocabulary:
    """
    Vocabulary class for managing word-to-index and index-to-word mappings.
    
    Handles tokenization and encoding of questions and answers.
    """
    
    def __init__(self, freq_threshold: int = 5):
        """
        Initialize Vocabulary.
        
        Args:
            freq_threshold: Minimum word frequency to include in vocabulary
        """
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_freq = {}
        self.idx = 4
        
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def build_vocabulary(self, sentences: List[str]) -> None:
        """
        Build vocabulary from a list of sentences.
        
        Args:
            sentences: List of text sentences to build vocabulary from
        """
        # Count word frequencies
        for sentence in sentences:
            for word in self.tokenize(sentence):
                if word not in self.word_freq:
                    self.word_freq[word] = 0
                self.word_freq[word] += 1
        
        # Add words that meet frequency threshold
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens (words)
        """
        # Basic tokenization - can be improved with NLTK
        return text.lower().strip().split()
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to indices.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (None for no padding/truncation)
            
        Returns:
            List of word indices
        """
        tokens = self.tokenize(text)
        encoded = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        
        if max_length:
            if len(encoded) < max_length:
                encoded += [self.word2idx["<PAD>"]] * (max_length - len(encoded))
            else:
                encoded = encoded[:max_length]
        
        return encoded
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode indices back to text.
        
        Args:
            indices: List of word indices
            
        Returns:
            Decoded text string
        """
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices]
        # Remove special tokens
        words = [w for w in words if w not in ["<PAD>", "<START>", "<END>"]]
        return " ".join(words)


class VQADataset(Dataset):
    """
    PyTorch Dataset for Visual Question Answering.
    
    Loads images, questions, and answers, applying appropriate preprocessing.
    
    Expected data format:
        - Images: COCO-style image files
        - Questions: JSON with format [{"image_id": int, "question": str, "question_id": int}]
        - Annotations: JSON with format [{"question_id": int, "answers": [{"answer": str}]}]
    """
    
    def __init__(
        self,
        image_dir: str,
        questions_file: str,
        annotations_file: Optional[str] = None,
        vocab: Optional[Vocabulary] = None,
        transform: Optional[callable] = None,
        max_question_length: int = 20,
        answer_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize VQA Dataset.
        
        Args:
            image_dir: Directory containing images
            questions_file: Path to questions JSON file
            annotations_file: Path to annotations JSON file (None for test set)
            vocab: Vocabulary object for encoding text
            transform: Torchvision transforms for image preprocessing
            max_question_length: Maximum question length for padding/truncation
            answer_to_idx: Dictionary mapping answers to class indices
        """
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.max_question_length = max_question_length
        self.answer_to_idx = answer_to_idx
        
        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
            self.questions = questions_data.get('questions', [])
        
        # Load annotations if provided
        self.annotations = None
        if annotations_file:
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
                # Create mapping from question_id to annotation
                self.annotations = {
                    ann['question_id']: ann 
                    for ann in annotations_data.get('annotations', [])
                }
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of sample
            
        Returns:
            Dictionary containing:
                - image: Preprocessed image tensor, shape (3, H, W)
                - question: Encoded question tensor, shape (max_question_length,)
                - answer: Answer class index (if annotations available), shape ()
                - question_id: Question ID for reference
        """
        question_data = self.questions[idx]
        question_id = question_data['question_id']
        image_id = question_data['image_id']
        question_text = question_data['question']
        
        # Load and preprocess image
        # Note: Adjust image filename format based on your dataset (COCO uses 12-digit format)
        image_path = f"{self.image_dir}/COCO_train2014_{image_id:012d}.jpg"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Encode question
        if self.vocab:
            question_encoded = torch.tensor(
                self.vocab.encode(question_text, self.max_question_length),
                dtype=torch.long
            )
        else:
            question_encoded = torch.tensor([])
        
        # Prepare output
        sample = {
            'image': image,
            'question': question_encoded,
            'question_id': question_id,
        }
        
        # Add answer if annotations available
        if self.annotations and question_id in self.annotations:
            annotation = self.annotations[question_id]
            # Get most common answer (VQA uses consensus of 10 annotators)
            answers = [ans['answer'] for ans in annotation['answers']]
            # For now, take the first answer - can be improved with voting
            answer_text = answers[0] if answers else ""
            
            if self.answer_to_idx and answer_text in self.answer_to_idx:
                answer_idx = self.answer_to_idx[answer_text]
                sample['answer'] = torch.tensor(answer_idx, dtype=torch.long)
        
        return sample
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dictionary with:
                - images: shape (batch_size, 3, H, W)
                - questions: shape (batch_size, max_question_length)
                - answers: shape (batch_size,) if available
                - question_ids: List of question IDs
        """
        images = torch.stack([item['image'] for item in batch])
        questions = torch.stack([item['question'] for item in batch])
        question_ids = [item['question_id'] for item in batch]
        
        batched = {
            'images': images,
            'questions': questions,
            'question_ids': question_ids,
        }
        
        # Add answers if available
        if 'answer' in batch[0]:
            answers = torch.stack([item['answer'] for item in batch])
            batched['answers'] = answers
        
        return batched


def build_answer_vocab(annotations_file: str, top_k: int = 1000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build answer vocabulary from top-k most frequent answers.
    
    VQA is typically formulated as classification over most common answers.
    
    Args:
        annotations_file: Path to annotations JSON file
        top_k: Number of most frequent answers to include
        
    Returns:
        Tuple of (answer_to_idx, idx_to_answer) dictionaries
    """
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
        annotations = annotations_data.get('annotations', [])
    
    # Count answer frequencies
    answer_freq = {}
    for ann in annotations:
        for answer_data in ann['answers']:
            answer = answer_data['answer']
            answer_freq[answer] = answer_freq.get(answer, 0) + 1
    
    # Get top-k answers
    top_answers = sorted(answer_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Create mappings
    answer_to_idx = {answer: idx for idx, (answer, _) in enumerate(top_answers)}
    idx_to_answer = {idx: answer for answer, idx in answer_to_idx.items()}
    
    return answer_to_idx, idx_to_answer
