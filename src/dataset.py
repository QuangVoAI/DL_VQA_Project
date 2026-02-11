"""
VQA Dataset Module

This module contains classes for handling Visual Question Answering datasets,
including vocabulary management and dataset loading.
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset
from PIL import Image
import nltk


class Vocabulary:
    """
    Vocabulary class for tokenizing and managing word-to-index mappings.
    
    This class builds a vocabulary from text data and provides methods to convert
    between words and indices.
    
    Attributes:
        word2idx (dict): Mapping from words to indices
        idx2word (dict): Mapping from indices to words
        pad_token (str): Padding token
        unk_token (str): Unknown token
        start_token (str): Start of sequence token
        end_token (str): End of sequence token
    """
    
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>', 
                 start_token='<START>', end_token='<END>'):
        """
        Initialize vocabulary with special tokens.
        
        Args:
            pad_token (str): Padding token
            unk_token (str): Unknown token
            start_token (str): Start of sequence token
            end_token (str): End of sequence token
        """
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        
        self.word2idx = {}
        self.idx2word = {}
        
        # Add special tokens first
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.pad_token, self.unk_token, 
                         self.start_token, self.end_token]
        for token in special_tokens:
            self._add_word(token)
    
    def _add_word(self, word: str):
        """
        Add a word to the vocabulary.
        
        Args:
            word (str): Word to add
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocabulary(self, sentences: List[str], min_word_freq: int = 5):
        """
        Build vocabulary from a list of sentences.
        
        Args:
            sentences (List[str]): List of sentences to build vocabulary from
            min_word_freq (int): Minimum word frequency to include in vocabulary
        """
        word_counter = Counter()
        
        # Tokenize and count words
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            word_counter.update(tokens)
        
        # Add words that meet minimum frequency
        for word, freq in word_counter.items():
            if freq >= min_word_freq:
                self._add_word(word.lower())
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text using NLTK word tokenizer.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        try:
            tokens = nltk.word_tokenize(text.lower())
        except LookupError:
            # Download NLTK data if not available
            nltk.download('punkt', quiet=True)
            tokens = nltk.word_tokenize(text.lower())
        return tokens
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Convert text to list of indices.
        
        Args:
            text (str): Text to encode
            max_length (Optional[int]): Maximum sequence length (truncate or pad)
            
        Returns:
            List[int]: List of word indices
        """
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx[self.unk_token]) 
                   for token in tokens]
        
        if max_length is not None:
            if len(indices) < max_length:
                # Pad sequence
                indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
            else:
                # Truncate sequence
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert list of indices back to text.
        
        Args:
            indices (List[int]): List of word indices
            skip_special_tokens (bool): Whether to skip special tokens in output
            
        Returns:
            str: Decoded text
        """
        special_tokens = {self.pad_token, self.unk_token, 
                         self.start_token, self.end_token}
        
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """
        Save vocabulary to file.
        
        Args:
            filepath (str): Path to save vocabulary
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'start_token': self.start_token,
                'end_token': self.end_token
            }
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load vocabulary from file.
        
        Args:
            filepath (str): Path to vocabulary file
            
        Returns:
            Vocabulary: Loaded vocabulary object
        """
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        special_tokens = vocab_data['special_tokens']
        vocab = cls(
            pad_token=special_tokens['pad_token'],
            unk_token=special_tokens['unk_token'],
            start_token=special_tokens['start_token'],
            end_token=special_tokens['end_token']
        )
        
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        
        return vocab


class VQADataset(Dataset):
    """
    PyTorch Dataset for Visual Question Answering.
    
    This dataset handles loading images, questions, and answers for VQA tasks.
    
    Input shapes:
        - Image: PIL Image or Tensor of shape (3, H, W) after transforms
        - Question: String or Tensor of shape (max_question_length,) after encoding
        - Answer: String or Tensor of shape (max_answer_length,) after encoding
    
    Output shapes:
        - image: Tensor of shape (3, H, W)
        - question: Tensor of shape (max_question_length,)
        - answer: Tensor of shape (max_answer_length,) or class index
        - metadata: Dict with additional information
    """
    
    def __init__(self, 
                 image_dir: str,
                 questions_file: str,
                 answers_file: str,
                 vocab: Vocabulary,
                 transform=None,
                 max_question_length: int = 20,
                 max_answer_length: int = 10):
        """
        Initialize VQA Dataset.
        
        Args:
            image_dir (str): Directory containing images
            questions_file (str): Path to questions JSON file
            answers_file (str): Path to answers JSON file
            vocab (Vocabulary): Vocabulary object for tokenization
            transform: Optional image transforms (torchvision.transforms)
            max_question_length (int): Maximum question length
            max_answer_length (int): Maximum answer length
        """
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        
        # Load questions and answers
        with open(questions_file, 'r') as f:
            self.questions_data = json.load(f)
        
        with open(answers_file, 'r') as f:
            self.answers_data = json.load(f)
        
        # Build sample list (to be implemented based on dataset format)
        self._build_samples()
    
    def _build_samples(self):
        """
        Build list of samples from questions and answers data.
        
        This method should be customized based on your dataset format.
        Expected format:
            questions_data: {'questions': [{'question_id': ..., 'image_id': ..., 'question': ...}]}
            answers_data: {'annotations': [{'question_id': ..., 'answer': ...}]}
        """
        # TODO: Customize based on your VQA dataset format
        # Example implementation:
        self.samples = []
        
        # This is a placeholder - adapt to your dataset structure
        if isinstance(self.questions_data, dict) and 'questions' in self.questions_data:
            questions = self.questions_data['questions']
            
            # Build answer lookup
            answer_lookup = {}
            if isinstance(self.answers_data, dict) and 'annotations' in self.answers_data:
                for ann in self.answers_data['annotations']:
                    answer_lookup[ann['question_id']] = ann['answer']
            
            # Build samples
            for q in questions:
                question_id = q['question_id']
                if question_id in answer_lookup:
                    self.samples.append({
                        'question_id': question_id,
                        'image_id': q['image_id'],
                        'question': q['question'],
                        'answer': answer_lookup[question_id]
                    })
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of sample
            
        Returns:
            Tuple containing:
                - image (torch.Tensor): Image tensor of shape (3, H, W)
                - question (torch.Tensor): Encoded question of shape (max_question_length,)
                - answer (torch.Tensor): Encoded answer of shape (max_answer_length,)
                - metadata (Dict): Additional information (question_id, image_id, etc.)
        """
        sample = self.samples[idx]
        
        # Load image
        image_filename = f"{sample['image_id']}.jpg"  # Adjust based on your format
        image_path = os.path.join(self.image_dir, image_filename)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            # Fallback: try different extensions
            for ext in ['.png', '.jpeg', '.JPEG', '.JPG']:
                alt_path = os.path.join(self.image_dir, f"{sample['image_id']}{ext}")
                if os.path.exists(alt_path):
                    image = Image.open(alt_path).convert('RGB')
                    break
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode question
        question_indices = self.vocab.encode(sample['question'], 
                                            max_length=self.max_question_length)
        question_tensor = torch.tensor(question_indices, dtype=torch.long)
        
        # Encode answer
        answer_indices = self.vocab.encode(sample['answer'], 
                                          max_length=self.max_answer_length)
        answer_tensor = torch.tensor(answer_indices, dtype=torch.long)
        
        # Metadata
        metadata = {
            'question_id': sample['question_id'],
            'image_id': sample['image_id'],
            'question_text': sample['question'],
            'answer_text': sample['answer']
        }
        
        return image, question_tensor, answer_tensor, metadata
