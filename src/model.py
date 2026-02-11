"""
VQA Model Architecture.

This module implements the encoder-decoder architecture for Visual Question Answering:
- EncoderCNN: Extracts visual features from images using ResNet
- DecoderRNN: Processes questions and generates answers using LSTM
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class EncoderCNN(nn.Module):
    """
    CNN-based image encoder using pretrained ResNet.
    
    Extracts visual features from input images for the VQA task.
    """
    
    def __init__(self, embed_size: int = 512, pretrained: bool = True):
        """
        Initialize EncoderCNN.
        
        Args:
            embed_size: Dimension of output feature embeddings
            pretrained: Whether to use pretrained ResNet weights
        """
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        # ResNet outputs 2048-dim features from avgpool layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to project ResNet features to embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        # Freeze ResNet parameters for initial training
        # Can be unfrozen later for fine-tuning
        self.freeze_resnet()
    
    def freeze_resnet(self):
        """Freeze ResNet parameters to prevent updating during training."""
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def unfreeze_resnet(self):
        """Unfreeze ResNet parameters to allow fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            images: Input images, shape (batch_size, 3, H, W)
                   Expected H=W=224 for ResNet
        
        Returns:
            Image features, shape (batch_size, embed_size)
        """
        # Extract features from ResNet
        # Shape: (batch_size, 2048, 1, 1)
        with torch.no_grad():
            features = self.resnet(images)
        
        # Reshape: (batch_size, 2048)
        features = features.view(features.size(0), -1)
        
        # Project to embed_size
        # Shape: (batch_size, embed_size)
        features = self.fc(features)
        features = self.bn(features)
        
        return features


class DecoderRNN(nn.Module):
    """
    LSTM-based decoder for processing questions and generating answers.
    
    Combines visual features from EncoderCNN with question embeddings
    to predict answers.
    """
    
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        """
        Initialize DecoderRNN.
        
        Args:
            embed_size: Dimension of image features from encoder
            hidden_size: LSTM hidden state dimension
            vocab_size: Size of question vocabulary
            num_classes: Number of answer classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Word embedding for questions
        self.word_embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # LSTM for question encoding
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism placeholder (can be extended)
        # For baseline, we use simple concatenation
        
        # Fusion layer: combine image features and question features
        self.fusion = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier: predict answer from fused features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        questions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            image_features: Visual features from encoder, shape (batch_size, embed_size)
            questions: Encoded questions, shape (batch_size, seq_length)
        
        Returns:
            Answer logits, shape (batch_size, num_classes)
        """
        batch_size = questions.size(0)
        
        # Embed questions
        # Shape: (batch_size, seq_length, embed_size)
        question_embeds = self.word_embed(questions)
        question_embeds = self.dropout(question_embeds)
        
        # Encode questions with LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(question_embeds)
        
        # Use the last hidden state as question representation
        # Shape: (batch_size, hidden_size)
        question_features = hidden[-1]
        
        # Alternative: use mean pooling over sequence
        # question_features = lstm_out.mean(dim=1)
        
        # Fuse image and question features
        # Concatenate along feature dimension
        # Shape: (batch_size, embed_size + hidden_size)
        combined = torch.cat([image_features, question_features], dim=1)
        
        # Apply fusion layer
        # Shape: (batch_size, hidden_size)
        fused = self.fusion(combined)
        
        # Classify to get answer logits
        # Shape: (batch_size, num_classes)
        logits = self.classifier(fused)
        
        return logits


class VQAModel(nn.Module):
    """
    Complete VQA model combining EncoderCNN and DecoderRNN.
    
    This is the main model that takes images and questions as input
    and outputs answer predictions.
    """
    
    def __init__(
        self,
        embed_size: int = 512,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        num_classes: int = 1000,
        num_layers: int = 1,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        """
        Initialize VQA model.
        
        Args:
            embed_size: Dimension of embeddings
            hidden_size: LSTM hidden dimension
            vocab_size: Size of question vocabulary
            num_classes: Number of answer classes
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained: Use pretrained ResNet weights
        """
        super(VQAModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size, pretrained)
        self.decoder = DecoderRNN(
            embed_size,
            hidden_size,
            vocab_size,
            num_classes,
            num_layers,
            dropout,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        questions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the complete VQA model.
        
        Args:
            images: Input images, shape (batch_size, 3, H, W)
            questions: Encoded questions, shape (batch_size, seq_length)
        
        Returns:
            Answer logits, shape (batch_size, num_classes)
        """
        # Encode images
        # Shape: (batch_size, embed_size)
        image_features = self.encoder(images)
        
        # Decode to get answer logits
        # Shape: (batch_size, num_classes)
        logits = self.decoder(image_features, questions)
        
        return logits
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        self.encoder.freeze_resnet()
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        self.encoder.unfreeze_resnet()


def get_model(
    vocab_size: int,
    num_classes: int,
    embed_size: int = 512,
    hidden_size: int = 512,
    num_layers: int = 1,
    dropout: float = 0.5,
    pretrained: bool = True,
) -> VQAModel:
    """
    Factory function to create VQA model.
    
    Args:
        vocab_size: Size of question vocabulary
        num_classes: Number of answer classes
        embed_size: Dimension of embeddings
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        pretrained: Use pretrained ResNet weights
    
    Returns:
        Initialized VQA model
    """
    model = VQAModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        pretrained=pretrained,
    )
    return model
