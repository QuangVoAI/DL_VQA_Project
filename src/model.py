"""
VQA Model Module

This module contains the CNN-LSTM architecture for Visual Question Answering,
including the encoder (CNN) and decoder (RNN) components.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN Encoder for extracting visual features from images.
    
    Uses a pretrained ResNet as the backbone to extract image features.
    
    Input shape:
        - images: (batch_size, 3, H, W) where H, W are image dimensions
    
    Output shape:
        - features: (batch_size, feature_dim) where feature_dim is the encoded feature size
    """
    
    def __init__(self, feature_dim: int = 2048, pretrained: bool = True, 
                 freeze_backbone: bool = True):
        """
        Initialize CNN Encoder.
        
        Args:
            feature_dim (int): Dimension of output features (default: 2048 for ResNet)
            pretrained (bool): Whether to use pretrained ResNet weights
            freeze_backbone (bool): Whether to freeze backbone weights during training
        """
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.feature_dim = feature_dim
        
        # Optional: Add a projection layer to change feature dimension
        # Uncomment if you want a different feature dimension than 2048
        # self.projection = nn.Linear(2048, feature_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract image features.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Image features of shape (batch_size, feature_dim)
        """
        with torch.set_grad_enabled(not self._is_frozen()):
            features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        
        features = features.reshape(features.size(0), -1)  # (batch_size, 2048)
        
        # Optional projection
        # features = self.projection(features)
        
        return features
    
    def _is_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return not next(self.resnet.parameters()).requires_grad
    
    def unfreeze(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True


class DecoderRNN(nn.Module):
    """
    RNN Decoder that takes image features and question embeddings to generate answers.
    
    Uses LSTM to process the concatenated image and question features and 
    generates answer tokens sequentially.
    
    Input shapes:
        - image_features: (batch_size, image_feature_dim)
        - question_encoded: (batch_size, question_length, embed_dim) or (batch_size, question_length)
    
    Output shapes:
        - outputs: (batch_size, max_answer_length, vocab_size) - logits for each token
        - hidden: Tuple of (h_n, c_n) - final hidden states
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 image_feature_dim: int = 2048,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Initialize RNN Decoder.
        
        Args:
            vocab_size (int): Size of vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of LSTM hidden state
            image_feature_dim (int): Dimension of image features from encoder
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super(DecoderRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.image_feature_dim = image_feature_dim
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Project image features to match LSTM input dimension
        self.image_projection = nn.Linear(image_feature_dim, hidden_dim)
        
        # LSTM for processing question and generating answer
        # Input: concatenated image features and question embeddings
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection to vocabulary
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                image_features: torch.Tensor, 
                questions: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Forward pass to generate answer logits.
        
        Args:
            image_features (torch.Tensor): Image features of shape (batch_size, image_feature_dim)
            questions (torch.Tensor): Question token indices of shape (batch_size, question_length)
            hidden (tuple, optional): Initial hidden state (h_0, c_0)
            
        Returns:
            tuple: (outputs, hidden) where
                - outputs: Tensor of shape (batch_size, question_length, vocab_size)
                - hidden: Tuple of (h_n, c_n) final hidden states
        """
        batch_size = questions.size(0)
        
        # Embed questions
        # Input: (batch_size, question_length)
        # Output: (batch_size, question_length, embed_dim)
        embedded = self.embedding(questions)
        embedded = self.dropout(embedded)
        
        # Project image features and use as initial hidden state
        if hidden is None:
            # Project image features to hidden dimension
            image_hidden = self.image_projection(image_features)  # (batch_size, hidden_dim)
            image_hidden = torch.tanh(image_hidden)
            
            # Initialize hidden state with image features
            # h_0: (num_layers, batch_size, hidden_dim)
            h_0 = image_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c_0 = torch.zeros_like(h_0)
            hidden = (h_0, c_0)
        
        # Pass through LSTM
        # Input: (batch_size, question_length, embed_dim)
        # Output: (batch_size, question_length, hidden_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Project to vocabulary size
        # Input: (batch_size, question_length, hidden_dim)
        # Output: (batch_size, question_length, vocab_size)
        outputs = self.fc_out(self.dropout(lstm_out))
        
        return outputs, hidden
    
    def generate(self, 
                 image_features: torch.Tensor,
                 start_token: int,
                 end_token: int,
                 max_length: int = 20,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate answer tokens autoregressively.
        
        Args:
            image_features (torch.Tensor): Image features of shape (batch_size, image_feature_dim)
            start_token (int): Index of start token
            end_token (int): Index of end token
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            torch.Tensor: Generated token indices of shape (batch_size, generated_length)
        """
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Initialize with start token
        current_token = torch.full((batch_size, 1), start_token, 
                                   dtype=torch.long, device=device)
        generated_tokens = [current_token]
        
        # Initialize hidden state with image features
        image_hidden = self.image_projection(image_features)
        image_hidden = torch.tanh(image_hidden)
        h_0 = image_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        hidden = (h_0, c_0)
        
        # Generate tokens one by one
        for _ in range(max_length - 1):
            # Embed current token
            embedded = self.embedding(current_token)  # (batch_size, 1, embed_dim)
            
            # LSTM forward
            lstm_out, hidden = self.lstm(embedded, hidden)
            
            # Get logits for next token
            logits = self.fc_out(lstm_out[:, -1, :])  # (batch_size, vocab_size)
            
            # Sample next token with temperature
            probs = torch.softmax(logits / temperature, dim=-1)
            current_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            generated_tokens.append(current_token)
            
            # Check if all sequences have generated end token
            if (current_token == end_token).all():
                break
        
        # Concatenate all generated tokens
        generated_sequence = torch.cat(generated_tokens, dim=1)  # (batch_size, generated_length)
        
        return generated_sequence


class VQAModel(nn.Module):
    """
    Complete VQA model combining CNN encoder and RNN decoder.
    
    Input shapes:
        - images: (batch_size, 3, H, W)
        - questions: (batch_size, question_length)
    
    Output shapes:
        - outputs: (batch_size, question_length, vocab_size) - answer logits
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 image_feature_dim: int = 2048,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 pretrained_encoder: bool = True,
                 freeze_encoder: bool = True):
        """
        Initialize VQA Model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embed_dim (int): Dimension of word embeddings
            hidden_dim (int): Dimension of LSTM hidden state
            image_feature_dim (int): Dimension of image features
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            pretrained_encoder (bool): Whether to use pretrained CNN encoder
            freeze_encoder (bool): Whether to freeze encoder weights
        """
        super(VQAModel, self).__init__()
        
        self.encoder = EncoderCNN(
            feature_dim=image_feature_dim,
            pretrained=pretrained_encoder,
            freeze_backbone=freeze_encoder
        )
        
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete VQA model.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, H, W)
            questions (torch.Tensor): Question tokens of shape (batch_size, question_length)
            
        Returns:
            torch.Tensor: Answer logits of shape (batch_size, question_length, vocab_size)
        """
        # Extract image features
        image_features = self.encoder(images)  # (batch_size, image_feature_dim)
        
        # Generate answer logits
        outputs, _ = self.decoder(image_features, questions)  # (batch_size, seq_len, vocab_size)
        
        return outputs
    
    def generate_answer(self, 
                       images: torch.Tensor,
                       start_token: int,
                       end_token: int,
                       max_length: int = 20) -> torch.Tensor:
        """
        Generate answer for given images.
        
        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, H, W)
            start_token (int): Index of start token
            end_token (int): Index of end token
            max_length (int): Maximum answer length
            
        Returns:
            torch.Tensor: Generated answer tokens of shape (batch_size, generated_length)
        """
        # Extract image features
        image_features = self.encoder(images)
        
        # Generate answer
        answer_tokens = self.decoder.generate(
            image_features, 
            start_token, 
            end_token, 
            max_length
        )
        
        return answer_tokens
