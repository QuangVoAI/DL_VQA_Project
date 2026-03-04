"""
Advanced adapters for VQA:
1. BertQuestionEncoder (Using Contextual Embeddings from BERT)
2. BUTD_FasterRCNN_Encoder (Extract Region Features from Faster R-CNN)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class BertQuestionEncoder(nn.Module):
    """
    Bert will read the entire question to encode contextually (Contextual Embeddings).
    """
    def __init__(self, hidden_size: int = 512, dropout: float = 0.3):
        super().__init__()
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            
            # Freeze BERT blocks to save VRAM and speed up training (exploit pre-trained)
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # Project from 768-dimensional space of DistilBERT to hidden_size of Decoder
            self.proj = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout)
            )
        except ImportError:
            raise ImportError("Lỗi! Để dùng BERT, hãy chạy lệnh: pip install transformers")

    def forward(self, raw_questions: list[str], device: torch.device) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Tokenize directly when Text array runs through forward
        encoded = self.tokenizer(raw_questions, padding=True, truncation=True, return_tensors='pt', max_length=50)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Run through BERT
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Project to hidden_size of Decoder
        projected = self.proj(hidden_states) # (B, seq_len, hidden_size)
        
        # Take first token (CLS token proxy in DistilBert) as semantic vector for (h, c) of LSTM Decoder
        cls_feat = projected[:, 0, :] # (B, hidden_size)
        
        # Mock LSTM structure (h, c) 2 layers from old LSTM to perfectly fit with AnswerDecoder
        # Required size: (num_layers, B, hidden_size) -> (2, B, 512)
        h = cls_feat.unsqueeze(0).repeat(2, 1, 1).contiguous()
        c = torch.zeros_like(h).to(device)
        
        return projected, (h, c), attention_mask


class BUTD_FasterRCNN_Encoder(nn.Module):
    """
    Bottom-Up Top-Down (BUTD) thay thế cho ResNet-50.
    Sử dụng Faster R-CNN để khoanh vùng 36 vật thể đáng chú ý nhất thay vì chia ảnh thành không gian lưới 7x7 ngẫu nhiên.
    """
    def __init__(self, out_dim: int = 512, max_regions: int = 36):
        super().__init__()
        import torchvision
        # Load Faster R-CNN model
        self.faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        self.faster_rcnn.eval()
        
        # Freeze parameters
        for param in self.faster_rcnn.parameters():
            param.requires_grad = False
            
        self.max_regions = max_regions
        self.proj = nn.Linear(1024, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Extracting RoI (Region of Interest) directly from Torchvision is extremely expensive in terms of VRAM and training time
        due to the fact that the number of bounding boxes on each image is different.
        Standard SOTA: You must extract (Extract Offline) in advance into numpy arrays (B, 36, 1024) 
        save to disk, then during training just read it! 
        """
        raise NotImplementedError("To run full BUTD Pipeline smoothly, you should extract features offline (into .npy/.h5 files) on Kaggle before training the LSTM.")
